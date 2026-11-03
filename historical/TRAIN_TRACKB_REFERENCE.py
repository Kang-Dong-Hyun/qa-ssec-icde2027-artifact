#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QA-SSEC Second Research
B01-B04 Track-B frozen 5-fold Small-LM experiment.

Track B = Same Token Budget / Budget-equivalent.
Each representation independently packs as many frozen-order metric records
as fit into the same 8064-token evidence payload per chunk.

Frozen resource caps:
  K = 1, 2, 4, 8 chunks

Total jobs:
  4 caps x 5 folds x 4 representations = 80

Scientific controls:
- Same local SmolLM2-135M base
- Same tokenizer/model context
- Same fold registry
- Same seed / epoch / optimizer / scoring
- Same frozen source metric order
- No target-aware or test-aware metric selection
- Test metrics are not printed until all 80 jobs are complete
- Resume uses deterministic DONE.json per job
"""

import argparse
import csv
import gc
import json
import math
import random
import shutil
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

VERSION = "QA_SSEC_B01_B04_SMOLLM2_TRACKB_5FOLD_V1"

MODEL_CONTEXT = 8192
EVIDENCE_BUDGET = 8064

CAPS = [1, 2, 4, 8]
FOLDS = [0, 1, 2, 3, 4]
SEED = 20260915
EPOCHS = 1
LR = 2e-5
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

VARIANTS = [
    "B01_EXT_EXACT",
    "B02_EXT_COMPACT",
    "B03_EXT_HMAC",
    "B04_EXT_ROLE",
]

TASKS = [
    ("ROOT_CAUSE_SERVICE", "root_cause_service"),
    ("FAULT", "fault"),
]

PROMPT_SUFFIX = {
    "ROOT_CAUSE_SERVICE": "\nTASK=ROOT_CAUSE_SERVICE\nANSWER=",
    "FAULT": "\nTASK=FAULT\nANSWER=",
}


def stable_json(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="raise")
        w.writeheader()
        w.writerows(rows)


def load_repr(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            uid = str(r["record_uid"])
            if uid in out:
                raise RuntimeError("DUPLICATE_UID:%s" % uid)
            out[uid] = json.loads(str(r["representation"]))
    return out


def metric_key(variant):
    return "metrics" if variant == "B01_EXT_EXACT" else "m"


def metric_count(obj, variant):
    return len(obj[metric_key(variant)])


def subset_obj(obj, variant, start, end):
    k = metric_key(variant)
    x = dict(obj)
    x[k] = obj[k][start:end]
    return x


class Counter:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.cache = {}

    def subset_tokens(self, variant, uid, start, end, obj):
        key = (variant, uid, start, end)
        if key not in self.cache:
            text = stable_json(subset_obj(obj, variant, start, end))
            self.cache[key] = self.tok.encode(text, add_special_tokens=False)
        return self.cache[key]

    def subset_len(self, variant, uid, start, end, obj):
        return len(self.subset_tokens(variant, uid, start, end, obj))


def greedy_variant_chunks(counter, variant, uid, obj):
    n = metric_count(obj, variant)
    chunks = []
    start = 0
    while start < n:
        best = None
        end = start + 1
        while end <= n:
            if counter.subset_len(variant, uid, start, end, obj) <= EVIDENCE_BUDGET:
                best = end
                end += 1
            else:
                break
        if best is None:
            raise RuntimeError(
                "SINGLE_METRIC_EXCEEDS_EVIDENCE_BUDGET:%s:%s:%d"
                % (variant, uid, start)
            )
        chunks.append((start, best))
        start = best
    return chunks


def fold_registry(split_rows, fold):
    rows = [r for r in split_rows if int(r["fold"]) == fold]
    if len(rows) != 75:
        raise RuntimeError("FOLD_ROW_COUNT:%d:%d" % (fold, len(rows)))

    meta = {}
    roles = defaultdict(list)
    for r in rows:
        uid = r["record_uid"]
        meta[uid] = r
        roles[r["role"]].append(uid)

    for role, expected in [("train", 45), ("validation", 15), ("test", 15)]:
        if len(roles[role]) != expected:
            raise RuntimeError("ROLE_COUNT:%d:%s:%d" % (fold, role, len(roles[role])))

    # Train-fold-only candidate coverage must hold for validation/test.
    for task, col in TASKS:
        train = set(meta[u][col] for u in roles["train"])
        for role in ["validation", "test"]:
            missing = set(meta[u][col] for u in roles[role]) - train
            if missing:
                raise RuntimeError("UNSEEN_COMPONENT:%d:%s:%s:%r" % (fold, role, task, sorted(missing)))

    return meta, roles


def encode_prompt(tokenizer, evidence_ids, task):
    suffix = tokenizer.encode(PROMPT_SUFFIX[task], add_special_tokens=False)
    bos = [int(tokenizer.bos_token_id)] if tokenizer.bos_token_id is not None else []
    ids = bos + list(evidence_ids) + suffix
    if len(ids) >= MODEL_CONTEXT:
        raise RuntimeError("PROMPT_OVERFLOW:%s:%d" % (task, len(ids)))
    return ids


def encode_answer(tokenizer, answer):
    ids = tokenizer.encode(str(answer), add_special_tokens=False)
    if not ids:
        raise RuntimeError("EMPTY_ANSWER")
    return ids


def make_model(model_root, device):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_root),
            local_files_only=True,
            attn_implementation="sdpa",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_root),
            local_files_only=True,
        )
    model.to(device=device, dtype=torch.float32)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model


def train_job(
    fold, cap, variant, model_root, tokenizer, counter, representations,
    packed_chunks, fold_meta, roles, device, job_dir
):
    seed_all(SEED)
    model = make_model(model_root, device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    trace = []
    train_uids = sorted(roles["train"])
    start_time = time.time()

    for epoch in range(EPOCHS):
        order = list(train_uids)
        random.Random(SEED + epoch).shuffle(order)

        for case_idx, uid in enumerate(order, 1):
            optimizer.zero_grad(set_to_none=True)

            chunks = packed_chunks[variant][uid][:cap]
            if not chunks:
                raise RuntimeError("NO_CHUNKS:%s:%s" % (variant, uid))

            n_chunks = len(chunks)
            weight = 1.0 / (2.0 * n_chunks)
            weighted_case_loss = 0.0
            raw_losses = []

            for task, target_col in TASKS:
                answer_ids = encode_answer(tokenizer, fold_meta[uid][target_col])

                for start, end in chunks:
                    evidence_ids = counter.subset_tokens(
                        variant, uid, start, end, representations[variant][uid]
                    )
                    prompt = encode_prompt(tokenizer, evidence_ids, task)
                    ids = prompt + answer_ids
                    if len(ids) > MODEL_CONTEXT:
                        raise RuntimeError("TRAIN_OVERFLOW:%d:%d:%s:%s" % (fold, cap, variant, uid))

                    labels = [-100] * len(prompt) + answer_ids
                    x = torch.tensor([ids], dtype=torch.long, device=device)
                    y = torch.tensor([labels], dtype=torch.long, device=device)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(input_ids=x, labels=y, use_cache=False)
                        raw = out.loss
                        loss = raw * weight

                    if not torch.isfinite(loss):
                        raise RuntimeError("NONFINITE_LOSS:%d:%d:%s:%s" % (fold, cap, variant, uid))

                    loss.backward()
                    weighted_case_loss += float(loss.detach().float().cpu())
                    raw_losses.append(float(raw.detach().float().cpu()))

                    del x, y, out, raw, loss

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            if not math.isfinite(float(grad_norm)):
                raise RuntimeError("NONFINITE_GRAD:%d:%d:%s:%s" % (fold, cap, variant, uid))
            optimizer.step()

            trace.append({
                "fold": fold,
                "cap": cap,
                "variant": variant,
                "case_step": case_idx,
                "record_uid": uid,
                "chunks_used": n_chunks,
                "case_weighted_loss": weighted_case_loss,
                "raw_task_chunk_loss_mean": sum(raw_losses) / len(raw_losses),
                "grad_norm": float(grad_norm.detach().float().cpu())
                if torch.is_tensor(grad_norm) else float(grad_norm),
                "elapsed_sec": time.time() - start_time,
            })

            if case_idx == 1 or case_idx % 15 == 0 or case_idx == 45:
                print(
                    "TRAIN cap=%d fold=%d variant=%s case=%d/45 chunks=%d loss=%.6f"
                    % (cap, fold, variant, case_idx, n_chunks, weighted_case_loss),
                    flush=True,
                )

    write_csv(job_dir / "train_trace.csv", trace)

    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    return model, optimizer


@torch.inference_mode()
def score_candidates(model, prompt_ids, candidates, tokenizer, device):
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(input_ids=x, use_cache=True)

    base_past = out.past_key_values
    first_lp = F.log_softmax(out.logits[0, -1].float(), dim=-1)
    scores = {}

    for candidate in candidates:
        cand_ids = encode_answer(tokenizer, candidate)
        if len(prompt_ids) + len(cand_ids) > MODEL_CONTEXT:
            raise RuntimeError("EVAL_OVERFLOW:%s" % candidate)

        total = float(first_lp[int(cand_ids[0])].cpu())
        count = 1
        past = base_past
        prev = int(cand_ids[0])

        for nxt in cand_ids[1:]:
            t = torch.tensor([[prev]], dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                o = model(input_ids=t, past_key_values=past, use_cache=True)
            past = o.past_key_values
            lp = F.log_softmax(o.logits[0, -1].float(), dim=-1)
            total += float(lp[int(nxt)].cpu())
            count += 1
            prev = int(nxt)
            del t, o

        scores[candidate] = total / count

    del x, out, base_past
    return scores


def evaluate(
    fold, cap, role, variant, model, tokenizer, counter, representations,
    packed_chunks, fold_meta, roles, device
):
    train_rows = [fold_meta[u] for u in roles["train"]]
    candidates_by_task = {
        task: sorted(set(r[col] for r in train_rows))
        for task, col in TASKS
    }

    preds = []
    for idx, uid in enumerate(sorted(roles[role]), 1):
        chunks = packed_chunks[variant][uid][:cap]
        pred_task = {}

        for task, target_col in TASKS:
            per_candidate = {c: [] for c in candidates_by_task[task]}

            for start, end in chunks:
                evidence_ids = counter.subset_tokens(
                    variant, uid, start, end, representations[variant][uid]
                )
                prompt = encode_prompt(tokenizer, evidence_ids, task)
                scores = score_candidates(
                    model, prompt, candidates_by_task[task], tokenizer, device
                )
                for c, s in scores.items():
                    per_candidate[c].append(s)

            agg = {c: sum(v) / len(v) for c, v in per_candidate.items()}
            ordered = sorted(agg)
            pred_task[task] = max(ordered, key=lambda c: agg[c])

        gold_s = fold_meta[uid]["root_cause_service"]
        gold_f = fold_meta[uid]["fault"]
        pred_s = pred_task["ROOT_CAUSE_SERVICE"]
        pred_f = pred_task["FAULT"]

        preds.append({
            "fold": fold,
            "cap": cap,
            "variant": variant,
            "split": role,
            "record_uid": uid,
            "chunks_used": len(chunks),
            "metric_count_total": metric_count(representations[variant][uid], variant),
            "covered_metrics": chunks[-1][1],
            "metric_coverage": chunks[-1][1] / metric_count(representations[variant][uid], variant),
            "gold_root_cause_service": gold_s,
            "pred_root_cause_service": pred_s,
            "service_correct": int(pred_s == gold_s),
            "gold_fault": gold_f,
            "pred_fault": pred_f,
            "fault_correct": int(pred_f == gold_f),
            "pair_exact": int(pred_s == gold_s and pred_f == gold_f),
        })

        if role == "validation" and (idx == 1 or idx % 5 == 0 or idx == 15):
            print("VALIDATION cap=%d fold=%d variant=%s case=%d/15" % (cap, fold, variant, idx), flush=True)
        if role == "test" and idx == 15:
            print("TEST_EVAL cap=%d fold=%d variant=%s cases=15 metrics=SEALED" % (cap, fold, variant), flush=True)

    n = len(preds)
    metrics = {
        "cap": cap,
        "fold": fold,
        "variant": variant,
        "split": role,
        "cases": n,
        "mean_metric_coverage": sum(r["metric_coverage"] for r in preds) / n,
        "service_accuracy": sum(r["service_correct"] for r in preds) / n,
        "fault_accuracy": sum(r["fault_correct"] for r in preds) / n,
        "pair_exact_accuracy": sum(r["pair_exact"] for r in preds) / n,
    }
    return preds, metrics


def run_job(
    fold, cap, variant, model_root, tokenizer, counter, representations,
    packed_chunks, split_rows, device, job_dir
):
    fold_meta, roles = fold_registry(split_rows, fold)
    model, optimizer = train_job(
        fold, cap, variant, model_root, tokenizer, counter, representations,
        packed_chunks, fold_meta, roles, device, job_dir
    )

    val_pred, val_metrics = evaluate(
        fold, cap, "validation", variant, model, tokenizer, counter,
        representations, packed_chunks, fold_meta, roles, device
    )
    test_pred, test_metrics = evaluate(
        fold, cap, "test", variant, model, tokenizer, counter,
        representations, packed_chunks, fold_meta, roles, device
    )

    write_csv(job_dir / "validation_predictions.csv", val_pred)
    write_csv(job_dir / "test_predictions.csv", test_pred)

    done = {
        "version": VERSION,
        "status": "PASS",
        "fold": fold,
        "cap": cap,
        "variant": variant,
        "validation": val_metrics,
        "test": test_metrics,
        "test_metrics_disclosed_during_job": False,
    }
    with open(job_dir / "DONE.json", "w", encoding="utf-8") as f:
        json.dump(done, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    del optimizer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return done


def aggregate(job_summaries):
    per_fold = []
    for j in job_summaries:
        per_fold.append({
            "cap": j["cap"],
            "fold": j["fold"],
            "variant": j["variant"],
            "validation_metric_coverage": j["validation"]["mean_metric_coverage"],
            "validation_service_accuracy": j["validation"]["service_accuracy"],
            "validation_fault_accuracy": j["validation"]["fault_accuracy"],
            "validation_pair_exact_accuracy": j["validation"]["pair_exact_accuracy"],
            "test_metric_coverage": j["test"]["mean_metric_coverage"],
            "test_service_accuracy": j["test"]["service_accuracy"],
            "test_fault_accuracy": j["test"]["fault_accuracy"],
            "test_pair_exact_accuracy": j["test"]["pair_exact_accuracy"],
        })

    agg = []
    for cap in CAPS:
        for variant in VARIANTS:
            rows = [r for r in per_fold if r["cap"] == cap and r["variant"] == variant]
            if len(rows) != 5:
                raise RuntimeError("AGG_COUNT:%d:%s:%d" % (cap, variant, len(rows)))

            out = {"cap": cap, "variant": variant, "folds": 5}
            for col in [
                "test_metric_coverage",
                "test_service_accuracy",
                "test_fault_accuracy",
                "test_pair_exact_accuracy",
            ]:
                vals = [float(r[col]) for r in rows]
                out[col + "_mean"] = statistics.mean(vals)
                out[col + "_std"] = statistics.stdev(vals)
                out[col + "_min"] = min(vals)
                out[col + "_max"] = max(vals)
            agg.append(out)

    return per_fold, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--representation-root", required=True)
    ap.add_argument("--split-registry", required=True)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    model_root = Path(args.model_root).resolve()
    repr_root = Path(args.representation_root).resolve()
    split_path = Path(args.split_registry).resolve()
    out = Path(args.output_root).resolve()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16_REQUIRED")

    seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_root),
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    tokenizer.model_max_length = 10**9

    representations = {}
    for variant in VARIANTS:
        p = repr_root / (variant + "_LOCAL_V1.jsonl")
        if not p.is_file():
            raise FileNotFoundError(str(p))
        representations[variant] = load_repr(p)

    uids = set(representations["B01_EXT_EXACT"])
    if len(uids) != 75:
        raise RuntimeError("CASE_COUNT_NOT_75")
    for variant in VARIANTS[1:]:
        if set(representations[variant]) != uids:
            raise RuntimeError("UID_SET_MISMATCH:%s" % variant)

    split_rows = read_csv(split_path)
    for fold in FOLDS:
        fold_registry(split_rows, fold)

    counter = Counter(tokenizer)

    # Variant-specific greedy packing, same frozen source order.
    packed_chunks = {v: {} for v in VARIANTS}
    coverage_rows = []

    for variant in VARIANTS:
        for uid in sorted(uids):
            chunks = greedy_variant_chunks(
                counter, variant, uid, representations[variant][uid]
            )
            packed_chunks[variant][uid] = chunks
            total = metric_count(representations[variant][uid], variant)

            for cap in CAPS:
                used = chunks[:cap]
                covered = used[-1][1]
                coverage_rows.append({
                    "variant": variant,
                    "record_uid": uid,
                    "cap": cap,
                    "chunks_available": len(chunks),
                    "chunks_used": len(used),
                    "metric_count_total": total,
                    "covered_metrics": covered,
                    "metric_coverage": covered / total,
                    "full_evidence": int(covered == total),
                })

    out.mkdir(parents=True, exist_ok=True)
    audit = out / "audit_safe"
    local = out / "local_only"
    jobs_root = local / "jobs"
    audit.mkdir(exist_ok=True)
    local.mkdir(exist_ok=True)
    jobs_root.mkdir(exist_ok=True)

    write_csv(local / "track_b_case_coverage.csv", coverage_rows)

    print("============================================================", flush=True)
    print("TRACK-B FROZEN 5-FOLD B01-B04 EXPERIMENT START", flush=True)
    print("VERSION=" + VERSION, flush=True)
    print("GPU=" + torch.cuda.get_device_name(0), flush=True)
    print("CAPS=1,2,4,8", flush=True)
    print("FOLDS=0,1,2,3,4", flush=True)
    print("VARIANTS=" + ",".join(VARIANTS), flush=True)
    print("TOTAL_JOBS=80", flush=True)
    print("TEST_DISCLOSURE=AFTER_ALL_80_JOBS", flush=True)
    print("RESUME=ENABLED", flush=True)
    print("============================================================", flush=True)

    summaries = []
    total_jobs = len(CAPS) * len(FOLDS) * len(VARIANTS)
    job_no = 0
    start_all = time.time()

    for cap in CAPS:
        for fold in FOLDS:
            for variant in VARIANTS:
                job_no += 1
                job_dir = jobs_root / ("cap%d__fold%d__%s" % (cap, fold, variant))
                done_path = job_dir / "DONE.json"

                if done_path.is_file():
                    with open(done_path, "r", encoding="utf-8") as f:
                        old = json.load(f)
                    if old.get("version") != VERSION or old.get("status") != "PASS":
                        raise RuntimeError("INVALID_DONE:%s" % done_path)
                    summaries.append(old)
                    print(
                        "JOB_RESUME_SKIP %d/%d cap=%d fold=%d variant=%s"
                        % (job_no, total_jobs, cap, fold, variant),
                        flush=True,
                    )
                    continue

                if job_dir.exists():
                    shutil.rmtree(job_dir)
                job_dir.mkdir(parents=True)

                print(
                    "============================================================\n"
                    "JOB_START %d/%d cap=%d fold=%d variant=%s\n"
                    "============================================================"
                    % (job_no, total_jobs, cap, fold, variant),
                    flush=True,
                )

                done = run_job(
                    fold, cap, variant, model_root, tokenizer, counter,
                    representations, packed_chunks, split_rows, device, job_dir
                )
                summaries.append(done)

                print(
                    "JOB_COMPLETE %d/%d cap=%d fold=%d variant=%s test_metrics=SEALED"
                    % (job_no, total_jobs, cap, fold, variant),
                    flush=True,
                )

    if len(summaries) != total_jobs:
        raise RuntimeError("SUMMARY_COUNT:%d" % len(summaries))

    per_fold, agg = aggregate(summaries)
    write_csv(audit / "QA_SSEC_B01_B04_TRACKB_5FOLD_PER_FOLD_METRICS_V1.csv", per_fold)
    write_csv(audit / "QA_SSEC_B01_B04_TRACKB_5FOLD_AGGREGATE_METRICS_V1.csv", agg)

    final = {
        "version": VERSION,
        "final_status": "PASS",
        "track": "TRACK_B_SAME_TOKEN_BUDGET",
        "caps": CAPS,
        "folds": FOLDS,
        "variants": VARIANTS,
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "model_context": MODEL_CONTEXT,
        "evidence_budget": EVIDENCE_BUDGET,
        "aggregate_metrics": agg,
        "runtime_sec_this_invocation": time.time() - start_all,
        "test_metrics_disclosed_only_after_all_jobs": True,
        "next": "ANALYZE_TRACK_A_VS_TRACK_B_THEN_BUILD_B09_B11_TRAIN_ONLY_REPRESENTATIONS",
    }

    with open(audit / "QA_SSEC_B01_B04_TRACKB_5FOLD_SUMMARY_V1.json", "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    lines = [
        "VERSION=" + VERSION,
        "FINAL_STATUS=PASS",
        "TRACK=TRACK_B_SAME_TOKEN_BUDGET",
        "CAPS=1,2,4,8",
        "FOLDS=5",
        "TEST_METRICS_DISCLOSED_AFTER_ALL_JOBS=1",
    ]
    for r in agg:
        lines.append(
            "cap=%d %s coverage=%.6f service=%.6f fault=%.6f pair=%.6f"
            % (
                r["cap"],
                r["variant"],
                r["test_metric_coverage_mean"],
                r["test_service_accuracy_mean"],
                r["test_fault_accuracy_mean"],
                r["test_pair_exact_accuracy_mean"],
            )
        )
    lines.append("NEXT=ANALYZE_TRACK_A_VS_TRACK_B_THEN_BUILD_B09_B11_TRAIN_ONLY_REPRESENTATIONS")

    (audit / "QA_SSEC_B01_B04_TRACKB_5FOLD_SUMMARY_V1.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("============================================================", flush=True)
    print("ALL 80 JOBS COMPLETE -- TEST METRICS NOW DISCLOSED", flush=True)
    print("\n".join(lines), flush=True)
    print("OUT_ROOT=" + str(out), flush=True)
    print("============================================================", flush=True)


if __name__ == "__main__":
    main()
