"""Exact, proof-producing comparisons of frozen witness outcomes.
Python 3.8+ standard library only. No models, original queries, packing or network.
The unit is the uniform finite seed panel, NOT an estimated future distribution.
Partial stochastic dominance / monotone coupling are established mathematics.
"""
import argparse
import ast
import csv
import hashlib
import io
import itertools
import json
import random
import zipfile
from collections import defaultdict, deque
from fractions import Fraction
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def exact(value):
    value = Fraction(value)
    return {"fraction": str(value), "value": float(value)}


def rows_csv(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def subset(a, b):
    return (a & b) == a


def pop(a):
    return bin(a).count("1")


def accepts(mask, generators):
    return any(subset(g, mask) for g in generators)


def check_support(a, b, dimension):
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("POSITIVE_INTEGER_DIMENSION_REQUIRED")
    if not a or len(a) != len(b):
        raise ValueError("EQUAL_NONEMPTY_UNIFORM_PANELS_REQUIRED")
    if any(type(x) is not int or x < 0 or x >= (1 << dimension) for x in a + b):
        raise ValueError("INVALID_WITNESS_MASK")


def certify(a, b, dimension):
    """Maximize mean f(A)-mean f(B) over all increasing f in [0,1].

    A directed output of zero certifies that B weakly dominates A on this
    empirical panel. Both directions are needed for a complete comparison.
    Matching is a proof device; it does not change the recorded seed pairing.
    """
    a, b = list(a), list(b)
    check_support(a, b, dimension)
    n = len(a)
    adjacency = [[j for j, bj in enumerate(b) if subset(ai, bj)] for ai in a]
    right_owner = [-1] * n

    def augment(i, seen):
        for j in adjacency[i]:
            if j in seen:
                continue
            seen.add(j)
            if right_owner[j] < 0 or augment(right_owner[j], seen):
                right_owner[j] = i
                return True
        return False

    flow = sum(int(augment(i, set())) for i in range(n))
    matching = sorted((i, j) for j, i in enumerate(right_owner) if i >= 0)
    matched_left = {i for i, _ in matching}
    reach_left = {i for i in range(n) if i not in matched_left}
    reach_right = set()
    queue = deque(sorted(reach_left))
    while queue:
        i = queue.popleft()
        for j in adjacency[i]:
            if j in reach_right:
                continue
            reach_right.add(j)
            owner = right_owner[j]
            if owner >= 0 and owner not in reach_left:
                reach_left.add(owner)
                queue.append(owner)

    candidates = sorted({a[i] for i in reach_left}, key=lambda x: (pop(x), x))
    generators = []
    for mask in candidates:
        if not any(subset(g, mask) for g in generators):
            generators.append(mask)
    a_pass = [i for i, mask in enumerate(a) if accepts(mask, generators)]
    b_pass = [i for i, mask in enumerate(b) if accepts(mask, generators)]
    certificate = {
        "dimension": dimension,
        "denominator": n,
        "a_masks": a,
        "b_masks": b,
        "matching": [list(pair) for pair in matching],
        "matched_pairs": flow,
        "deficit_numerator": n - flow,
        "max_empirical_advantage_a_over_b": exact(Fraction(n - flow, n)),
        "upper_set_minimal_generators": generators,
        "a_accept_indices": a_pass,
        "b_accept_indices": b_pass,
        "reachable_left": sorted(reach_left),
        "reachable_right": sorted(reach_right),
        "same_label_coordinate_violations": sum(not subset(x, y) for x, y in zip(a, b)),
        "graph_outcome_vertices": 2 * n,
        "compatible_edges": sum(map(len, adjacency)),
        "distinct_observed_states": len(set(a + b)),
    }
    verify_certificate(certificate)
    return certificate


def verify_certificate(c):
    """Verify upper AND lower bound using the proof object, without a solver.

    A feasible matching of F pairs gives an upper bound (n-F)/n for EVERY
    increasing [0,1] function. A concrete increasing DNF attaining that gap
    proves equality. This also verifies maximality without trusting the solver.
    """
    a, b = c["a_masks"], c["b_masks"]
    check_support(a, b, c["dimension"])
    n = len(a)
    pairs = c["matching"]
    if c["denominator"] != n or c["matched_pairs"] != len(pairs):
        raise ValueError("INCONSISTENT_CERTIFICATE_SIZE")
    if len({i for i, j in pairs}) != len(pairs) or len({j for i, j in pairs}) != len(pairs):
        raise ValueError("REUSED_MATCH_INDEX")
    for i, j in pairs:
        if not (0 <= i < n and 0 <= j < n and subset(a[i], b[j])):
            raise ValueError("INVALID_DOMINATED_PAIR")
    generators = c["upper_set_minimal_generators"]
    if any(type(g) is not int or not 0 <= g < (1 << c["dimension"]) for g in generators):
        raise ValueError("INVALID_DNF_GENERATOR")
    a_pass = [i for i, x in enumerate(a) if accepts(x, generators)]
    b_pass = [i for i, x in enumerate(b) if accepts(x, generators)]
    deficit = n - len(pairs)
    if len(a_pass) - len(b_pass) != deficit:
        raise ValueError("LOWER_BOUND_DOES_NOT_ATTAIN_MATCHING_BOUND")
    if a_pass != c["a_accept_indices"] or b_pass != c["b_accept_indices"]:
        raise ValueError("COUNTEREXAMPLE_EVALUATION_MISMATCH")
    if c["deficit_numerator"] != deficit:
        raise ValueError("DEFICIT_MISMATCH")
    if Fraction(c["max_empirical_advantage_a_over_b"]["fraction"]) != Fraction(deficit, n):
        raise ValueError("FRACTION_MISMATCH")
    return True


def load_evidence(path):
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError("INPUT_CRC_FAILURE")
        provenance = json.loads(z.read("PROVENANCE.json"))
        for name, digest in provenance["member_sha256"].items():
            if sha(z.read(name)) != digest:
                raise ValueError("INPUT_MEMBER_HASH_MISMATCH:" + name)
        frozen_raw = z.read("FIXED_COUNT_SELECTIONS.csv")
        rows = rows_csv(frozen_raw)
        original = {}
        for name, panel, force_arm in [
            ("DEV_SHARED.csv", "DEVELOPMENT", "SHARED"),
            ("DEV_CASE_LOCAL.csv", "DEVELOPMENT", "CASE_LOCAL"),
            ("TRANSFER_SELECTIONS.csv", "TRANSFER", None),
        ]:
            for r in rows_csv(z.read(name)):
                key = (panel, r["case_key"], r["variant"], force_arm or r["arm"], r["seed_id"])
                if key in original:
                    raise ValueError("DUPLICATE_ORIGINAL")
                original[key] = r
    if len(rows) != 9600 or len(original) != 9600:
        raise ValueError("UNEXPECTED_SELECTION_ROWS")
    grouped = defaultdict(list)
    matrices = defaultdict(dict)
    seen = set()
    all_cases = defaultdict(set)
    for r in rows:
        panel, case, variant, arm, label = [r[k] for k in ("panel", "case_key", "variant", "arm", "seed_id")]
        key = (panel, case, variant, arm, label)
        if key in seen:
            raise ValueError("DUPLICATE_FIXED_ROW")
        seen.add(key)
        o = original[key]
        for left, right in [("source_metrics", "source_metrics"), ("gold_positive", "gold_positive"),
                            ("original_selected_metrics", "selected_metrics"), ("original_witness", "witness")]:
            if int(r[left]) != int(o[right]):
                raise ValueError("ORIGINAL_PARITY_FAILURE:" + str((key, left)))
        if not 0 <= int(r["witness"]) <= int(r["original_witness"]) <= 1:
            raise ValueError("INVALID_BINARY_WITNESS")
        all_cases[panel].add(case)
        grouped[(panel, case, variant)].append(r)
        if int(r["source_metrics"]) <= 100 or int(r["gold_positive"]) != 1:
            continue
        for analysis, column in [("PRIMARY", "original_witness"), ("FIXED_COUNT", "witness")]:
            matrices[(analysis, panel, variant, arm)][case, label] = int(r[column])
    if any(len(v) != 75 for v in all_cases.values()):
        raise ValueError("CASE_COUNT_MISMATCH")
    for group in grouped.values():
        if len(group) != 32 or len({int(r["fixed_count"]) for r in group}) != 1:
            raise ValueError("FIXED_COUNT_GROUP_MISMATCH")
        if int(group[0]["fixed_count"]) != min(int(r["original_selected_metrics"]) for r in group):
            raise ValueError("FIXED_COUNT_RULE_MISMATCH")
    samples = {}
    for key, matrix in matrices.items():
        cases = sorted({c for c, s in matrix})
        labels = sorted({s for c, s in matrix})
        if len(labels) != 16:
            raise ValueError("LABEL_COUNT_MISMATCH")
        masks = [sum(matrix[c, s] << i for i, c in enumerate(cases)) for s in labels]
        samples[key] = {"cases": cases, "labels": labels, "masks": masks}
    return samples, {
        "input_zip_sha256": sha(path.read_bytes()),
        "fixed_count_selections_sha256": sha(frozen_raw),
        "source_rows_crosschecked": 9600,
        "fixed_count_groups_checked": len(grouped),
        "input_provenance": provenance,
    }


def compare(left, right, metadata):
    if left["cases"] != right["cases"] or left["labels"] != right["labels"]:
        raise ValueError("DOMAINS_OR_LABELS_DIFFER")
    m = len(left["cases"])
    ab = certify(left["masks"], right["masks"], m)
    ba = certify(right["masks"], left["masks"], m)
    if ab["deficit_numerator"] == ba["deficit_numerator"] == 0:
        status = "EQUAL_EMPIRICAL_LAW_ON_MONOTONE_REQUIREMENTS"
    elif ab["deficit_numerator"] == 0:
        status = "B_DOMINATES_A_ON_FINITE_PANEL"
    elif ba["deficit_numerator"] == 0:
        status = "A_DOMINATES_B_ON_FINITE_PANEL"
    else:
        status = "INCOMPARABLE_ON_FINITE_PANEL"
    for c in (ab, ba):
        c["dnf_minimal_case_sets"] = [
            [name for i, name in enumerate(left["cases"]) if mask & (1 << i)]
            for mask in c["upper_set_minimal_generators"]
        ]
    gained = sum(pop(y & ~x) for x, y in zip(left["masks"], right["masks"]))
    lost = sum(pop(x & ~y) for x, y in zip(left["masks"], right["masks"]))
    return dict(metadata, cases=left["cases"], labels=left["labels"], status=status,
                a_to_b=ab, b_to_a=ba, paired_witness_gain=gained,
                paired_witness_loss=lost, paired_witness_comparisons=16*m)


def all_monotone_truth_tables(m):
    states = 1 << m
    for table in range(1 << states):
        if all(not ((table >> x) & 1) or ((table >> (x | (1 << j))) & 1)
               for x in range(states) for j in range(m) if not (x & (1 << j))):
            yield table


def self_test():
    # Exhaustive finite checks; test data are not study observations.
    checks = 0
    histograms = {}
    for m in (1, 2):
        mfs = list(all_monotone_truth_tables(m))
        histograms[str(m)] = len(mfs)
        panels = list(itertools.combinations_with_replacement(range(1 << m), 3))
        for a in panels:
            for b in panels:
                c = certify(a, b, m)
                truth_max = max(sum((t >> x) & 1 for x in a) - sum((t >> y) & 1 for y in b) for t in mfs)
                assert truth_max == c["deficit_numerator"]
                checks += 1
    mfs = list(all_monotone_truth_tables(3))
    histograms["3"] = len(mfs)
    rng = random.Random(20261101)
    for _ in range(512):
        a = [rng.randrange(8) for i in range(4)]
        b = [rng.randrange(8) for i in range(4)]
        c = certify(a, b, 3)
        truth_max = max(sum((t >> x) & 1 for x in a) - sum((t >> y) & 1 for y in b) for t in mfs)
        assert c["deficit_numerator"] == truth_max
        checks += 1
    # Show that checking all pure conjunctions/disjunctions can miss a mixed one.
    a = [6, 5, 3, 0]
    b = [7, 7, 0, 0]
    # General tests below cover any relation; no claim relies on these as empirical data.
    c = certify(a, b, 3)
    verify_certificate(c)
    corrupted = json.loads(json.dumps(c))
    corrupted["deficit_numerator"] += 1
    try:
        verify_certificate(corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError("CORRUPTED_CERTIFICATE_ACCEPTED")
    ast.parse(Path(__file__).read_text(encoding="utf-8"), feature_version=(3, 8))
    return {"exact_against_enumerated_monotone_functions": checks,
            "boolean_monotone_function_counts_m1_m2_m3": histograms,
            "corrupted_certificate_rejected": True, "python38_syntax": "PASS",
            "these_are_synthetic_code_checks_not_new_empirical_cases": True}


def pure_and_or_max(a, b, m):
    best_a = best_b = 0
    for req in range(1, 1 << m):
        for operator in ("AND", "OR"):
            hit = (lambda y: subset(req, y)) if operator == "AND" else (lambda y: bool(req & y))
            d = sum(hit(x) for x in a) - sum(hit(y) for y in b)
            best_a = max(best_a, d)
            best_b = max(best_b, -d)
    return {"a_over_b_numerator": best_a, "b_over_a_numerator": best_b}


def main(input_path, output):
    samples, provenance = load_evidence(input_path)
    comparisons = []
    for analysis in ("PRIMARY", "FIXED_COUNT"):
        for panel in ("DEVELOPMENT", "TRANSFER"):
            for variant in ("B01", "B02"):
                a, b = samples[analysis, panel, variant, "SHARED"], samples[analysis, panel, variant, "CASE_LOCAL"]
                c = compare(a, b, {"contrast": "ORDERING", "analysis": analysis, "panel": panel,
                                   "variant": variant, "a": "SHARED", "b": "CASE_LOCAL"})
                c["pure_and_or_maximum"] = pure_and_or_max(a["masks"], b["masks"], len(a["cases"]))
                comparisons.append(c)
            for arm in ("SHARED", "CASE_LOCAL"):
                a, b = samples[analysis, panel, "B01", arm], samples[analysis, panel, "B02", arm]
                comparisons.append(compare(a, b, {"contrast": "SERIALIZATION", "analysis": analysis,
                    "panel": panel, "arm": arm, "a": "B01", "b": "B02"}))
    tests = self_test()
    for c in comparisons:
        verify_certificate(c["a_to_b"])
        verify_certificate(c["b_to_a"])
    summary = []
    for c in comparisons:
        ab, ba = c["a_to_b"], c["b_to_a"]
        summary.append({"contrast": c["contrast"], "analysis": c["analysis"], "panel": c["panel"],
            "condition": c.get("variant", c.get("arm")), "a": c["a"], "b": c["b"],
            "positive_cases": len(c["cases"]), "seed_labels": 16,
            "max_advantage_a_over_b_numerator": ab["deficit_numerator"],
            "max_advantage_b_over_a_numerator": ba["deficit_numerator"],
            "max_advantage_a_over_b": ab["max_empirical_advantage_a_over_b"]["value"],
            "max_advantage_b_over_a": ba["max_empirical_advantage_a_over_b"]["value"],
            "a_to_b_compatible_matches": ab["matched_pairs"],
            "b_to_a_compatible_matches": ba["matched_pairs"],
            "a_to_b_same_label_coordinate_violations": ab["same_label_coordinate_violations"],
            "witness_gain_a_to_b": c["paired_witness_gain"],
            "witness_loss_a_to_b": c["paired_witness_loss"],
            "paired_witness_comparisons": c["paired_witness_comparisons"],
            "status": c["status"]})
    output.mkdir(parents=True, exist_ok=True)
    result = {"status": "COMPLETE_EXACT_FINITE_PANEL_CERTIFICATION", "provenance": provenance,
        "analysis_code_sha256": sha(Path(__file__).read_bytes()),
        "scope": {"recorded_seed_panel_not_population": True,
          "comparison_unit": "joint vector of named-case existential witnesses",
          "function_class": "all coordinatewise nondecreasing functions in [0,1]",
          "oracle_access": "stored offline gold-positive case population only",
          "matching_role": "algebraic proof device, not original paired design or deployment coupling",
          "cost_scope": "evidence witness utility only; no numerical fidelity, nonmonotone objectives or latency claim",
          "raw_query_model_packing_network_or_install_calls": 0,
          "new_empirical_cases_or_seeds": 0,
          "synthetic_codechecks_are_separate": True,
          "new_algorithm_invention_claim": False},
        "self_tests": tests,
        "computations": {"comparison_pairs": len(comparisons), "directed_certificates": 2*len(comparisons),
          "solver_independent_certificate_checks": 2*len(comparisons),
          "graph_outcome_vertices_per_direction": 32,
          "maximum_compatibility_edges_per_direction": 256,
          "source_selection_rows": 9600},
        "comparisons": comparisons}
    (output/"CERTIFICATES.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n",encoding="utf-8")
    with (output/"SUMMARY.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    print("FINITE_PANEL_CERTIFICATION=COMPLETE")
    print("DIRECTED_CERTIFICATES_VERIFIED=32/32")
    print("SYNTHETIC_CODE_CHECKS=" + str(tests["exact_against_enumerated_monotone_functions"]))
    print("ORIGINAL_QUERIES_MODELS_PACKING_INSTALL_NETWORK=0")
    print("OUTPUT=" + str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("INPUT_EVIDENCE.zip"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("recomputed"))
    parser.add_argument("--verify", type=Path, help="Only verify an existing proof bundle; do not solve.")
    args = parser.parse_args()
    if args.verify:
        obj = json.loads(args.verify.read_text(encoding="utf-8"))
        samples, prov = load_evidence(args.input)
        if prov != obj["provenance"]:
            raise ValueError("CERTIFICATE_INPUT_PROVENANCE_MISMATCH")
        for item in obj["comparisons"]:
            verify_certificate(item["a_to_b"]); verify_certificate(item["b_to_a"])
            if item["contrast"] == "ORDERING":
                ka = (item["analysis"], item["panel"], item["variant"], item["a"])
                kb = (item["analysis"], item["panel"], item["variant"], item["b"])
            else:
                ka = (item["analysis"], item["panel"], item["a"], item["arm"])
                kb = (item["analysis"], item["panel"], item["b"], item["arm"])
            if item["a_to_b"]["a_masks"] != samples[ka]["masks"] or item["a_to_b"]["b_masks"] != samples[kb]["masks"]:
                raise ValueError("PROOF_MASKS_DIFFER_FROM_STORED_WITNESSES")
            if item["b_to_a"]["a_masks"] != samples[kb]["masks"] or item["b_to_a"]["b_masks"] != samples[ka]["masks"]:
                raise ValueError("REVERSE_PROOF_MASKS_DIFFER")
            if item["cases"] != samples[ka]["cases"] or item["labels"] != samples[ka]["labels"]:
                raise ValueError("PROOF_COORDINATES_DIFFER")
        print("SOLVER_FREE_PROOF_CHECK=PASS_" + str(2*len(obj["comparisons"])))
    else:
        main(args.input, args.output)
