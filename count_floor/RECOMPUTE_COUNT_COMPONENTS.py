#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""저장된 질의 결과만 읽는 COUNT 오차 분해. 원본 질의/모델/패킹 실행 없음.

Python 3.8+ 표준 라이브러리만 사용합니다.
사용: python RECOMPUTE_COUNT_COMPONENTS.py --source-dir <세 원본 ZIP 폴더> --out <새 출력 폴더>
출력 원본을 덮어쓰지 않으며 네트워크/환경 설치를 하지 않습니다.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path

COUNTS = {
    'COUNT_POSITIVE_MEAN_SHIFT', 'COUNT_MEAN_MEDIAN_SIGN_DISAGREEMENT',
    'COUNT_MISSING_RATE_INCREASE', 'COUNT_STD_INCREASE'
}
KEYS = ['analysis', 'panel', 'variant', 'arm', 'query_family']

def read_rows(path, member):
    with zipfile.ZipFile(str(path)) as archive:
        with archive.open(member) as raw:
            return list(csv.DictReader(io.TextIOWrapper(raw, encoding='utf-8-sig', newline='')))

def as_int(value):
    v = float(value)
    i = int(round(v))
    if not math.isfinite(v) or not math.isclose(v, i, rel_tol=0, abs_tol=1e-8):
        raise ValueError('NONINTEGER_STORED_COUNT')
    return i

def main(source, output):
    if output.exists():
        raise FileExistsError('OUTPUT_EXISTS_NO_OVERWRITE: ' + str(output))
    specifications = [
        ('FIXED_COUNT_RESULT.zip', 'FIXED_COUNT_QUERIES.csv', 'FIXED', None),
        ('QUERY_REPLAY_RESULT.zip', 'QUERY_REPLAY_CASE_RESULTS.csv', 'PRIMARY', 'DEVELOPMENT'),
        ('NEXT_CASE_TRANSFER_RESULT.zip', 'QUERY_RESULTS.csv', 'PRIMARY', 'TRANSFER')
    ]
    rows, source_hashes = [], {}
    for filename, member, analysis, panel in specifications:
        path = source / filename
        source_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in read_rows(path, member):
            row['analysis'] = analysis
            row['panel'] = panel or row['panel']
            row['cohort'] = row.get('cohort', row.get('dataset', ''))
            row['arm'] = row.get('arm', row.get('ordering_arm', ''))
            keep = (row['panel'] == 'DEVELOPMENT' and row['cohort'] == 'LARGE_199_232') or (
                row['panel'] == 'TRANSFER' and row['cohort'] == 'RE1-TT')
            if keep and row['query_family'] in COUNTS:
                row['k'] = as_int(row['fixed_count'] if analysis == 'FIXED' else row['selected_metrics'])
                rows.append(row)
    gold_by_case, neutral_by_case = defaultdict(set), defaultdict(set)
    for row in rows:
        key = (row['panel'], row['case_key'], row['query_family'])
        neutral_by_case[key].add(as_int(row['gold_neutral']))
        error = float(row['abs_error'])
        scaled = float(row['scaled_abs_error'])
        if as_int(row['gold_neutral']):
            gold_by_case[key].add(0)
        elif error > 0 and scaled > 0:
            gold_by_case[key].add(as_int(error / scaled))
    for key in neutral_by_case:
        if len(neutral_by_case[key]) != 1 or len(gold_by_case[key]) != 1:
            raise ValueError('GOLD_NOT_UNIQUELY_RECOVERABLE: ' + str(key))
    grouped = defaultdict(list)
    for row in rows:
        key = (row['panel'], row['case_key'], row['query_family'])
        m = next(iter(gold_by_case[key]))
        error = float(row['abs_error'])
        observed = as_int(m - error)
        k = row['k']
        if not (0 <= observed <= min(m, k)):
            raise ValueError('SUBSET_COUNT_CONTRACT_FAILED')
        floor = max(m - k, 0)
        residual = min(m, k) - observed
        if not math.isclose(error, floor + residual, rel_tol=0, abs_tol=1e-8):
            raise ValueError('DECOMPOSITION_MISMATCH')
        row.update(m=m, observed_count=observed, cardinality_lower_bound=floor,
                   composition_residual=residual, error=error)
        grouped[tuple(row[x] for x in KEYS)].append(row)
    summary = []
    for key, group in sorted(grouped.items()):
        n = len(group)
        total_error = math.fsum(x['error'] for x in group)
        total_floor = sum(x['cardinality_lower_bound'] for x in group)
        d = dict(zip(KEYS, key))
        d.update(rows=n, cases=len(set(x['case_key'] for x in group)),
                 mean_source_count=sum(x['m'] for x in group)/n,
                 mean_k=sum(x['k'] for x in group)/n, mae=total_error/n,
                 mean_lower_bound=total_floor/n,
                 mean_composition_residual=sum(x['composition_residual'] for x in group)/n,
                 bound_share_of_total_error=total_floor/total_error if total_error else None)
        summary.append(d)
    if len(rows) != 25600 or len(summary) != 64:
        raise ValueError('UNEXPECTED_ANALYSIS_GRID')
    output.mkdir(parents=True)
    with (output / 'COUNT_ERROR_COMPONENTS.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    details = {
        'date': '2026-10-31', 'analysis': 'DESCRIPTIVE_STORED_OUTPUT_ALGEBRA',
        'source_sha256': source_hashes, 'count_rows_used': len(rows), 'summary_rows': len(summary),
        'source_gold_case_query_pairs': len(gold_by_case),
        'identity': 'm-c = max(m-k,0) + min(m,k)-c',
        'invariants': ['integer gold consistent across existing nonzero-error rows',
                       '0 <= retained positives <= min(gold,k)',
                       'decomposition equals existing error on all selected rows'],
        'interpretation': 'cardinality-only lower bound, not token-feasible oracle; residual is not proven avoidable under actual packing',
        'new_raw_queries': 0, 'new_experiments': 0, 'new_settings': 0,
        'original_results_modified': False, 'p_values_or_population_inference': None
    }
    (output/'CALCULATION_RECORD.json').write_text(json.dumps(details, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('STORED_COUNT_DECOMPOSITION=COMPLETED;ROWS=%d;SUMMARIES=%d' % (len(rows), len(summary)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.source_dir, args.out)
