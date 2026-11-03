#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""저장된 결과 ZIP만 재집계합니다. 원본 질의, 패킹, 토큰화, 설치, 네트워크를 실행하지 않습니다.
Python 3.8+ 표준 라이브러리만 사용합니다.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import zipfile
from fractions import Fraction
from pathlib import Path

PRIMARY = {'DEVELOPMENT': 'LARGE_199_232', 'TRANSFER': 'RE1-TT'}
FAMILIES = [
    'COUNT_POSITIVE_MEAN_SHIFT', 'FRACTION_POSITIVE_MEAN_SHIFT',
    'COUNT_MEAN_MEDIAN_SIGN_DISAGREEMENT', 'COUNT_MISSING_RATE_INCREASE',
    'MEAN_MISSING_RATE_CHANGE', 'COUNT_STD_INCREASE', 'EXISTS_MISSING_RATE_INCREASE'
]
INTS = {
    'source_metrics','original_selected_metrics','selected_metrics','fixed_count',
    'gold_positive','original_witness','witness','stored_queries_reused',
    'source_valid_metrics','selected_valid_metrics','returned','agreement',
    'oracle_scope_complete','correct_but_incomplete','visible_witness_certified_true',
    'gold_neutral','largest_original_count','original_settings','unchanged_prefix_settings'
}


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def csv_rows(blob):
    rows = list(csv.DictReader(io.StringIO(blob.decode('utf-8-sig'))))
    for r in rows:
        for k in INTS.intersection(r):
            r[k] = int(r[k])
        for k in ('abs_error', 'scaled_abs_error'):
            if k in r:
                r[k] = float(r[k]) if r[k] not in ('', None) else None
    return rows


def pk(r):
    return tuple(r[k] for k in ('panel','case_key','variant','arm','seed_id'))


def qk(r):
    return pk(r) + (r['query_family'],)


def avg(values):
    values = list(values)
    return math.fsum(values) / len(values) if values else None


def frac_mean(values):
    values = list(values)
    return sum(values, Fraction(0)) / len(values)


def near(a,b):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-12)


def groups(rows, keys):
    out = collections.defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def witness_stat(rows, outcome, count_field):
    positive = [r for r in rows if r['gold_positive']]
    by_case = groups(positive,['case_key'])
    by_seed = groups(positive,['seed_id'])
    if len(by_seed) != 16 or any(len(v)!=16 for v in by_case.values()):
        raise ValueError('WITNESS_PANEL_INCOMPLETE')
    hits = {s[0]:sum(r[outcome] for r in rr) for s,rr in sorted(by_seed.items())}
    ps = [Fraction(sum(r[outcome] for r in rr),16) for _,rr in sorted(by_case.items())]
    mean_t = frac_mean(Fraction(v) for v in hits.values())
    var_t = frac_mean((Fraction(v)-mean_t)**2 for v in hits.values())
    marginal = sum((p*(1-p) for p in ps), Fraction(0))
    cov = var_t-marginal
    coverage = frac_mean(Fraction(r[count_field],r['source_metrics']) for r in rows)
    return {
        'positive_cases':len(by_case),'positive_settings':len(positive),
        'witness_hits':sum(hits.values()),'mean_recall':float(mean_t/len(by_case)),
        'all_miss_seeds':sum(v==0 for v in hits.values()),
        'all_miss_seed_ids':[s for s,v in hits.items() if v==0],
        'hit_count_variance':float(var_t),'hit_count_variance_exact':str(var_t),
        'marginal_variance_sum':float(marginal),'covariance_term':float(cov),
        'primary_mean_record_coverage':float(coverage),'hits_by_seed':hits,
        'case_hits':{c[0]:sum(r[outcome] for r in rr) for c,rr in sorted(by_case.items())}
    }


def query_stat(rows):
    errors = [r['abs_error'] for r in rows if r['abs_error'] is not None]
    scaled = [r['scaled_abs_error'] for r in rows if r['scaled_abs_error'] is not None]
    return {
        'settings':len(rows),'agreement_count':sum(r['agreement'] for r in rows),
        'scope_complete_count':sum(r['oracle_scope_complete'] for r in rows),
        'returned_count':sum(r['returned'] for r in rows),
        'no_answer_count':sum(1-r['returned'] for r in rows),
        'error_defined_count':len(errors),'mean_absolute_error':avg(errors),
        'mean_scaled_absolute_error':avg(scaled)
    }


def paired_witness(rows):
    lookup = {pk(r):r for r in rows if r['gold_positive']}
    pairs=[]
    for r in rows:
        if r['arm']!='SHARED' or not r['gold_positive']:continue
        k=(r['panel'],r['case_key'],r['variant'],'CASE_LOCAL',r['seed_id'])
        pairs.append((r,lookup[k]))
    diffs=collections.defaultdict(int)
    transitions=collections.Counter()
    for a,b in pairs:
        transitions[(a['witness'],b['witness'])]+=1
        diffs[a['case_key']]+=b['witness']-a['witness']
    return {'settings':len(pairs),'miss_to_hit':transitions[0,1],
            'hit_to_miss':transitions[1,0],'both_hit':transitions[1,1],
            'both_miss':transitions[0,0],
            'cases_improved':sum(d>0 for d in diffs.values()),
            'cases_worsened':sum(d<0 for d in diffs.values()),
            'cases_equal':sum(d==0 for d in diffs.values()),
            'case_hit_delta':dict(sorted(diffs.items()))}


def query_pairs(rows):
    lookup={pk(r):r for r in rows}
    ds=[];by_case=collections.defaultdict(list);equal_valid=0
    gain=harm=0
    for a in rows:
        if a['arm']!='SHARED':continue
        b=lookup[(a['panel'],a['case_key'],a['variant'],'CASE_LOCAL',a['seed_id'])]
        equal_valid+=int(a['selected_valid_metrics']==b['selected_valid_metrics'])
        gain+=int(a['agreement']==0 and b['agreement']==1)
        harm+=int(a['agreement']==1 and b['agreement']==0)
        if a['abs_error'] is not None and b['abs_error'] is not None:
            d=b['abs_error']-a['abs_error'];ds.append(d);by_case[a['case_key']].append(d)
    cm={c:avg(v) for c,v in sorted(by_case.items())}
    return {'paired_settings':len(rows)//2,'equal_selected_valid_metrics':equal_valid,
            'unequal_selected_valid_metrics':len(rows)//2-equal_valid,
            'agreement_gain':gain,'agreement_harm':harm,
            'numeric_error_worse':sum(d>1e-12 for d in ds),
            'numeric_error_better':sum(d<-1e-12 for d in ds),
            'numeric_error_equal':sum(abs(d)<=1e-12 for d in ds),
            'cases_mean_error_worse':sum(d>1e-12 for d in cm.values()),
            'cases_mean_error_better':sum(d<-1e-12 for d in cm.values()),
            'cases_mean_error_equal':sum(abs(d)<=1e-12 for d in cm.values())}


def analyze(source_path):
    with zipfile.ZipFile(source_path) as source:
        blobs={name:source.read(name) for name in source.namelist() if not name.endswith('/')}
    fixed_blob=blobs['FIXED_COUNT_RESULT.zip']
    with zipfile.ZipFile(io.BytesIO(fixed_blob)) as z:
        if z.testzip() is not None:raise ValueError('INPUT_CRC_FAILURE')
        result=json.loads(z.read('FIXED_COUNT_RESULT.json'))
        for name,h in result['output_hashes'].items():
            if digest(z.read(name))!=h:raise ValueError('OUTPUT_HASH_MISMATCH:'+name)
        selection=csv_rows(z.read('FIXED_COUNT_SELECTIONS.csv'))
        queries=csv_rows(z.read('FIXED_COUNT_QUERIES.csv'))
        caps=csv_rows(z.read('FIXED_COUNTS.csv'))
        saved_qsummary=list(csv.DictReader(io.StringIO(z.read('QUERY_SUMMARY.csv').decode('utf-8-sig'))))
    previous=[]
    for name,member,panel in [('QUERY_REPLAY_RESULT.zip','QUERY_REPLAY_CASE_RESULTS.csv','DEVELOPMENT'),
                              ('NEXT_CASE_TRANSFER_RESULT.zip','QUERY_RESULTS.csv','TRANSFER')]:
        with zipfile.ZipFile(io.BytesIO(blobs[name])) as z:
            rr=csv_rows(z.read(member))
        for r in rr:
            r['panel']=panel
            if panel=='DEVELOPMENT':r['arm']=r.pop('ordering_arm')
            else:r['cohort']=r.pop('dataset')
        previous.extend(rr)
    if len(selection)!=9600 or len(queries)!=67200 or len(caps)!=300:
        raise ValueError('RESULT_SIZE_MISMATCH')
    if len({pk(r) for r in selection})!=9600 or len({qk(r) for r in queries})!=67200:
        raise ValueError('DUPLICATE_KEYS')
    capgroups=groups(selection,['panel','case_key','variant'])
    for k,rr in capgroups.items():
        if len(rr)!=32 or len({r['fixed_count'] for r in rr})!=1:
            raise ValueError('COUNT_NOT_MATCHED')
        if rr[0]['fixed_count']!=min(r['original_selected_metrics'] for r in rr):
            raise ValueError('CAP_NOT_MINIMUM')
        if any(r['fixed_count']>r['original_selected_metrics'] or r['witness']>r['original_witness'] for r in rr):
            raise ValueError('PREFIX_SUBSET_INCONSISTENCY')
    old={qk(r):r for r in previous}
    selection_lookup={pk(r):r for r in selection}
    reused=0
    identical_fields=['returned','agreement','oracle_scope_complete','correct_but_incomplete',
        'visible_witness_certified_true','gold_neutral','abs_error','scaled_abs_error',
        'source_valid_metrics','selected_valid_metrics']
    for r in queries:
        a=old[qk(r)];s=selection_lookup[pk(r)]
        if r['fixed_count']!=s['fixed_count'] or r['stored_queries_reused']!=s['stored_queries_reused']:
            raise ValueError('QUERY_PREFIX_ALIGNMENT_MISMATCH')
        if r['query_family']=='EXISTS_MISSING_RATE_INCREASE':
            if r['visible_witness_certified_true']!=s['witness']:
                raise ValueError('EXISTS_WITNESS_MISMATCH')
        if r['stored_queries_reused']:
            reused+=1
            if not all(near(r[f],a[f]) for f in identical_fields):
                raise ValueError('REUSED_ANSWER_DIFFERENT')
    if reused!=46403 or len(queries)-reused!=20797:raise ValueError('REUSE_COUNT_DIFFERENT')
    # 기존 저장 집계와 새 집계의 대조. 실제 질의 함수를 다시 호출하지 않습니다.
    qgroups=collections.defaultdict(list)
    for r in queries:
        for co in ('ALL',r['cohort']):
            qgroups[(r['panel'],co,r['arm'],r['variant'],r['query_family'])].append(r)
    computed_q={k:query_stat(v) for k,v in qgroups.items()}
    if len(computed_q)!=196:raise ValueError('SUMMARY_COUNT_DIFFERENT')
    for r in saved_qsummary:
        k=tuple(r[f] for f in ('panel','cohort','arm','variant','query_family'))
        for f,v in computed_q[k].items():
            stored=None if r[f]=='' else float(r[f])
            if not near(v,stored):raise ValueError('SUMMARY_VALUE_DIFFERS:'+f)
    out={
        'analysis_date':'2026-10-30','analysis':'STORED_OUTPUT_REAGGREGATION_AND_CLAIM_SYNTHESIS',
        'source_zip_sha256':{n:digest(b) for n,b in sorted(blobs.items())},
        'validation':{'prefix_rows':9600,'query_rows':67200,'cap_groups':300,
                      'all_cap_groups_match_across_arms_and_seeds':True,
                      'stored_answers_reused_identically':reused,'new_queries_in_original_projection':20797,
                      'query_summaries_recomputed':196,'returned_rows':sum(r['returned'] for r in queries),
                      'numeric_error_rows':sum(r['abs_error'] is not None for r in queries),
                      'undefined_exists_error_rows':sum(r['abs_error'] is None for r in queries)},
        'new_experiments_or_query_evaluations_this_readout':0,
        'new_tokenizer_packing_model_network_or_install_calls':0,
        'comparisons':[], 'operators':[], 'cap_ranges':[],
        'limits':['Post-hoc fixed-count diagnostic, not a new independent dataset or confirmatory test.',
                  'Fixed physical count is not fixed selected-valid scope or identical content.',
                  'Projected prefixes are not retokenized or newly certified against the token budget.',
                  '16 deterministic seed labels per panel; no p-values, population CI or risk ratios.',
                  'Development and transfer differ in both cases and seeds; their effects are not separated.',
                  'No tuning, additional projection, or change to primary results is performed.']
    }
    rows_out=[]
    for (p,co,v),rr in sorted(groups(selection,['panel','cohort','variant']).items()):
        out['cap_ranges'].append({'panel':p,'cohort':co,'variant':v,'min':min(r['fixed_count'] for r in rr),
                                  'max':max(r['fixed_count'] for r in rr),'cases':len({r['case_key'] for r in rr})})
    for panel,cohort in PRIMARY.items():
        for variant in ('B01','B02'):
            rr=[r for r in selection if r['panel']==panel and r['cohort']==cohort and r['variant']==variant]
            comp={'panel':panel,'cohort':cohort,'variant':variant,'primary':{},'fixed':{},'paired_fixed':paired_witness(rr)}
            for arm in ('SHARED','CASE_LOCAL'):
                a=[r for r in rr if r['arm']==arm]
                comp['primary'][arm]=witness_stat(a,'original_witness','original_selected_metrics')
                comp['fixed'][arm]=witness_stat(a,'witness','fixed_count')
            s=comp['fixed']['SHARED'];l=comp['fixed']['CASE_LOCAL']
            comp['fixed_recall_delta_pp']=(l['mean_recall']-s['mean_recall'])*100
            comp['fixed_variance_reduction_percent']=100*(1-l['hit_count_variance']/s['hit_count_variance'])
            out['comparisons'].append(comp)
            for design in ('primary','fixed'):
                for arm in ('SHARED','CASE_LOCAL'):
                    stat=comp[design][arm]
                    rows_out.append(dict(row_type='WITNESS',panel=panel,cohort=cohort,variant=variant,design=design,arm=arm,
                        **{k:v for k,v in stat.items() if not isinstance(v,(dict,list))}))
            for family in FAMILIES:
                rrs=[r for r in queries if r['panel']==panel and r['cohort']==cohort and r['variant']==variant and r['query_family']==family]
                pp=[r for r in previous if r['panel']==panel and r['cohort']==cohort and r['variant']==variant and r['query_family']==family]
                op={'panel':panel,'cohort':cohort,'variant':variant,'query_family':family,'fixed':{},'primary':{},'paired_fixed':query_pairs(rrs)}
                for design,data in [('fixed',rrs),('primary',pp)]:
                    for arm in ('SHARED','CASE_LOCAL'):
                        op[design][arm]=query_stat([r for r in data if r['arm']==arm])
                        rows_out.append(dict(row_type='QUERY',panel=panel,cohort=cohort,variant=variant,design=design,arm=arm,query_family=family,**op[design][arm]))
                if family=='COUNT_MISSING_RATE_INCREASE':
                    pos={r['case_key'] for r in rr if r['gold_positive']}
                    op['positive_exact_counts']={arm:sum(r['agreement'] for r in rrs if r['arm']==arm and r['case_key'] in pos) for arm in ('SHARED','CASE_LOCAL')}
                out['operators'].append(op)
                rows_out.append(dict(row_type='PAIRED_QUERY',panel=panel,cohort=cohort,variant=variant,query_family=family,**op['paired_fixed']))
    return out,rows_out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',default=str(Path(__file__).with_name('07_SOURCE_RESULTS.zip')))
    ap.add_argument('--out',default=None)
    args=ap.parse_args()
    summary,rows=analyze(Path(args.source))
    text=json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.out is None:
        print(text);return
    output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    targets=[output/'04_ANALYSIS_SUMMARY.json',output/'03_COMPARISON_TABLES.csv']
    if any(p.exists() for p in targets):raise FileExistsError('OUTPUT_EXISTS_NO_OVERWRITE')
    targets[0].write_text(text,encoding='utf-8')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with targets[1].open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print('STORED_RESULT_REAGGREGATION=PASS')
    print('FIXED_COUNT_ROWS=9600;QUERY_ROWS=67200;SUMMARY_ROWS=196')
    print('NEW_QUERY_TOKENIZER_PACKING_MODEL_OR_NETWORK_CALLS=0')

if __name__=='__main__':
    main()
