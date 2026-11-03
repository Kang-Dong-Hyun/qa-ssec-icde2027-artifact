"""기존 ZIP 출력만 읽는 사후 기전 분석. 원본 질의/토큰화/학습/다운로드 없음.
Python 3.8+ 표준 라이브러리. 기본 사용: python RECOMPUTE.py
기본 출력은 현재 패키지의 recomputed 하위 폴더이며 기존 산출물을 덮어쓰지 않습니다.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
from fractions import Fraction as F
from pathlib import Path
import zipfile

COUNT = 'COUNT_POSITIVE_MEAN_SHIFT'
FRACTION = 'FRACTION_POSITIVE_MEAN_SHIFT'
TOL = F(1, 10**12)
PINS = {'QUERY_REPLAY_RESULT.zip': '3b937f17f2be1d63912e408f2676b72b8dc4637e57910942b685d39f9ff319e8', 'NEXT_CASE_TRANSFER_RESULT.zip': '7919d4bc6b8142d1ff2cc96e823bb3bc6d7348ffe6925a7a789231fdea94a228', 'FIXED_COUNT_RESULT.zip': 'df08f1d7e3dc43b564752fe4f7df643ece13e3dc8deb15a6dcbd149cd3390037'}

def sha(b):
    return hashlib.sha256(b).hexdigest()

def csv_rows(blob):
    return list(csv.DictReader(io.StringIO(blob.decode('utf-8-sig'))))

def number(x):
    return {'exact': str(x), 'value': float(x)}

def int_checked(value):
    v = float(value)
    i = round(v)
    if not math.isfinite(v) or abs(v-i) > 1e-8:
        raise ValueError('NON_INTEGER_COUNT')
    return i

def direction(x, tol=TOL):
    return 1 if x > tol else -1 if x < -tol else 0

def emit_json(path, value):
    blob = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False) + '\n').encode('utf-8')
    if path.exists() and path.read_bytes() != blob:
        raise RuntimeError('OUTPUT_CONFLICT: ' + str(path))
    path.write_bytes(blob)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evidence', type=Path, default=Path(__file__).with_name('INPUT_RESULTS.zip'))
    ap.add_argument('--out', type=Path, default=Path(__file__).with_name('recomputed'))
    args = ap.parse_args()
    inputs = {}
    with zipfile.ZipFile(args.evidence) as outer:
        for name, expected in PINS.items():
            b = outer.read(name)
            if sha(b) != expected:
                raise RuntimeError('SOURCE_HASH_MISMATCH: ' + name)
            inputs[name] = b
    def read(name, member):
        with zipfile.ZipFile(io.BytesIO(inputs[name])) as z:
            return csv_rows(z.read(member))
    records = []
    for panel, regime, name, member in [
        ('DEVELOPMENT','PRIMARY','QUERY_REPLAY_RESULT.zip','QUERY_REPLAY_CASE_RESULTS.csv'),
        ('TRANSFER','PRIMARY','NEXT_CASE_TRANSFER_RESULT.zip','QUERY_RESULTS.csv'),
        ('BOTH','FIXED','FIXED_COUNT_RESULT.zip','FIXED_COUNT_QUERIES.csv')]:
        for r in read(name, member):
            r['panel'] = r.get('panel', panel)
            r['regime'] = regime
            r['arm'] = r.get('arm', r.get('ordering_arm'))
            r['physical_k'] = int(r.get('fixed_count', r.get('selected_metrics', 0)))
            records.append(r)
    assert len(records) == 134400
    # 공개 기록의 큰 사례 집단만 사용: 개발 199-232, 전이 198-238.
    count_rows = [r for r in records if r['query_family'] == COUNT and int(r['source_metrics']) > 100]
    fraction_rows = [r for r in records if r['query_family'] == FRACTION and int(r['source_metrics']) > 100]
    assert len(count_rows) == len(fraction_rows) == 6400
    def key(r):
        return (r['panel'], r['regime'], r['case_key'], r['variant'], r['arm'], r['seed_id'])
    fractions = {key(r): r for r in fraction_rows}
    assert len(fractions) == 6400
    candidates = collections.defaultdict(set)
    for r in count_rows:
        if int(r['gold_neutral']):
            candidates[(r['panel'],r['case_key'])].add(0)
        elif float(r['abs_error']) > 0:
            scaled = float(r['scaled_abs_error'])
            if scaled <= 0:
                raise RuntimeError('INVALID_SCALED_ERROR')
            m = int_checked(float(r['abs_error']) / scaled)
            candidates[(r['panel'],r['case_key'])].add(m)
    if len(candidates) != 50 or any(len(x) != 1 for x in candidates.values()):
        raise RuntimeError('GOLD_COUNT_NOT_UNIQUELY_RECOVERED_FOR_50_CASES')
    gold = {k: next(iter(v)) for k,v in candidates.items()}
    all_cells = {}; max_difference = 0.0
    for r in count_rows:
        f = fractions[key(r)]
        m = gold[(r['panel'],r['case_key'])]
        e = int_checked(r['abs_error']); c = m-e
        N = int(r['source_valid_metrics']); n = int(r['selected_valid_metrics'])
        assert N == int(f['source_valid_metrics']) and n == int(f['selected_valid_metrics'])
        assert 0 <= c <= min(m,n) and 0 < n <= N and 0 < m < N
        assert n <= r['physical_k'] <= int(r['source_metrics'])
        p = F(m,N); rho = F(n,N); r_pos = F(c,m); r_neg = F(n-c,N-m)
        signed = F(c,n)-p
        assert rho == p*r_pos + (1-p)*r_neg
        assert signed == p*(1-p)*(r_pos-r_neg)/rho
        delta = abs(float(abs(signed))-float(f['abs_error']))
        max_difference = max(max_difference, delta)
        if delta > 2e-12:
            raise RuntimeError('FRACTION_RECONSTRUCTION_MISMATCH')
        assert int(r['agreement']) == int(e == 0)
        assert int(f['agreement']) == int(math.isclose(float(F(c,n)),float(p),rel_tol=1e-12,abs_tol=1e-12))
        expanded_count = F(N*c,n)
        assert abs(expanded_count-m) == N*abs(signed)
        all_cells[key(r)] = dict(N=N,n=n,m=m,c=c,k=r['physical_k'],
            count_error=F(e),fraction_error=abs(signed),signed_fraction_error=signed,
            expansion_error=abs(expanded_count-m),physical_N=int(r['source_metrics']))
    assert len(all_cells) == 6400
    pairs = collections.defaultdict(lambda: collections.defaultdict(dict))
    for k,v in all_cells.items():
        panel,regime,case,variant,arm,seed = k
        pairs[(panel,regime,variant)][(case,seed)][arm] = v
    summaries=[]; examples=[]; checked_criteria=0; expansion_pair_checks=0
    for (panel,regime,variant), group in sorted(pairs.items()):
        tally=collections.Counter(); sums=collections.defaultdict(F)
        for (case,seed), aa in sorted(group.items()):
            assert set(aa)=={'SHARED','CASE_LOCAL'}
            s,l=aa['SHARED'],aa['CASE_LOCAL']
            assert s['N']==l['N'] and s['m']==l['m']
            eq_n=(s['n']==l['n']); eq_k=(s['k']==l['k'])
            d_c=direction(l['count_error']-s['count_error'])
            d_f=direction(l['fraction_error']-s['fraction_error'])
            inv=(d_c*d_f==-1)
            d_exp=direction(l['expansion_error']-s['expansion_error'])
            assert d_exp==d_f
            expansion_pair_checks+=1
            tally['paired_settings']+=1; tally['equal_physical_count']+=eq_k
            tally['equal_valid_count']+=eq_n; tally['opposite_error_directions']+=inv
            tally['opposite_error_directions_equal_valid_count']+=inv and eq_n
            tally['local_count_better_fraction_worse']+=(d_c<0 and d_f>0)
            tally['local_count_worse_fraction_better']+=(d_c>0 and d_f<0)
            tally['equal_count_fraction_tie']+=d_c==0 or d_f==0
            tally['expansion_fraction_opposite_directions']+=d_exp*d_f==-1
            if eq_n:
                # 동일 n에서 COUNT와 FRACTION의 역순위 조건을 유리수로 검산합니다.
                criterion = l['c'] != s['c'] and F(l['c']+s['c']) > F(2*s['n']*s['m'],s['N'])
                assert inv == criterion
                checked_criteria+=1
            for arm,v in aa.items():
                sums[arm+'_count_mae']+=v['count_error']
                sums[arm+'_fraction_mae']+=v['fraction_error']
                sums[arm+'_expansion_mae']+=v['expansion_error']
            if panel=='TRANSFER' and regime=='FIXED' and variant=='B02' and case=='01c4e13b704b1ded' and seed=='SEED_21':
                examples.append(dict(panel=panel,regime=regime,variant=variant,case_key=case,seed=seed,
                    SHARED={k:(number(v) if isinstance(v,F) else v) for k,v in s.items()},
                    CASE_LOCAL={k:(number(v) if isinstance(v,F) else v) for k,v in l.items()}))
        assert tally['paired_settings']==400
        summaries.append(dict(panel=panel,regime=regime,variant=variant,**dict(tally),
            means={k:number(v/400) for k,v in sorted(sums.items())}))
    # 양성 사례의 변동을 분해: 고정 수량 기록의 원래 witness 결과 사용.
    selection=read('FIXED_COUNT_RESULT.zip','FIXED_COUNT_SELECTIONS.csv')
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in selection:
        if int(r['source_metrics'])>100 and int(r['gold_positive']):
            groups[(r['panel'],r['variant'])][r['arm']].append(r)
    variance=[]; offdiag_checks=0
    for (panel,variant),arms in sorted(groups.items()):
        stats={}
        for arm,rr in arms.items():
            bycase=collections.defaultdict(dict)
            for r in rr:
                if r['seed_id'] in bycase[r['case_key']]:
                    raise RuntimeError('DUPLICATE_CASE_SEED')
                bycase[r['case_key']][r['seed_id']]=int(r['witness'])
            seedsets={tuple(sorted(v)) for v in bycase.values()}
            assert len(seedsets)==1
            labels=next(iter(seedsets));assert len(labels)==16
            Ys=[[v[s] for s in labels] for _,v in sorted(bycase.items())]
            means=[F(sum(y),16) for y in Ys]
            T=[sum(y[j] for y in Ys) for j in range(16)]
            mu=F(sum(T),16)
            var=sum((F(x)-mu)**2 for x in T)/16
            assert var==sum(F(x*x,16) for x in T)-mu*mu
            marginal=sum(p*(1-p) for p in means)
            offdiag=F(0)
            for i in range(len(Ys)):
                for j in range(i+1,len(Ys)):
                    offdiag+=2*(sum(F(a*b,16) for a,b in zip(Ys[i],Ys[j]))-means[i]*means[j])
                    offdiag_checks+=1
            assert var==marginal+offdiag
            stats[arm]=dict(positive_cases=len(Ys),positive_settings=len(rr),hits=sum(T),
                variance=var,marginal_variance=marginal,offdiagonal_sum=offdiag)
        s,l=stats['SHARED'],stats['CASE_LOCAL']
        dv=s['variance']-l['variance'];dm=s['marginal_variance']-l['marginal_variance'];dc=s['offdiagonal_sum']-l['offdiagonal_sum']
        assert dv==dm+dc and dv>0
        variance.append(dict(panel=panel,variant=variant,
            arms={a:{k:(number(v) if isinstance(v,F) else v) for k,v in x.items()} for a,x in sorted(stats.items())},
            total_drop=number(dv),marginal_drop=number(dm),offdiagonal_drop=number(dc),
            offdiagonal_share_of_drop=number(dc/dv)))
    result={
        'analysis_date':'2026-11-01','basis':'Stored outputs; post-hoc arithmetic diagnostics',
        'source_zip_sha256':PINS,'input_result_rows_read':len(records),
        'large_case_count_fraction_settings':6400,'large_cases_across_two_panels':50,
        'paired_settings':3200,'fraction_reconstruction_max_abs_difference':max_difference,
        'gold_count_reconstruction':'Unique integer per panel/case; same COUNT predicate as FRACTION',
        'counts_and_fractions':summaries,'variance_decomposition':variance,'example':examples,
        'checks':{'composition_identities':6400,'expansion_identities':6400,
                  'equal_valid_denominator_inversion_criteria':checked_criteria,
                  'expansion_vs_fraction_paired_directions':expansion_pair_checks,
                  'offdiagonal_pairs_recomputed':offdiag_checks},
        'interpretation':{
          'covariance_fraction_is_causal':False,'diagnostics_are_preregistered':False,
          'repeated_rows_are_independent':False,
          'scope_expanded_count_is_new_tested_deployment_method':False,
          'source_valid_count_availability':'N is evaluation-oracle metadata, not assumed in model input',
          'normalization_comparison':'Per same case/predicate only; cross-case aggregate weighting can differ',
          'new_sampling_or_database_theorem_claimed':False},
        'new_raw_queries':0,'new_data':0,'new_models_or_inference':0,
        'tokenization_packing_network_install_calls':0}
    args.out.mkdir(parents=True,exist_ok=True)
    emit_json(args.out/'RESULTS.json',result)
    table=[]
    for x in summaries:
        y={k:v for k,v in x.items() if k!='means'}
        y['type']='COUNT_FRACTION'
        for k,v in x['means'].items(): y[k]=v['value']
        table.append(y)
    for x in variance:
        y={'type':'VARIANCE','panel':x['panel'],'variant':x['variant'],'regime':'FIXED'}
        for k in ('total_drop','marginal_drop','offdiagonal_drop','offdiagonal_share_of_drop'):y[k]=x[k]['value']
        table.append(y)
    keys=['type','panel','regime','variant']+sorted(set().union(*(r.keys() for r in table))-{'type','panel','regime','variant'})
    buf=io.StringIO(newline='');w=csv.DictWriter(buf,keys,lineterminator='\n');w.writeheader();w.writerows(table)
    blob=buf.getvalue().encode('utf-8-sig');out=args.out/'ANALYSIS_TABLES.csv'
    if out.exists() and out.read_bytes()!=blob:raise RuntimeError('OUTPUT_CONFLICT')
    out.write_bytes(blob)
    print('STORED_OUTPUT_MECHANISM_ANALYSIS=COMPLETE')
    print('COUNT_FRACTION_SETTINGS=6400;PAIRED_SETTINGS=3200')
    print('ORIGINAL_QUERY_MODEL_TOKENIZATION_AND_INSTALL=NOT_RUN')

if __name__=='__main__':
    main()
