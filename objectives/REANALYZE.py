"""Read-only analysis of frozen witness indicators; Python 3.8+, stdlib only.
No model, source-query, tokenization, packing, or network calls.
The ordered thresholds and utility bases describe the finite empirical seed panel.
"""
import argparse
import csv
import hashlib
import io
import itertools
import json
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(data):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))


def scalar(x):
    x = Fraction(x)
    return {'exact': str(x), 'value': float(x)}


def avg(values):
    return Fraction(sum(values), len(values))


def variance(values):
    mean = avg(values)
    return sum((Fraction(x)-mean)**2 for x in values)/len(values)


def compare(s, l, m):
    if len(s) != len(l) or not s:
        raise ValueError('unmatched seed panels')
    n = len(s)
    tails_s = [sum(x >= r for x in s) for r in range(1, m+1)]
    tails_l = [sum(x >= r for x in l) for r in range(1, m+1)]
    gaps = [b-a for a,b in zip(tails_s,tails_l)]
    cap_gaps = [sum(min(x,r) for x in l)-sum(min(x,r) for x in s)
                for r in range(1, m+1)]
    # Integer-basis checks independently connect caps to survival sums.
    assert all(cap_gaps[r-1] == sum(gaps[:r]) for r in range(1,m+1))
    assert sum(tails_s) == sum(s) and sum(tails_l) == sum(l)
    return dict(
        shared_mean=scalar(avg(s)), local_mean=scalar(avg(l)),
        shared_recall=scalar(avg(s)/m), local_recall=scalar(avg(l)/m),
        shared_variance=scalar(variance(s)), local_variance=scalar(variance(l)),
        shared_all_miss=sum(x==0 for x in s), local_all_miss=sum(x==0 for x in l),
        shared_all_hit=sum(x==m for x in s), local_all_hit=sum(x==m for x in l),
        shared_tails=tails_s, local_tails=tails_l,
        local_better_thresholds=[r for r,d in enumerate(gaps,1) if d>0],
        tied_thresholds=[r for r,d in enumerate(gaps,1) if d==0],
        shared_better_thresholds=[r for r,d in enumerate(gaps,1) if d<0],
        empirical_first_order_local_over_shared=all(x>=0 for x in gaps),
        empirical_first_order_shared_over_local=all(x<=0 for x in gaps),
        empirical_survival_curves_cross=min(gaps)<0<max(gaps),
        empirical_increasing_concave_local_over_shared=all(x>=0 for x in cap_gaps),
        cap_gain_numerators=cap_gaps, denominator=n)


def basis_selftest():
    checked=0
    # Nondecreasing integer utility increments, finite telescoping identity.
    for m in range(1,7):
        for ds in itertools.product(range(3), repeat=m):
            u=[0]
            for a in ds: u.append(u[-1]+a)
            for t in range(m+1):
                assert u[t] == sum(ds[r-1]*int(t>=r) for r in range(1,m+1))
                if all(ds[i]>=ds[i+1] for i in range(m-1)):
                    w=[ds[r-1]-ds[r] for r in range(1,m)] + [ds[-1]]
                    assert u[t] == sum(w[r-1]*min(t,r) for r in range(1,m+1))
                checked+=1
    return checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(__file__).with_name('INPUT_EVIDENCE.zip'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('recomputed'))
    a=parser.parse_args()
    with zipfile.ZipFile(a.input) as z:
        if z.testzip() is not None: raise RuntimeError('input ZIP CRC failure')
        prov=json.loads(z.read('PROVENANCE.json'))
        for name,pin in prov['member_sha256'].items():
            if sha(z.read(name)) != pin: raise RuntimeError('input hash mismatch: '+name)
        fixed=read_csv(z.read('FIXED_COUNT_SELECTIONS.csv'))
        old={}
        for name,panel,force_arm in [('DEV_SHARED.csv','DEVELOPMENT','SHARED'),
                                    ('DEV_CASE_LOCAL.csv','DEVELOPMENT','CASE_LOCAL'),
                                    ('TRANSFER_SELECTIONS.csv','TRANSFER',None)]:
            for row in read_csv(z.read(name)):
                arm=force_arm or row['arm']
                key=(panel,row['case_key'],row['variant'],arm,row['seed_id'])
                if key in old: raise ValueError('duplicate original selection')
                old[key]=row
        summary=read_csv(z.read('WITNESS_SUMMARY.csv'))
    assert len(old)==9600 and len(fixed)==9600
    cases=defaultdict(set); groups=defaultdict(list); golds=defaultdict(set)
    seen=set(); matrices={}; positive={}
    for row in fixed:
        panel,case,v,arm,seed=[row[k] for k in ['panel','case_key','variant','arm','seed_id']]
        key=(panel,case,v,arm,seed)
        assert key not in seen; seen.add(key)
        source=old[key]
        for lhs,rhs in [('original_witness','witness'),('original_selected_metrics','selected_metrics'),
                        ('gold_positive','gold_positive'),('source_metrics','source_metrics')]:
            assert int(row[lhs])==int(source[rhs]), (key,lhs)
        y0,y1=int(row['original_witness']),int(row['witness'])
        assert y0 in (0,1) and y1 in (0,1) and y1<=y0
        cases[panel].add(case); groups[(panel,case,v)].append(row)
        golds[(panel,case)].add((int(row['gold_positive']),int(row['source_metrics'])))
        if int(row['source_metrics'])<=100 or int(row['gold_positive'])!=1: continue
        positive.setdefault(panel,set()).add(case)
        for analysis,col in [('PRIMARY','original_witness'),('FIXED_COUNT','witness')]:
            g=(analysis,panel,v,arm)
            matrices.setdefault(g,{}).setdefault(seed,{})[case]=int(row[col])
    assert all(len(x)==75 for x in cases.values())
    assert not (cases['DEVELOPMENT'] & cases['TRANSFER'])
    assert all(len(x)==1 for x in golds.values())
    assert len(positive['DEVELOPMENT'])==11 and len(positive['TRANSFER'])==10
    for group,rows in groups.items():
        assert len(rows)==32 and len({int(x['fixed_count']) for x in rows})==1
        assert int(rows[0]['fixed_count'])==min(int(x['original_selected_metrics']) for x in rows)
    threshold_rows=[]; seed_rows=[]; comparisons=[]
    for analysis in ['PRIMARY','FIXED_COUNT']:
        for panel in ['DEVELOPMENT','TRANSFER']:
            for v in ['B01','B02']:
                sm=matrices[(analysis,panel,v,'SHARED')]; lm=matrices[(analysis,panel,v,'CASE_LOCAL')]
                assert set(sm)==set(lm) and len(sm)==16
                labels=sorted(sm); m=len(positive[panel])
                assert all(set(sm[s])==positive[panel] and set(lm[s])==positive[panel] for s in labels)
                st=[sum(sm[s].values()) for s in labels];lt=[sum(lm[s].values()) for s in labels]
                co=compare(st,lt,m)
                co.update(analysis=analysis,panel=panel,variant=v,positive_cases=m,
                          labels=labels,shared_counts=st,local_counts=lt)
                loo=[]
                for i,lab in enumerate(labels):
                    r=compare(st[:i]+st[i+1:],lt[:i]+lt[i+1:],m)
                    loo.append(dict(omitted_seed=lab,curves_cross=r['empirical_survival_curves_cross'],
                                    increasing_concave=r['empirical_increasing_concave_local_over_shared'],
                                    min_cap_gain=scalar(Fraction(min(r['cap_gain_numerators']),15))))
                co['leave_one_seed']=loo
                co['crossing_retained_seed_deletions']=sum(x['curves_cross'] for x in loo)
                co['increasing_concave_retained_seed_deletions']=sum(x['increasing_concave'] for x in loo)
                if analysis=='FIXED_COUNT':
                    for arm,t in [('SHARED',st),('CASE_LOCAL',lt)]:
                        ref=[x for x in summary if x['panel']==panel and x['variant']==v and x['arm']==arm][0]
                        assert Fraction(ref['hit_count_variance_exact'])==variance(t)
                        assert int(ref['witness_hits'])==sum(t)
                # Monotone/capped objectives calculated two ways, including linear/AND examples.
                utilities={'linear':list(range(m+1)), 'complete':[0]*m+[1],
                           'diminishing':[Fraction(t,m+t) for t in range(m+1)]}
                for name,u in utilities.items():
                    for t,tails in [(st,co['shared_tails']),(lt,co['local_tails'])]:
                        direct=avg([u[x] for x in t])
                        reconstructed=u[0]+sum((u[r]-u[r-1])*Fraction(tails[r-1],16)
                                             for r in range(1,m+1))
                        assert direct==reconstructed
                for r in range(1,m+1):
                    ss=co['shared_tails'][r-1];ll=co['local_tails'][r-1]
                    threshold_rows.append(dict(analysis=analysis,panel=panel,variant=v,
                        positive_cases=m,threshold=r,shared_meeting_threshold=ss,
                        local_meeting_threshold=ll,seed_denominator=16,
                        attainment_difference_exact=str(Fraction(ll-ss,16)),
                        attainment_difference=float(Fraction(ll-ss,16)),
                        capped_gain_exact=str(Fraction(co['cap_gain_numerators'][r-1],16))))
                for lab,s,l in zip(labels,st,lt):
                    seed_rows.append(dict(analysis=analysis,panel=panel,variant=v,seed=lab,
                                          positive_cases=m,shared_witness_cases=s,local_witness_cases=l))
                comparisons.append(co)
    out=dict(status='COMPLETE_POST_HOC_EMPIRICAL_OBJECTIVE_PROFILE',
             data_basis='Frozen case-by-seed witness outcomes; NOT new runs or operational batches',
             source_provenance=prov,selection_rows_checked=len(fixed),
             original_selection_rows_cross_checked=len(old),count_matching_groups=len(groups),
             positive_case_labels_used=21,threshold_comparison_rows=len(threshold_rows),
             seed_count_rows=len(seed_rows),comparisons=comparisons,
             checks=dict(fixed_summary_parity=True,primary_witness_exact_parity=True,
                         preserved_monotone_shortening=True,
                         utility_basis_enumerations=basis_selftest(),
                         telescoping_and_capped_identities=True),
             interpretation=dict(
                weighting='16 existing labels have equal empirical weight within a panel; panels are not pooled',
                all_hit='Every positive case retains at least one witness; not every metric or every query is preserved',
                first_order='A property of the empirical distributions only; no population dominance test',
                utility_class='Symmetric functions of the number of witness-positive cases; no identities or per-case weights',
                costs='Within the fixed experimental budget/representation; no measured operational cost objective',
                inference='Post-hoc endpoint analysis; not a future risk estimate, operational utility, or p-value',
                changes='0 new sources, seeds, model/query/packing calls or environment changes'))
    assert len(threshold_rows)==84 and len(seed_rows)==128
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'RESULTS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    for name,rows in [('THRESHOLD_PROFILE.csv',threshold_rows),('SEED_COUNTS.csv',seed_rows)]:
        with (a.output/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('STORED_SELECTION_PARITY=9600/9600')
    print('OBJECTIVE_PROFILES=8;THRESHOLD_COMPARISONS=84')
    print('CROSSING_EMPIRICAL_SURVIVAL_PROFILES='+str(sum(x['empirical_survival_curves_cross'] for x in comparisons)))
    print('NEW_EXPERIMENTS_QUERIES_TOKENIZATION_OR_MODEL_RUNS=0')
    print('OUTPUT='+str(a.output))

if __name__=='__main__': main()
