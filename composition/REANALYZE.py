"""Read-only identity-aware composition diagnostics of stored witness outcomes.
Python 3.8+ standard library. No source query/model/packing/network calls.
Every nonempty subset of the existing budget-limited positive cases is reported.
The constructed AND/OR workloads are post-hoc diagnostics, NOT observed workloads.
"""
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import zipfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


def sha(b):
    return hashlib.sha256(b).hexdigest()


def csvrows(b):
    return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'))))


def pop(n):
    return bin(n).count('1')


def exact(x):
    x = Fraction(x)
    return {'fraction': str(x), 'value': float(x)}


def choose(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


def writecsv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair_detail(cases, i, j, masks, counts, labels):
    arms = {}
    for arm in ('SHARED', 'CASE_LOCAL'):
        a, b = masks[arm][i], masks[arm][j]
        pa, pb = Fraction(pop(a), 16), Fraction(pop(b), 16)
        joint = Fraction(pop(a & b), 16)
        union = Fraction(pop(a | b), 16)
        assert joint + union == pa + pb
        arms[arm] = dict(case_a_hits=pop(a), case_b_hits=pop(b),
                         and_hits=pop(a & b), or_hits=pop(a | b),
                         marginal_product=exact(pa * pb), covariance=exact(joint - pa * pb),
                         case_a_hit_labels=[s for n,s in enumerate(labels) if a & (1 << n)],
                         case_b_hit_labels=[s for n,s in enumerate(labels) if b & (1 << n)])
    s, l = arms['SHARED'], arms['CASE_LOCAL']
    prod = Fraction(l['marginal_product']['fraction'])-Fraction(s['marginal_product']['fraction'])
    cov = Fraction(l['covariance']['fraction'])-Fraction(s['covariance']['fraction'])
    joint = Fraction(l['and_hits']-s['and_hits'], 16)
    assert prod + cov == joint
    return dict(case_a=cases[i], case_b=cases[j],
                fixed_counts=[counts[cases[i]], counts[cases[j]]], labels=labels,
                arms=arms, local_minus_shared=dict(marginal_product=exact(prod),
                covariance=exact(cov), and_frequency=exact(joint)))


def analyze(input_path, output):
    with zipfile.ZipFile(input_path) as z:
        if z.testzip() is not None:
            raise ValueError('INPUT_ZIP_CRC_FAILED')
        prov=json.loads(z.read('PROVENANCE.json'))
        for name,pin in prov['member_sha256'].items():
            if sha(z.read(name)) != pin:
                raise ValueError('INPUT_MEMBER_HASH_MISMATCH: '+name)
        raw=z.read('FIXED_COUNT_SELECTIONS.csv')
        records=csvrows(raw)
        originals={}
        for name,panel,force_arm in [('DEV_SHARED.csv','DEVELOPMENT','SHARED'),
                                    ('DEV_CASE_LOCAL.csv','DEVELOPMENT','CASE_LOCAL'),
                                    ('TRANSFER_SELECTIONS.csv','TRANSFER',None)]:
            for r in csvrows(z.read(name)):
                k=(panel,r['case_key'],r['variant'],force_arm or r['arm'],r['seed_id'])
                if k in originals:
                    raise ValueError('DUPLICATE_ORIGINAL')
                originals[k]=r
    assert len(records)==len(originals)==9600
    allcases=defaultdict(set); groups=defaultdict(list); matrices=defaultdict(dict)
    fixed_counts={}; seen=set()
    for r in records:
        p,c,v,a,s=[r[k] for k in ['panel','case_key','variant','arm','seed_id']]
        key=(p,c,v,a,s)
        assert key not in seen;seen.add(key)
        o=originals[key]
        for lhs,rhs in [('source_metrics','source_metrics'),('gold_positive','gold_positive'),
                        ('original_selected_metrics','selected_metrics'),('original_witness','witness')]:
            assert int(r[lhs])==int(o[rhs]), (key,lhs)
        assert int(r['witness']) in (0,1) and int(r['witness'])<=int(r['original_witness'])
        allcases[p].add(c);groups[(p,c,v)].append(r)
        fixed_counts[(p,c,v)]=int(r['fixed_count'])
        if int(r['source_metrics'])<=100 or r['gold_positive']!='1':
            continue
        for analysis,col in [('PRIMARY','original_witness'),('FIXED_COUNT','witness')]:
            matrices[(analysis,p,v)][(a,c,s)]=int(r[col])
    assert all(len(c)==75 for c in allcases.values())
    assert not allcases['DEVELOPMENT'].intersection(allcases['TRANSFER'])
    for rows in groups.values():
        assert len(rows)==32 and len({int(r['fixed_count']) for r in rows})==1
        assert int(rows[0]['fixed_count'])==min(int(r['original_selected_metrics']) for r in rows)

    profile=[]; subset_rows=[]; comparisons=[]; counterexample=None
    for analysis in ('PRIMARY','FIXED_COUNT'):
        for panel in ('DEVELOPMENT','TRANSFER'):
            for variant in ('B01','B02'):
                m=matrices[(analysis,panel,variant)]
                cases=sorted({c for a,c,s in m});labels=sorted({s for a,c,s in m})
                M=len(cases);assert M==(11 if panel=='DEVELOPMENT' else 10)
                assert len(labels)==16 and len(m)==M*32
                masks={a:[sum(m[(a,c,s)] << n for n,s in enumerate(labels)) for c in cases]
                       for a in ('SHARED','CASE_LOCAL')}
                counts={c:fixed_counts[(panel,c,variant)] for c in cases}
                totals={a:[sum(m[(a,c,s)] for c in cases) for s in labels] for a in masks}
                case_details=[dict(case_key=c,shared_hits=pop(masks['SHARED'][i]),
                                   local_hits=pop(masks['CASE_LOCAL'][i]),fixed_count=counts[c])
                              for i,c in enumerate(cases)]
                equal_total=[]
                for n,s in enumerate(labels):
                    sh={c for i,c in enumerate(cases) if masks['SHARED'][i] & (1<<n)}
                    lo={c for i,c in enumerate(cases) if masks['CASE_LOCAL'][i] & (1<<n)}
                    if len(sh)==len(lo) and sh!=lo:
                        equal_total.append(dict(seed=s,total_hits=len(sh),
                                                shared_only=sorted(sh-lo),local_only=sorted(lo-sh)))
                own_profiles=[]; reversal_pairs=[]
                for q in range(1,M+1):
                    rows=[]
                    for idx in itertools.combinations(range(M),q):
                        hits={}
                        for arm in masks:
                            aa=(1<<16)-1;oo=0
                            for i in idx:
                                aa &= masks[arm][i];oo |= masks[arm][i]
                            hits[arm]=(pop(aa),pop(oo))
                        ds=hits['CASE_LOCAL'][0]-hits['SHARED'][0]
                        marg=[(pop(masks['SHARED'][i]),pop(masks['CASE_LOCAL'][i])) for i in idx]
                        noninferior=all(l>=s for s,l in marg)
                        strict=all(l>s for s,l in marg)
                        row=dict(analysis=analysis,panel=panel,variant=variant,subset_size=q,
                                 case_keys='|'.join(cases[i] for i in idx),seed_denominator=16,
                                 shared_and_hits=hits['SHARED'][0],local_and_hits=hits['CASE_LOCAL'][0],
                                 shared_or_hits=hits['SHARED'][1],local_or_hits=hits['CASE_LOCAL'][1],
                                 all_local_marginals_at_least_shared=int(noninferior),
                                 all_local_marginals_strictly_higher=int(strict),
                                 local_marginal_dominance_and_joint_loss=int(noninferior and ds<0))
                        rows.append(row);subset_rows.append(row)
                        if q==2 and noninferior and ds<0:
                            detail=pair_detail(cases,*idx,masks,counts,labels)
                            detail['both_marginals_strictly_higher']=strict
                            reversal_pairs.append(detail)
                            if (analysis,panel,variant,detail['case_a'],detail['case_b']) == (
                                'FIXED_COUNT','TRANSFER','B02','6862fae3f3c27725','c1c73991d617f258'):
                                counterexample=detail
                    den=16*len(rows)
                    sums={arm:{op:sum(r[arm.lower()+'_'+op+'_hits'] for r in rows)
                               for op in ('and','or')} for arm in ('SHARED','LOCAL')}
                    # Closed combinatorial identities independently check all named-subset enumeration.
                    for arm,key in [('SHARED','SHARED'),('CASE_LOCAL','LOCAL')]:
                        assert sums[key]['and']==sum(choose(t,q) for t in totals[arm])
                        assert sums[key]['or']==sum(choose(M,q)-choose(M-t,q) for t in totals[arm])
                    st=dict(analysis=analysis,panel=panel,variant=variant,positive_cases=M,
                            subset_size=q,distinct_subsets=len(rows),seed_subsets_denominator=den,
                            shared_and_hits=sums['SHARED']['and'],local_and_hits=sums['LOCAL']['and'],
                            shared_or_hits=sums['SHARED']['or'],local_or_hits=sums['LOCAL']['or'])
                    for arm in ('shared','local'):
                        for op in ('and','or'):
                            val=Fraction(st[arm+'_'+op+'_hits'],den)
                            st[arm+'_'+op+'_frequency_exact']=str(val)
                            st[arm+'_'+op+'_frequency']=float(val)
                    for op in ('and','or'):
                        for relation,cmp in [('shared_better',lambda a,b:a>b),
                                             ('local_better',lambda a,b:a<b),('tied',lambda a,b:a==b)]:
                            st[op+'_'+relation+'_subsets']=sum(cmp(r['shared_'+op+'_hits'],r['local_'+op+'_hits']) for r in rows)
                    st['marginal_dominance_and_joint_loss_subsets']=sum(r['local_marginal_dominance_and_joint_loss'] for r in rows)
                    st['strict_marginal_dominance_and_joint_loss_subsets']=sum(
                        r['all_local_marginals_strictly_higher'] and r['local_marginal_dominance_and_joint_loss'] for r in rows)
                    profile.append(st);own_profiles.append(st)
                comparisons.append(dict(analysis=analysis,panel=panel,variant=variant,
                    positive_cases=M,labels=labels,case_marginals=case_details,
                    shared_totals=totals['SHARED'],local_totals=totals['CASE_LOCAL'],
                    subset_sizes_favoring_shared_and=[r['subset_size'] for r in own_profiles if r['shared_and_hits']>r['local_and_hits']],
                    subset_sizes_favoring_local_and=[r['subset_size'] for r in own_profiles if r['shared_and_hits']<r['local_and_hits']],
                    subset_sizes_tied_and=[r['subset_size'] for r in own_profiles if r['shared_and_hits']==r['local_and_hits']],
                    average_or_local_noninferior_all_sizes=all(r['local_or_hits']>=r['shared_or_hits'] for r in own_profiles),
                    named_pairs=choose(M,2),pair_marginal_joint_reversals=reversal_pairs,
                    equal_total_but_different_case_sets=equal_total))
    assert len(profile)==84 and len(subset_rows)==12280 and len(comparisons)==8
    assert counterexample is not None
    # Finite code checks, not additional empirical data.
    tested=0
    for M in range(1,9):
        for y in itertools.product((0,1),repeat=M):
            t=sum(y)
            for q in range(1,M+1):
                subs=list(itertools.combinations(range(M),q))
                assert sum(all(y[i] for i in s) for s in subs)==choose(t,q)
                assert sum(any(y[i] for i in s) for s in subs)==choose(M,q)-choose(M-t,q)
                tested+=1
    result=dict(status='COMPLETE_POST_HOC_IDENTITY_AWARE_BOOLEAN_COMPOSITION',
        evidence=dict(input_zip_sha256=sha(input_path.read_bytes()),selection_member_sha256=sha(raw),
                      selection_rows=9600,original_rows_crosschecked=9600,count_matching_groups=300,
                      positive_cases={'DEVELOPMENT':11,'TRANSFER':10},source_provenance=prov),
        computations=dict(comparisons=8,subset_size_profile_rows=len(profile),
                          subset_comparison_rows=len(subset_rows),pair_comparisons=400,
                          finite_formula_codechecks=tested, source_query_reruns=0,
                          packing_tokenization_model_calls=0,new_cases_or_seeds=0),
        counterexample=counterexample,comparisons=comparisons,
        interpretation=dict(analysis='Post-hoc Boolean compositions of stored per-case EXISTS witnesses',
          workload='Uniform equal weight over all q-subsets is an analytic reference, not an observed operating workload',
          events='Cases were not shown to be logically related incidents. AND/OR describe imposed preservation requirements only',
          independence='No independence assumption or product approximation used for reported empirical AND/OR',
          uncertainty='16 fixed labels; exact enumeration does not create independent data or a population probability estimate',
          identity='Full T distributions characterize symmetric count utilities, not requirements for named case subsets',
          novelty='Known dependence/provenance identities; contribution is telemetry-specific empirical diagnosis',
          modification='Original source/results/environments unchanged'))
    output.mkdir(parents=True,exist_ok=True)
    (output/'RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    writecsv(output/'GROUP_PROFILE.csv',profile)
    writecsv(output/'SUBSET_RESULTS.csv',subset_rows)
    print('COMPOSITION_REANALYSIS=COMPLETE_DESCRIPTIVE_ONLY')
    print('EXISTING_SELECTIONS=9600;SUBSET_COMPARISONS=12280;SIZE_PROFILES=84')
    print('ORIGINAL_QUERIES_MODEL_PACKING_AND_INSTALL_RUN=0')
    print('OUTPUT='+str(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).with_name('INPUT_EVIDENCE.zip'))
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('recomputed'))
    args=parser.parse_args()
    analyze(args.input,args.output)
