#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QA-SSEC public-origin 75-case adapter compiler, CPU-only.

The build subcommand reads ONLY:
  (1) a pinned public RCAEval cases.parquet index;
  (2) 75 public-origin metrics.parquet plus inject_time.txt pairs;
  (3) the published, hashed source-file manifest.
It never reads the historical frozen adapter or pre-existing B01/B02 as build inputs.

Generated full payloads stay in a local-only private directory; report ZIPs contain
no case names, metric IDs, raw readings, native targets or metric values.
The verify subcommand is a SEPARATE, post-build numerical comparison to the frozen
adapter; do not mistake a passed verifier for an independent machine clean-room run.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

VERSION = 'QA_SSEC_ICDE_EAB_PUBLIC_ORIGIN_COMPILER_V1_20261029'
PUBLIC_INDEX_SHA = 'c49a288920dbba2e8e724679a14636d5c7eb2b45426bba14007ef79a6c0ab1bb'
PUBLIC_SELECTED_SET_SHA = '8ec2fa51f4120cd4dfa46bef12176ff2987c7cc21561261e768cb90c6a8c25f0'
FROZEN_ADAPTER_SHA = '15da5b6f8396de767dd02423fefd20607f6a2981d28c1c956ba17e3fba033959'
PUBLIC_REVISION = 'afeacb11bcc94dadfd1c8f483ee4377b2b8b614e'
WINDOW = 247
EXPECTED_CASES = 75
EXPECTED_METRICS = 8299
TOL_ABS = 1e-8
TOL_REL = 1e-7
FIELDS = ('metric_id','normal_missing_rate','faulty_missing_rate','normal_mean','faulty_mean',
          'delta_mean','normal_std','faulty_std','normal_median','faulty_median',
          'delta_median','normal_q25','normal_q75','faulty_q25','faulty_q75',
          'normal_min','normal_max','faulty_min','faulty_max','delta_missing_rate')
NUMERIC_FIELDS = FIELDS[1:]
TOP_MAP={'schema_version':'v','window_timesteps':'w','metrics':'m'}
METRIC_MAP={
    'metric_id':'i','normal_missing_rate':'nmr','faulty_missing_rate':'fmr',
    'normal_mean':'nm','faulty_mean':'fm','delta_mean':'dm','normal_std':'ns',
    'faulty_std':'fs','normal_median':'nmed','faulty_median':'fmed',
    'delta_median':'dmed','normal_q25':'n25','normal_q75':'n75',
    'faulty_q25':'f25','faulty_q75':'f75','normal_min':'nmin',
    'normal_max':'nmax','faulty_min':'fmin','faulty_max':'fmax',
    'delta_missing_rate':'dmr'
}
REV_TOP={v:k for k,v in TOP_MAP.items()}
REV_METRIC={v:k for k,v in METRIC_MAP.items()}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):
            h.update(chunk)
    return h.hexdigest()


def stable_json(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def json_report(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def csv_report(path,rows,fields):
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='raise')
        w.writeheader();w.writerows(rows)


def import_pd():
    import numpy as np
    import pandas as pd
    return pd,np


def fp(text):
    return sha_bytes(str(text).encode('utf-8'))


def load_pins(path):
    rows=list(csv.DictReader(Path(path).open('r',encoding='utf-8-sig',newline='')))
    if len(rows)!=150:
        raise ValueError('SOURCE_PIN_COUNT_NOT_150')
    if any(set(row)!=set(('case_sha256','kind','bytes','sha256')) for row in rows):
        raise ValueError('SOURCE_PIN_COLUMNS')
    pins={}
    for r in rows:
        key=(r['case_sha256'],r['kind'])
        if key in pins or len(r['case_sha256'])!=64 or len(r['sha256'])!=64:
            raise ValueError('SOURCE_PIN_DUPLICATE_OR_BAD_DIGEST')
        if r['kind'] not in ('metrics.parquet','inject_time.txt'):
            raise ValueError('SOURCE_PIN_BAD_KIND')
        pins[key]=(int(r['bytes']),r['sha256'].lower())
    if len({k[0] for k in pins})!=75 or sum(x[0] for x in pins.values())!=25059084:
        raise ValueError('SOURCE_PIN_SET_OR_SIZE_NOT_FROZEN')
    return pins


def select_cases(df):
    pd,_=import_pd()
    names=['case','dataset','root_cause_service','fault','repetition','normal_timesteps','faulty_timesteps']
    if len(df)!=735 or not set(names).issubset(df.columns):
        raise ValueError('PUBLIC_INDEX_SHAPE_OR_COLUMNS')
    if df['case'].duplicated().any():
        raise ValueError('PUBLIC_INDEX_DUPLICATE_CASE')
    re1=df.loc[df['dataset'].astype(str).str.startswith('RE1')].copy()
    if len(re1)!=375:
        raise ValueError('PUBLIC_RE1_COUNT_NOT_375')
    re1_eligible=re1.loc[(re1['normal_timesteps']>0)&(re1['faulty_timesteps']>0)].copy()
    if len(re1_eligible)!=373:
        raise ValueError('PUBLIC_RE1_ELIGIBLE_DRIFT')
    group=['dataset','root_cause_service','fault']
    if re1_eligible.duplicated(subset=group+['repetition']).any():
        raise ValueError('AMBIGUOUS_REPETITION')
    re1_eligible=re1_eligible.sort_values(group+['repetition','case'],kind='mergesort')
    selected=re1_eligible.groupby(group,sort=True).head(1).sort_values('case',kind='mergesort')
    if len(selected)!=75:
        raise ValueError('SELECTED_COUNT_NOT_75')
    names=[str(v) for v in selected['case']]
    digest=sha_bytes(('\n'.join(sorted(names))+'\n').encode('utf-8'))
    if digest!=PUBLIC_SELECTED_SET_SHA:
        raise ValueError('SELECTED_CASE_SET_PIN_MISMATCH')
    if int(selected['normal_timesteps'].min())!=360 or int(selected['faulty_timesteps'].min())!=247:
        raise ValueError('SELECTED_WINDOW_BOUNDS_DRIFT')
    if selected.groupby('dataset').size().to_dict()!={'RE1-OB':25,'RE1-SS':25,'RE1-TT':25}:
        raise ValueError('SELECTED_DATASET_DISTRIBUTION_DRIFT')
    return selected


def find_case_pairs(root,selected):
    root=Path(root)
    if not root.is_dir():
        raise ValueError('RAW_SOURCE_ROOT_MISSING')
    wanted={str(x) for x in selected['case']}
    found=collections.defaultdict(dict)
    for kind in ('metrics.parquet','inject_time.txt'):
        for p in root.rglob(kind):
            if not p.is_file():
                continue
            case=p.parent.name
            if case in found[kind]:
                raise ValueError('DUPLICATE_PUBLIC_CASE_SOURCE_'+kind)
            found[kind][case]=p
        if set(found[kind])!=wanted:
            raise ValueError('PUBLIC_SOURCE_CASE_SET_MISMATCH_'+kind)
    return {name:(found['metrics.parquet'][name],found['inject_time.txt'][name]) for name in wanted}


def verify_source_pins(pairs,pins):
    actual=set()
    for case,(m,i) in pairs.items():
        for kind,path in (('metrics.parquet',m),('inject_time.txt',i)):
            key=(fp(case),kind)
            if key in actual or key not in pins:
                raise ValueError('PUBLIC_SOURCE_NOT_IN_PINNED_MANIFEST')
            actual.add(key)
            size,digest=pins[key]
            if path.stat().st_size!=size or sha_file(path)!=digest:
                raise ValueError('PUBLIC_SOURCE_HASH_OR_SIZE_MISMATCH')
    if actual!=set(pins):
        raise ValueError('PUBLIC_SOURCE_PIN_SET_INCOMPLETE')


def as_opt_float(value):
    pd,_=import_pd()
    if bool(pd.isna(value)):
        return None
    number=float(value)
    if not math.isfinite(number):
        raise ValueError('SOURCE_STAT_NONFINITE')
    return number


def diff(a,b):
    return None if a is None or b is None else (b-a)


def compute_model_input(df,inject,schema_version='v1',metadata=None):
    pd,np=import_pd()
    if not isinstance(schema_version,str) or not schema_version:
        raise ValueError('INVALID_SCHEMA_VERSION')
    if list(df.columns).count('time')!=1 or len(set(map(str,df.columns)))!=len(df.columns):
        raise ValueError('RAW_TIME_COLUMN_MISSING_OR_DUPLICATES')
    mids=sorted([str(x) for x in df.columns if str(x)!='time'])
    if not mids:
        raise ValueError('NO_METRICS')
    t=df['time']
    if not pd.api.types.is_numeric_dtype(t) or bool(t.isna().any()):
        raise ValueError('RAW_TIME_INVALID')
    if not t.is_monotonic_increasing:
        raise ValueError('RAW_TIME_NON_MONOTONIC')
    if metadata is not None:
        if 'n_timesteps' in metadata and metadata['n_timesteps'] is not None and not bool(pd.isna(metadata['n_timesteps'])):
            if len(df)!=int(metadata['n_timesteps']):
                raise ValueError('RAW_TIMESTEPS_MISMATCH_METADATA')
        if 'inject_time' in metadata and metadata['inject_time'] is not None and not bool(pd.isna(metadata['inject_time'])):
            if float(inject)!=float(metadata['inject_time']):
                raise ValueError('RAW_INJECTION_MISMATCH_METADATA')
    normal=df.loc[t<inject].tail(WINDOW)
    faulty=df.loc[t>=inject].head(WINDOW)
    if len(normal)!=WINDOW or len(faulty)!=WINDOW:
        raise ValueError('RAW_247_WINDOW_INSUFFICIENT')
    stats={}
    for side,frame in (('normal',normal),('faulty',faulty)):
        x=frame[mids]
        if any(not pd.api.types.is_numeric_dtype(x[c]) for c in mids):
            raise ValueError('NONNUMERIC_METRIC_DTYPE')
        if bool(np.isinf(x.to_numpy(dtype=float)).any()):
            raise ValueError('METRIC_INFINITY_UNSPECIFIED')
        stats[side]={
            'missing_rate':x.isna().mean(),
            'mean':x.mean(skipna=True),
            'std':x.std(skipna=True,ddof=0),
            'median':x.median(skipna=True),
            'q25':x.quantile(.25,interpolation='linear'),
            'q75':x.quantile(.75,interpolation='linear'),
            'min':x.min(skipna=True),
            'max':x.max(skipna=True),
        }
    records=[]
    for mid in mids:
        r={'metric_id':mid}
        for side in ('normal','faulty'):
            for stat,series in stats[side].items():
                r[side+'_'+stat]=as_opt_float(series[mid])
        r['delta_mean']=diff(r['normal_mean'],r['faulty_mean'])
        r['delta_median']=diff(r['normal_median'],r['faulty_median'])
        r['delta_missing_rate']=diff(r['normal_missing_rate'],r['faulty_missing_rate'])
        if set(r)!=set(FIELDS):
            raise AssertionError('NEW_MODEL_INPUT_METRIC_KEYS')
        records.append(r)
    return {'schema_version':schema_version,'window_timesteps':WINDOW,'metrics':records}


def to_compact(obj):
    if set(obj)!=set(TOP_MAP):
        raise ValueError('MODEL_INPUT_TOP_SCHEMA')
    rows=[]
    for m in obj['metrics']:
        if set(m)!=set(METRIC_MAP):
            raise ValueError('MODEL_INPUT_METRIC_SCHEMA')
        rows.append({METRIC_MAP[k]:v for k,v in m.items()})
    return {'v':obj['schema_version'],'w':obj['window_timesteps'],'m':rows}


def from_compact(obj):
    if set(obj)!=set(REV_TOP):
        raise ValueError('COMPACT_TOP_SCHEMA')
    rows=[]
    for m in obj['m']:
        if set(m)!=set(REV_METRIC):
            raise ValueError('COMPACT_METRIC_SCHEMA')
        rows.append({REV_METRIC[k]:v for k,v in m.items()})
    return {'schema_version':obj['v'],'window_timesteps':obj['w'],'metrics':rows}


def output_line(record_uid,representation):
    return stable_json({'record_uid':record_uid,'representation':stable_json(representation)})+'\n'


def make_report(private_run,summary,case_rows,mode='build'):
    out=Path(private_run).parent/(Path(private_run).name+'_'+mode+'_audit')
    out.mkdir(parents=True,exist_ok=False)
    names=[]
    p=out/'SUMMARY.json';json_report(p,summary);names.append(p)
    p=out/'CASE_AUDIT.csv';csv_report(p,case_rows,
        ('case_sha256','metric_count','positive_missing_rate_witness','compact_roundtrip','field_mismatches'));names.append(p)
    p=out/'LIMITATIONS.md'
    p.write_text('# Public-origin CPU compiler audit\n\n'
        'The model inputs are computed solely from the 75 pinned public-origin telemetry files, '
        'and the frozen adapter is NOT used by --mode build. '
        'The compiled JSONL payloads remain local-only and must not be uploaded to the chat. '
        'The canonical record_uid is a new deterministic SHA256 public-case fingerprint; '
        'historical JSONL whole-file SHA256 parity is not claimed. '
        'This is a local public-byte-equivalent build, not an independent-machine clean-room run. '
        'B01/B02 tokenizer/coverage and V4 witness results are not rerun here. '\
        'The verify mode separately reads a historical adapter for post-build numerical comparison.\n',
        encoding='utf-8');names.append(p)
    p=out/'MANIFEST_SHA256.csv';csv_report(p,
        [{'filename':f.name,'bytes':f.stat().st_size,'sha256':sha_file(f)} for f in names],
        ('filename','bytes','sha256'));names.append(p)
    dest=out.with_suffix('.zip')
    with zipfile.ZipFile(str(dest),'x',compression=zipfile.ZIP_DEFLATED) as z:
        for f in names:
            z.write(str(f),f.name)
    with zipfile.ZipFile(str(dest)) as z:
        if z.testzip() is not None:
            raise RuntimeError('REPORT_ZIP_CRC')
    return dest,sha_file(dest)


def run_build(args):
    pd,_=import_pd()
    index=Path(args.index)
    if not index.is_file() or sha_file(index)!=PUBLIC_INDEX_SHA:
        raise ValueError('PUBLIC_INDEX_SHA_MISMATCH_OR_MISSING')
    idx=pd.read_parquet(str(index))
    selected=select_cases(idx)
    pins=load_pins(args.pins)
    pairs=find_case_pairs(args.source_root,selected)
    verify_source_pins(pairs,pins)
    base=Path(args.output_root)
    base.mkdir(parents=True,exist_ok=True)
    stamp=dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    local=base/(stamp+'_PUBLIC_ORIGIN_PRIVATE_BUILD')
    local.mkdir(parents=True,exist_ok=False)
    a_file=local/'GENERATED_PRIVATE_ADAPTER_INPUTS.jsonl'
    b1_file=local/'GENERATED_PRIVATE_B01.jsonl'
    b2_file=local/'GENERATED_PRIVATE_B02.jsonl'
    nmetric=0;ncase=0;roundtrip=0;positive_cases=0;large_positive=0
    case_rows=[]
    with a_file.open('w',encoding='utf-8',newline='\n') as af, \
         b1_file.open('w',encoding='utf-8',newline='\n') as f1, \
         b2_file.open('w',encoding='utf-8',newline='\n') as f2:
        for _,sel in selected.iterrows():
            case=str(sel['case'])
            mfile,ifile=pairs[case]
            inject=int(ifile.read_text(encoding='utf-8-sig').strip())
            df=pd.read_parquet(str(mfile))
            mi=compute_model_input(df,inject,schema_version=args.schema_version,
                                   metadata=sel.to_dict())
            co=to_compact(mi)
            if stable_json(from_compact(co))!=stable_json(mi):
                raise ValueError('COMPACT_ROUNDTRIP_FAIL')
            uid='public_'+fp(case)
            af.write(stable_json({'record_uid':uid,'model_input':mi})+'\n')
            f1.write(output_line(uid,mi))
            f2.write(output_line(uid,co))
            roundtrip+=1;ncase+=1;nmetric+=len(mi['metrics'])
            positive=int(any(m['faulty_missing_rate']>m['normal_missing_rate'] for m in mi['metrics']))
            positive_cases+=positive
            if len(mi['metrics'])>=199:
                large_positive+=positive
            case_rows.append({'case_sha256':fp(case),'metric_count':len(mi['metrics']),
                              'positive_missing_rate_witness':positive,'compact_roundtrip':1,
                              'field_mismatches':''})
    if ncase!=75 or nmetric!=8299 or positive_cases!=14 or large_positive!=11 or roundtrip!=75:
        raise RuntimeError('PUBLIC_MODEL_INPUT_DISTRIBUTION_DRIFT')
    summary={
      'version':VERSION,'final_status':'PASS_PUBLIC_BYTE_EQUIVALENT_LOCAL_RAW_BUILD_ONLY',
      'cases':ncase,'metric_units':nmetric,'metric_numeric_field_units':nmetric*19,
      'public_index_sha256':PUBLIC_INDEX_SHA,'selected_set_sha256':PUBLIC_SELECTED_SET_SHA,
      'source_files_sha256_verified':len(pins),'source_files_total_bytes':sum(v[0] for v in pins.values()),
      'source_revision':PUBLIC_REVISION,'normal_rows':WINDOW,'faulty_rows':WINDOW,
      'metric_std_ddof':0,'positive_missing_rate_cases':positive_cases,
      'positive_large_cases':large_positive,'compact_roundtrip_cases':roundtrip,
      'generated_schema_version':args.schema_version,
      'generated_adapter_sha256':sha_file(a_file),'generated_b01_sha256':sha_file(b1_file),
      'generated_b02_sha256':sha_file(b2_file),
      'generation_input_frozen_adapter':False,'generation_input_frozen_b01_b02':False,
      'frozen_adapter_numeric_parity':'NOT_TESTED_IN_BUILD',
      'historical_adapter_full_byte_parity':'NOT_TESTED',
      'token_budget_evaluation':'NOT_RERUN','independent_machine_cleanroom':'NOT_TESTED',
      'gpu_used':False,'new_lm_training':False,'new_lm_inference':False,
      'raw_values_in_report_zip':False,
    }
    result,hashvalue=make_report(local,summary,case_rows)
    print('VERSION='+VERSION)
    print('FINAL_STATUS='+summary['final_status'])
    print('PUBLIC_SOURCE_SHA256=PASS_150_OF_150')
    print('SELECTED_CASES='+str(ncase))
    print('METRIC_UNITS='+str(nmetric))
    print('NUMERIC_CELLS='+str(nmetric*19))
    print('POSITIVE_CASES='+str(positive_cases))
    print('COMPACT_ROUNDTRIP='+str(roundtrip)+'/75')
    print('GPU_USED=0')
    print('GENERATED_PRIVATE_ADAPTER='+str(a_file))
    print('GENERATED_PRIVATE_B01='+str(b1_file))
    print('GENERATED_PRIVATE_B02='+str(b2_file))
    print('OUTPUT_ZIP='+str(result))
    print('OUTPUT_SHA256='+hashvalue)
    return 0


def nearly_equal(a,b):
    if a is None or b is None:
        return a is None and b is None
    try:
        a=float(a);b=float(b)
    except (ValueError,TypeError):
        return False
    if not math.isfinite(a) or not math.isfinite(b):
        return False
    return abs(a-b)<=TOL_ABS+TOL_REL*max(abs(a),abs(b))


def read_adapter(path):
    data=[]
    with Path(path).open('r',encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip():continue
            r=json.loads(line)
            mi=r['model_input']
            if set(mi)!={'schema_version','window_timesteps','metrics'}:
                raise ValueError('ADAPTER_TOP_KEYS_INVALID')
            if mi['window_timesteps']!=WINDOW:
                raise ValueError('ADAPTER_WINDOW_INVALID')
            mids=[str(m['metric_id']) for m in mi['metrics']]
            if mids!=sorted(mids) or len(set(mids))!=len(mids):
                raise ValueError('ADAPTER_METRIC_SORT_OR_DUPLICATE')
            if any(set(m)!=set(FIELDS) for m in mi['metrics']):
                raise ValueError('ADAPTER_METRIC_FIELDS_INVALID')
            data.append(mi)
    if len(data)!=75 or sum(len(mi['metrics']) for mi in data)!=8299:
        raise ValueError('ADAPTER_75_8299_DRIFT')
    return data


def match_rebuilt_to_frozen(built,frozen):
    # Use metric schema + source-derived numeric fingerprints to avoid relying on
    # historical record_uid / array order. There can be repeated schema signatures.
    frozen_groups=collections.defaultdict(list)
    for j,mi in enumerate(frozen):
        key=fp('\n'.join(str(m['metric_id']) for m in mi['metrics'])+'\n')
        frozen_groups[key].append(j)
    matches={};used=set();ambiguous=0
    for i,mi in enumerate(built):
        ids=[str(m['metric_id']) for m in mi['metrics']]
        key=fp('\n'.join(ids)+'\n')
        candidates=[]
        for j in frozen_groups.get(key,[]):
            raw_m=frozen[j]['metrics']
            ok=True
            for a,b in zip(mi['metrics'],raw_m):
                if a['metric_id']!=b['metric_id']:
                    ok=False;break
                if not all(nearly_equal(a[f],b[f]) for f in ('normal_mean','faulty_mean',
                         'normal_missing_rate','faulty_missing_rate')):
                    ok=False;break
            if ok:candidates.append(j)
        if len(candidates)!=1 or candidates[0] in used:
            ambiguous+=1;continue
        matches[i]=candidates[0];used.add(candidates[0])
    return matches,ambiguous


def run_verify(args):
    generated=Path(args.generated)
    frozen=Path(args.frozen)
    if not generated.is_file():
        raise ValueError('GENERATED_ADAPTER_MISSING')
    if not frozen.is_file() or sha_file(frozen)!=FROZEN_ADAPTER_SHA:
        raise ValueError('FROZEN_REFERENCE_SHA_MISMATCH')
    built=read_adapter(generated)
    refs=read_adapter(frozen)
    matches,ambiguous=match_rebuilt_to_frozen(built,refs)
    field_count=0;field_bad=0;exact=0;schema_equal=0;case_rows=[]
    for i,j in matches.items():
        a,b=built[i],refs[j]
        schema_equal+=int(a['schema_version']==b['schema_version'])
        bad=0
        for m,n in zip(a['metrics'],b['metrics']):
            for f in NUMERIC_FIELDS:
                field_count+=1
                if m[f] is None and n[f] is None:
                    exact+=1
                elif m[f] is not None and n[f] is not None and m[f]==n[f]:
                    exact+=1
                if not nearly_equal(m[f],n[f]):
                    field_bad+=1;bad+=1
        case_rows.append({'case_sha256':fp(str(i)),'metric_count':len(a['metrics']),
                          'positive_missing_rate_witness':'',
                          'compact_roundtrip':'','field_mismatches':bad})
    status=('PASS_PUBLIC_ORIGIN_19_FIELD_NUMERIC_PARITY_ONLY'
            if len(matches)==75 and ambiguous==0 and field_count==157681 and field_bad==0
            else 'HOLD_PUBLIC_ORIGIN_NUMERIC_MISMATCH_OR_MAPPING')
    summary={
      'version':VERSION,'final_status':status,
      'public_origin_build_reference_sha256':sha_file(generated),
      'frozen_reference_sha256':sha_file(frozen),
      'uniquely_matched_cases':len(matches),'ambiguous_cases':ambiguous,
      'numeric_fields_compared':field_count,'numeric_fields_mismatched':field_bad,
      'float_exact_or_both_null':exact,
      'schema_version_equal_cases':schema_equal,
      'full_jsonl_byte_parity':'NOT_TESTED_HISTORICAL_UID_DIFFERENT',
      'generation_input_frozen_adapter':False,'verification_reference_frozen_adapter':True,
      'token_budget_evaluation':'NOT_RERUN','independent_machine_cleanroom':'NOT_TESTED',
      'gpu_used':False,'raw_values_in_report_zip':False
    }
    report,report_sha=make_report(generated.parent,summary,case_rows,mode='verify')
    for key in ('final_status','uniquely_matched_cases','ambiguous_cases','numeric_fields_compared',
                'numeric_fields_mismatched','schema_version_equal_cases'):
        print(key.upper()+'='+str(summary[key]))
    print('OUTPUT_ZIP='+str(report))
    print('OUTPUT_SHA256='+report_sha)
    print('GPU_USED=0')
    return 0


def run_selftest():
    pd,_=import_pd()
    df=pd.DataFrame({'time':list(range(8)),
                     'svc_x':[0.,1.,None,3.,4.,5.,6.,7.],
                     'svc_y':[10.,10.,10.,10.,10.,10.,10.,10.]})
    original=WINDOW
    try:
        globals()['WINDOW']=3
        mi=compute_model_input(df,4,'v1')
        assert len(mi['metrics'])==2
        assert mi['metrics'][0]['metric_id']=='svc_x'
        assert nearly_equal(mi['metrics'][0]['normal_mean'],2.)
        assert nearly_equal(mi['metrics'][0]['faulty_mean'],5.)
        assert nearly_equal(mi['metrics'][0]['delta_mean'],3.)
        assert nearly_equal(mi['metrics'][0]['normal_std'],1.)
        assert nearly_equal(mi['metrics'][0]['normal_missing_rate'],1/3)
        assert stable_json(from_compact(to_compact(mi)))==stable_json(mi)
        a,b=read_adapter_for_test(mi)
        matches,amb=match_rebuilt_to_frozen([a],[b])
        assert matches=={0:0} and amb==0
        assert len(output_line('sample',mi))>50
        assert not nearly_equal(0.,1.) and nearly_equal(None,None)
        assert diff(None,4.) is None
    finally:
        globals()['WINDOW']=original
    # The real parity gate requires installed pandas + a Parquet reader on Windows.
    # We intentionally do not claim a real 75-case Parquet execution from this selftest.
    print('SELFTEST=PASS')
    print('COMPACT_ROUNDTRIP_SELFTEST=PASS')
    print('PANDAS_STAT_DDOF0_SELFTEST=PASS')
    print('ACTUAL_PARQUET_REAL_SOURCE=NOT_TESTED_IN_SELFTEST')
    print('GPU_USED=0')
    return 0


def read_adapter_for_test(mi):
    return mi,json.loads(stable_json(mi))


def main():
    ap=argparse.ArgumentParser(description='QA-SSEC CPU-only public-origin 247-row adapter compiler')
    ap.add_argument('--mode',required=True,choices=('selftest','build','verify'))
    ap.add_argument('--index',help='Pinned cases.parquet path')
    ap.add_argument('--source-root',help='Public-byte-identical 75-case telemetry source tree')
    ap.add_argument('--pins',default=str(Path(__file__).parent/'PINNED_PUBLIC_150_SOURCE_DIGESTS.csv'))
    ap.add_argument('--schema-version',default='v1',help='Public payload schema marker; version parity audited separately')
    ap.add_argument('--output-root',help='Output directory under project study branch')
    ap.add_argument('--generated',help='Generated private adapter JSONL for post-build audit')
    ap.add_argument('--frozen',help='Frozen adapter for comparison ONLY; not read by build')
    args=ap.parse_args()
    if args.mode=='selftest':return run_selftest()
    if args.mode=='build':
        if not all([args.index,args.source_root,args.output_root]):
            ap.error('build requires --index --source-root --output-root')
        return run_build(args)
    if args.mode=='verify':
        if not all([args.generated,args.frozen]):
            ap.error('verify requires --generated --frozen')
        return run_verify(args)
    return 3

if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('FINAL_STATUS=HOLD_BUILD_OR_VERIFICATION_ERROR',file=sys.stderr)
        print('REASON_CODE='+type(exc).__name__,file=sys.stderr)
        # Do not emit paths, metric IDs, raw values, native targets or exception detail.
        sys.exit(2)
