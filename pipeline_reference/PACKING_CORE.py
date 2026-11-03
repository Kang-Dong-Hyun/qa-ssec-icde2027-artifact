"""Extracted frozen computation definitions. No source discovery, file writes, network or automatic experiment.
See SOURCE_FUNCTION_PROVENANCE.json.
"""
import hashlib
import json
import math

BUDGET=8064


CAP=4


METRIC_MAP={
    'metric_id':'i', 'normal_missing_rate':'nmr','faulty_missing_rate':'fmr',
    'normal_mean':'nm','faulty_mean':'fm','delta_mean':'dm',
    'normal_std':'ns','faulty_std':'fs','normal_median':'nmed',
    'faulty_median':'fmed','delta_median':'dmed',
    'normal_q25':'n25','normal_q75':'n75','faulty_q25':'f25',
    'faulty_q75':'f75','normal_min':'nmin','normal_max':'nmax',
    'faulty_min':'fmin','faulty_max':'fmax','delta_missing_rate':'dmr'
}


def stable_json(obj):
    return json.dumps(obj,sort_keys=True,ensure_ascii=False,
                      separators=(',',':'),allow_nan=False)


def numeric(x):
    if isinstance(x,bool) or not isinstance(x,(int,float)):return None
    val=float(x)
    return val if math.isfinite(val) else None


def case_key(uid):return hashlib.sha256(str(uid).encode('utf-8')).hexdigest()[:16]


def seed_hash(metric_id,seed):
    payload=('QA_SSEC_ICDE_CPU_V4|20261028|SEED_%02d|%s' %
             (seed,str(metric_id))).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def compact_metric(m):
    unknown=set(m)-set(METRIC_MAP)
    if unknown:raise RuntimeError('HOLD_UNEXPECTED_METRIC_FIELDS')
    return {METRIC_MAP[k]:v for k,v in m.items()}


def serialize_slice(mi,ordered,start,end,variant):
    if variant=='B01':
        obj={'schema_version':mi['schema_version'],'window_timesteps':mi['window_timesteps'],
             'metrics':ordered[start:end]}
    else:
        obj={'v':mi['schema_version'],'w':mi['window_timesteps'],
             'm':[compact_metric(m) for m in ordered[start:end]]}
    return stable_json(obj)


def pack_exact(tok,case,ordered,variant,budget=BUDGET,cap=CAP):
    """V1's linear first-overflow contiguous greedy packing, stopping at K=4."""
    n=len(ordered)
    chunks=[];start=0;calls=0
    while start<n and len(chunks)<cap:
        end=start+1;best=None;best_tokens=0
        while end<=n:
            tokens=tok.encode(serialize_slice(case['mi'],ordered,start,end,variant),
                              add_special_tokens=False)
            size=len(tokens.ids) if hasattr(tokens,'ids') else len(tokens)
            calls+=1
            if size<=budget:
                best=end;best_tokens=size;end+=1
            else:break
        if best is None:raise RuntimeError('HOLD_SINGLE_METRIC_BUDGET_OVERFLOW')
        chunks.append((start,best,best_tokens))
        start=best
    count=start
    used=sum(chunk[2] for chunk in chunks)
    if count<1 or count>n or used>budget*cap or len(chunks)>cap:
        raise RuntimeError('HOLD_TOKEN_BUDGET_OR_CHUNKS')
    return count,used,len(chunks),calls
