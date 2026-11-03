"""Extracted frozen computation definitions. No source discovery, file writes, network or automatic experiment.
See SOURCE_FUNCTION_PROVENANCE.json.
"""
import math
import statistics

FAMILIES = (
    'COUNT_POSITIVE_MEAN_SHIFT',
    'FRACTION_POSITIVE_MEAN_SHIFT',
    'COUNT_MEAN_MEDIAN_SIGN_DISAGREEMENT',
    'COUNT_MISSING_RATE_INCREASE',
    'MEAN_MISSING_RATE_CHANGE',
    'COUNT_STD_INCREASE',
    'EXISTS_MISSING_RATE_INCREASE',
)


NUMERIC_FAMILIES = frozenset((
    'COUNT_POSITIVE_MEAN_SHIFT','FRACTION_POSITIVE_MEAN_SHIFT',
    'COUNT_MEAN_MEDIAN_SIGN_DISAGREEMENT',
    'COUNT_MISSING_RATE_INCREASE','MEAN_MISSING_RATE_CHANGE',
    'COUNT_STD_INCREASE'))


FAMILY_FIELD = {
 'COUNT_POSITIVE_MEAN_SHIFT': 'delta_mean > 0 per metric',
 'FRACTION_POSITIVE_MEAN_SHIFT': 'proportion(delta_mean > 0) among finite values',
 'COUNT_MEAN_MEDIAN_SIGN_DISAGREEMENT': 'sign(delta_mean) != sign(delta_median) per metric',
 'COUNT_MISSING_RATE_INCREASE': 'faulty_missing_rate > normal_missing_rate per metric',
 'MEAN_MISSING_RATE_CHANGE': 'mean(faulty_missing_rate - normal_missing_rate); unitless rates',
 'COUNT_STD_INCREASE': 'faulty_std > normal_std per metric; same metric units',
 'EXISTS_MISSING_RATE_INCREASE': 'exists metric with faulty_missing_rate > normal_missing_rate',
}


def finite(v):
    if isinstance(v,bool) or not isinstance(v,(int,float)):
        return None
    x=float(v)
    return x if math.isfinite(x) else None


def direction(v):
    return (v > 0.0) - (v < 0.0)


def source_series(metrics):
    # Unit-safe per-metric predicates, or dimensionsless missing-rate aggregate.
    # Never aggregate raw delta_mean magnitudes across heterogeneous KPI units.
    series={family:{} for family in FAMILIES}
    for m in metrics:
        mid=m['metric_id']
        d=finite(m.get('delta_mean'))
        dmed=finite(m.get('delta_median'))
        if d is not None:
            series['COUNT_POSITIVE_MEAN_SHIFT'][mid]=int(d>0)
            series['FRACTION_POSITIVE_MEAN_SHIFT'][mid]=int(d>0)
        if d is not None and dmed is not None:
            series['COUNT_MEAN_MEDIAN_SIGN_DISAGREEMENT'][mid]=int(direction(d)!=direction(dmed))
        norm=finite(m.get('normal_missing_rate'))
        fault=finite(m.get('faulty_missing_rate'))
        if norm is not None and fault is not None:
            delta=fault-norm
            series['COUNT_MISSING_RATE_INCREASE'][mid]=int(delta>0)
            series['MEAN_MISSING_RATE_CHANGE'][mid]=delta
            series['EXISTS_MISSING_RATE_INCREASE'][mid]=int(delta>0)
        ns=finite(m.get('normal_std'))
        fs=finite(m.get('faulty_std'))
        if ns is not None and fs is not None:
            series['COUNT_STD_INCREASE'][mid]=int(fs>ns)
    if any(not v for v in series.values()):
        raise RuntimeError('NO_ELIGIBLE_METRICS_FOR_QUERY_FAMILY')
    return series


def query_eval(family, full_series, selected_ids):
    # Gold scope defined by the ORIGINAL case, known only to offline oracle.
    # A partial input alone cannot generally prove it contains every metric.
    subset={m:val for m,val in full_series.items() if m in selected_ids}
    total=len(full_series)
    known=len(subset)
    complete=int(known==total)
    visible_witness_certified_true=0
    if family=='FRACTION_POSITIVE_MEAN_SHIFT':
        gold=sum(full_series.values()) / total
        answer=sum(subset.values()) / known if known else None
        scale=1.0   # dimensionless proportion
    elif family=='MEAN_MISSING_RATE_CHANGE':
        gold=statistics.mean(full_series.values())
        answer=statistics.mean(subset.values()) if subset else None
        scale=1.0   # dimensionless missing-rate percentage points
    elif family in ('COUNT_POSITIVE_MEAN_SHIFT',
                    'COUNT_MEAN_MEDIAN_SIGN_DISAGREEMENT',
                    'COUNT_MISSING_RATE_INCREASE','COUNT_STD_INCREASE'):
        gold=sum(full_series.values())
        answer=sum(subset.values())
        scale=max(gold,1)
    elif family=='EXISTS_MISSING_RATE_INCREASE':
        gold=any(full_series.values())
        answer=any(subset.values())
        visible_witness_certified_true=int(answer)  # positive witness proves True
        scale=None
    else:raise RuntimeError('UNKNOWN_QUERY_FAMILY')
    if family=='EXISTS_MISSING_RATE_INCREASE':
        agreement=int(gold==answer)
    else:
        agreement=int(math.isclose(float(gold),float(answer),rel_tol=1e-12,abs_tol=1e-12)) \
            if answer is not None else 0
    if complete and agreement!=1:
        raise RuntimeError('QUERY_COMPLETE_BUT_RESULT_NOT_EQUAL')
    if visible_witness_certified_true and not agreement:
        raise RuntimeError('VISIBLE_WITNESS_PROOF_UNSOUND')
    returned=int(answer is not None)
    error=(abs(float(gold)-float(answer)) if returned and family in NUMERIC_FAMILIES else None)
    error_scaled=(error/scale if error is not None else None)
    if error is not None and not math.isfinite(error_scaled):
        raise RuntimeError('NONFINITE_QUERY_ERROR')
    return {'source_valid_metrics':total,'selected_valid_metrics':known,
      'returned':returned,'agreement':agreement,
      'oracle_scope_complete':complete,
      'correct_but_incomplete':int(bool(agreement and not complete)),
      'visible_witness_certified_true':visible_witness_certified_true,
      'gold_neutral':int(gold==0 or gold is False),
      'abs_error':error,'scaled_abs_error':error_scaled,
      'top3_recall':None}
