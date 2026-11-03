"""Verify stored fold/aggregate/paired outputs; does not train or use a model."""
from pathlib import Path
import csv
import math
import statistics

ROOT = Path(__file__).resolve().parent

def rows(name):
    with (ROOT/name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def check():
    folds=rows('TRACKB_PER_FOLD.csv')
    aggregates=rows('TRACKB_ALL_ARMS.csv')
    paired=rows('TRACKB_PAIRED_COUNTS.csv')
    subset=rows('TRACKB_EXACT_COMPACT.csv')
    if (len(folds),len(aggregates),len(paired),len(subset)) != (80,16,12,8):
        raise ValueError('HISTORICAL_SHAPE_CHANGED')
    fields=('test_metric_coverage','test_service_accuracy',
            'test_fault_accuracy','test_pair_exact_accuracy')
    checked=0
    for a in aggregates:
        fs=[r for r in folds if (r['cap'],r['variant'])==(a['cap'],a['variant'])]
        if len(fs)!=5 or {r['fold'] for r in fs}!={str(i) for i in range(5)}:
            raise ValueError('FOLD_KEYS_CHANGED')
        for field in fields:
            values=[float(r[field]) for r in fs]
            for suffix,value in [('mean',statistics.mean(values)),('std',statistics.stdev(values))]:
                if not math.isclose(value,float(a[field+'_'+suffix]),rel_tol=1e-10,abs_tol=1e-12):
                    raise ValueError('HISTORICAL_AGGREGATE_MISMATCH')
                checked+=1
    for r in subset:
        a=next(x for x in aggregates if (x['cap'],x['variant'])==(r['cap'],r['variant']))
        for field in fields:
            if not math.isclose(float(r[field+'_mean']),float(a[field+'_mean']),abs_tol=1e-12):
                raise ValueError('SUBSET_AGGREGATE_MISMATCH')
    for r in paired:
        a,h,b,d=[int(r[k]) for k in ('both_correct','b01_only_correct','b02_only_correct','both_wrong')]
        if a+h+b+d!=int(r['cases']) or a+h!=int(r['b01_correct']) or a+b!=int(r['b02_correct']):
            raise ValueError('PAIRED_COUNTS_INCONSISTENT')
        p=min(1.,2*sum(math.comb(h+b,k) for k in range(min(h,b)+1))/(2**(h+b))) if h+b else 1.
        if not math.isclose(p,float(r['exact_mcnemar_p_two_sided_exploratory']),abs_tol=1e-14):
            raise ValueError('PAIRED_PVALUE_MISMATCH')
    expected={1:(15,18,2,5),2:(15,13,5,3),4:(15,15,4,4),8:(15,15,4,4)}
    for cap,counts in expected.items():
        r=next(x for x in paired if int(x['cap'])==cap and x['endpoint']=='service_correct')
        actual=tuple(int(r[k]) for k in ('b01_correct','b02_correct','b01_only_correct','b02_only_correct'))
        if actual!=counts:raise ValueError('TABLE_X_COUNTS_DIFFER')
        for variant,index in [('B01_EXT_EXACT',0),('B02_EXT_COMPACT',1)]:
            fs=[x for x in folds if int(x['cap'])==cap and x['variant']==variant]
            total=sum(round(15*float(x['test_service_accuracy'])) for x in fs)
            if total!=counts[index]:raise ValueError('TABLE_X_FOLD_COUNT_MISMATCH')
    return {'status':'PASS_HISTORICAL_STORED_OUTPUTS','fold_rows':80,
            'aggregate_rows':16,'mean_sd_values_checked':checked,'paired_groups_checked':12,
            'table_x_rows_checked':4,'training_or_inference_run':False}

if __name__=='__main__':
    print('HISTORICAL_STORED_OUTPUT_CHECK=PASS') if check()['status'].startswith('PASS') else None
    print('MEAN_SD_VALUES=128;PAIRED_GROUPS=12;TABLE_X_ROWS=4')
    print('NEW_TRAINING_INFERENCE_OR_QUERY_RUNS=0')
