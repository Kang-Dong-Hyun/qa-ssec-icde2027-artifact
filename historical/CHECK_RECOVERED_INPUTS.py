"""Check original public-benchmark-derived inputs and archived predictions.
Python 3.8+ stdlib. Does not load a model, tokenize, pack, train, run queries,
or connect to a network. Files in FROZEN_INPUTS.zip are read in memory.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

def require(condition, message):
    if not condition:
        raise ValueError(message)

def csv_bytes(blob):
    return list(csv.DictReader(io.StringIO(blob.decode('utf-8-sig'))))

def close(actual, expected):
    return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12)

def check():
    manifest = csv_bytes((HERE / 'INPUT_MANIFEST.csv').read_bytes())
    with zipfile.ZipFile(HERE / 'FROZEN_INPUTS.zip') as z:
        require(z.testzip() is None, 'INPUT_ZIP_CRC')
        require(len(z.namelist()) == len(set(z.namelist())), 'DUPLICATE_ARCHIVE_MEMBER')
        require(set(z.namelist()) == {r['path'] for r in manifest}, 'INPUT_MEMBER_SET')
        payload = {n: z.read(n) for n in z.namelist()}
    for r in manifest:
        b = payload[r['path']]
        require(len(b) == int(r['bytes']), 'INPUT_SIZE:' + r['path'])
        require(hashlib.sha256(b).hexdigest() == r['sha256'], 'INPUT_HASH:' + r['path'])
    targets_rows = csv_bytes(payload['development/targets.csv'])
    targets = {r['record_uid']: r for r in targets_rows}
    require(len(targets_rows) == len(targets) == 75, 'TARGET_SET')
    for uid, r in targets.items():
        require(hashlib.sha256(r['case'].encode('utf-8')).hexdigest() == uid, 'UID_RULE')
    adapter_rows = [json.loads(s) for s in payload['development/adapter.jsonl'].decode('utf-8').splitlines() if s.strip()]
    adapter = {r['record_uid']: r['model_input'] for r in adapter_rows}
    require(len(adapter_rows) == len(adapter) == 75 and set(adapter) == set(targets), 'ADAPTER_UIDS')
    require(sum(len(r['metrics']) for r in adapter.values()) == 8299, 'METRIC_COUNT')
    top_map = {'schema_version': 'v', 'window_timesteps': 'w', 'metrics': 'm'}
    metric_map = {
        'metric_id':'i','normal_missing_rate':'nmr','faulty_missing_rate':'fmr',
        'normal_mean':'nm','faulty_mean':'fm','delta_mean':'dm','normal_std':'ns',
        'faulty_std':'fs','normal_median':'nmed','faulty_median':'fmed','delta_median':'dmed',
        'normal_q25':'n25','normal_q75':'n75','faulty_q25':'f25','faulty_q75':'f75',
        'normal_min':'nmin','normal_max':'nmax','faulty_min':'fmin','faulty_max':'fmax',
        'delta_missing_rate':'dmr'}
    variants = ['B01_EXT_EXACT','B02_EXT_COMPACT','B03_EXT_HMAC','B04_EXT_ROLE']
    representations = {}
    for variant in variants:
        rs = [json.loads(s) for s in payload['representations/' + variant + '_LOCAL_V1.jsonl'].decode('utf-8').splitlines() if s.strip()]
        rs_map = {r['record_uid']: json.loads(r['representation']) for r in rs}
        require(len(rs) == len(rs_map) == 75 and set(rs_map) == set(targets), 'REPRESENTATION_UIDS:' + variant)
        representations[variant] = rs_map
    require(representations['B01_EXT_EXACT'] == adapter, 'B01_ADAPTER_IDENTITY')
    for uid, original in adapter.items():
        compact = {}
        for k, value in original.items():
            compact[top_map[k]] = [{metric_map[mk]: mv for mk,mv in m.items()} for m in value] if k == 'metrics' else value
        require(compact == representations['B02_EXT_COMPACT'][uid], 'B02_VALUE_IDENTITY:' + uid)
    fold_rows = csv_bytes(payload['development/folds.csv'])
    require(len(fold_rows) == 375, 'FOLD_ROW_COUNT')
    folds = {}
    for f in range(5):
        rs = [r for r in fold_rows if int(r['fold']) == f]
        require(Counter(r['role'] for r in rs) == {'train':45,'validation':15,'test':15}, 'FOLD_ROLE_COUNTS')
        require(len(rs) == len({r['record_uid'] for r in rs}) == 75, 'FOLD_UID_COUNT')
        folds[f] = {r['record_uid']: r['role'] for r in rs}
        require(set(folds[f]) == set(targets), 'FOLD_UID_SET')
        for r in rs:
            for field in ('case','dataset','system','root_cause_service','fault','repetition'):
                require(r[field] == targets[r['record_uid']][field], 'FOLD_TARGET_JOIN')
    require(all(sum(folds[f][u]=='test' for f in range(5)) == 1 for u in targets), 'OUT_OF_FOLD_TEST_UNIQUENESS')
    mapping = csv_bytes(payload['development/CASE_MAP.csv'])
    key_map = {u: hashlib.sha256(u.encode('utf-8')).hexdigest()[:16] for u in targets}
    require(len(mapping) == 75 and {r['record_uid'] for r in mapping} == set(targets), 'MAP_UID_SET')
    for r in mapping:
        require(r['case'] == targets[r['record_uid']]['case'] and r['case_key'] == key_map[r['record_uid']], 'CASE_MAP')
    with zipfile.ZipFile(ROOT/'inputs/QUERY_REPLAY_RESULT.zip') as z:
        query_rows = csv_bytes(z.read('QUERY_REPLAY_CASE_RESULTS.csv'))
    require({r['case_key'] for r in query_rows} == set(key_map.values()), 'QUERY_CASE_KEY_JOIN')
    predictions = {}
    fold_reference = {(int(r['cap']),int(r['fold']),r['variant']):r for r in csv_bytes((HERE/'TRACKB_PER_FOLD.csv').read_bytes())}
    files = sorted(n for n in payload if n.startswith('predictions/'))
    require(len(files) == 40, 'PREDICTION_FILE_COUNT')
    values_checked = 0
    for name in files:
        matched = re.fullmatch(r'predictions/k([1248])_f([0-4])_(B0[12])\.csv',name)
        require(matched is not None, 'PREDICTION_FILENAME')
        cap,f,arm = int(matched[1]),int(matched[2]),matched[3]
        variant = arm + '_EXT_' + ('EXACT' if arm=='B01' else 'COMPACT')
        rs = csv_bytes(payload[name])
        require(len(rs) == 15 and len({r['record_uid'] for r in rs}) == 15, 'PREDICTION_ROWS')
        require({r['record_uid'] for r in rs} == {u for u,role in folds[f].items() if role=='test'}, 'PREDICTION_TEST_SET')
        for r in rs:
            require(r['split']=='test' and int(r['cap'])==cap and int(r['fold'])==f and r['variant']==variant, 'PREDICTION_KEYS')
            uid = r['record_uid']; target = targets[uid]
            service = int(r['gold_root_cause_service'] == r['pred_root_cause_service'])
            fault = int(r['gold_fault'] == r['pred_fault'])
            require(int(r['service_correct'])==service and int(r['fault_correct'])==fault and int(r['pair_exact'])==service*fault, 'CORRECTNESS_FLAGS')
            require(r['gold_root_cause_service']==target['root_cause_service'] and r['gold_fault']==target['fault'], 'GOLD_TARGETS')
            n = len(adapter[uid]['metrics']); k = int(r['covered_metrics'])
            require(int(r['metric_count_total'])==n and 0<=k<=n and close(r['metric_coverage'], k/n), 'STORED_COVERAGE')
            predictions[cap,arm,uid] = r
        for obs,ref in [('metric_coverage','test_metric_coverage'),('service_correct','test_service_accuracy'),('fault_correct','test_fault_accuracy'),('pair_exact','test_pair_exact_accuracy')]:
            require(close(statistics.mean(float(r[obs]) for r in rs), fold_reference[cap,f,variant][ref]), 'FOLD_TEST_METRIC')
            values_checked += 1
    require(len(predictions) == 600, 'UNIQUE_PREDICTION_COUNT')
    paired=[]
    for r in csv_bytes((HERE/'TRACKB_PAIRED_COUNTS.csv').read_bytes()):
        cap=int(r['cap']); ep=r['endpoint']
        bs=[(int(predictions[cap,'B01',u][ep]),int(predictions[cap,'B02',u][ep])) for u in targets]
        a=sum(x==1 and y==1 for x,y in bs); h=sum(x==1 and y==0 for x,y in bs)
        b=sum(x==0 and y==1 for x,y in bs); d=sum(x==0 and y==0 for x,y in bs)
        for col,value in [('both_correct',a),('b01_only_correct',h),('b02_only_correct',b),('both_wrong',d),('b01_correct',a+h),('b02_correct',a+b)]:
            require(int(r[col])==value, 'PAIRED_COUNT:'+col)
        n=h+b
        p=min(1.,2*sum(math.comb(n,k) for k in range(min(h,b)+1))/2**n) if n else 1.
        require(close(p,r['exact_mcnemar_p_two_sided_exploratory']), 'PAIRED_PVALUE')
        paired.append({'cap':cap,'endpoint':ep,'b01_correct':a+h,'b02_correct':a+b,'both_correct':a,'b01_only':h,'b02_only':b,'both_wrong':d,'exact_p':p})
    table=[]
    for cap in (1,2,4,8):
        p=next(r for r in paired if r['cap']==cap and r['endpoint']=='service_correct')
        table.append({'cap':cap,
            'B01_coverage':statistics.mean(float(predictions[cap,'B01',u]['metric_coverage']) for u in targets),
            'B02_coverage':statistics.mean(float(predictions[cap,'B02',u]['metric_coverage']) for u in targets),
            'B01_correct':p['b01_correct'],'B02_correct':p['b02_correct'],
            'B01_only':p['b01_only'],'B02_only':p['b02_only']})
    return {'status':'PASS_RECOVERED_INPUTS_AND_ORIGINAL_PREDICTIONS',
        'input_archive_members':55,'original_payload_files':54,'derived_case_map_files':1,
        'adapter_cases':75,'adapter_metrics':8299,'fold_rows':375,
        'development_case_keys_linked':75,'B01_B02_value_identity_cases':75,
        'prediction_files':40,'prediction_rows':600,'fold_test_metrics_checked':values_checked,
        'paired_groups_checked':12,'table_x':table,
        'new_training_inference_queries_tokenization_or_network':False},table

def write_results(output, result, table):
    output = output.resolve()
    for protected in (HERE,ROOT/'inputs',ROOT/'certificate'):
        require(output != protected and output not in protected.parents, 'OUTPUT_CONTAINS_INPUTS')
    output.mkdir(parents=True,exist_ok=True)
    texts={'D4_CHECK.json':json.dumps(result,sort_keys=True,indent=2)+'\n'}
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    texts['TABLE_X_FROM_PREDICTIONS.csv']=stream.getvalue()
    for name,text in texts.items():
        path=output/name; blob=text.encode('utf-8')
        if path.exists():require(path.read_bytes()==blob,'EXISTING_OUTPUT_CONFLICT:'+name)
        else:path.write_bytes(blob)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reproduced_d4')
    args=parser.parse_args()
    result,table=check();write_results(args.output,result,table)
    print('RECOVERED_INPUTS=PASS_55_MEMBERS')
    print('ORIGINAL_PREDICTIONS=PASS_40_FILES_600_ROWS')
    print('FOLD_TEST_METRICS=160;PAIRED_GROUPS=12;TABLE_X_ROWS=4')
    print('CASE_KEY_LINKS=75;TRAINING_INFERENCE_QUERY_PACKING_NETWORK=0')

if __name__=='__main__':
    main()
