"""Offline, stored-output-only verifier for the selected public artifact.
Python 3.8+ stdlib; no network, pip/conda, original query, model, packing, or tokenizer.
"""
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'reproduced')
    args=ap.parse_args()
    if sys.version_info<(3,8):raise RuntimeError('PYTHON_3_8_REQUIRED')
    with (ROOT/'FILE_MANIFEST_SHA256.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        p=ROOT/row['path']
        if not p.is_file() or p.stat().st_size!=int(row['bytes']) or digest(p)!=row['sha256']:
            raise RuntimeError('INPUT_MISMATCH:'+row['path'])
    print('RELEASE_INPUT_FILES=PASS_'+str(len(rows)),flush=True)
    out=args.output.resolve();report=out/'VERIFY_REPORT.json'
    for row in rows:
        ip=(ROOT/row['path']).resolve()
        try:
            ip.relative_to(out)
        except ValueError:
            continue
        raise RuntimeError('OUTPUT_MUST_NOT_CONTAIN_RELEASE_INPUTS')
    if report.exists():
        obj=json.loads(report.read_text(encoding='utf-8'))
        if obj.get('status')!='PASS_STORED_OUTPUT_REPRODUCTION':raise RuntimeError('REVIEW_EXISTING_OUTPUT')
        for x in obj['generated_files']:
            p=out/x['path']
            if not p.is_file() or digest(p)!=x['sha256']:
                raise RuntimeError('EXISTING_RESULT_MODIFIED')
        print('REPRODUCTION=ALREADY_VERIFIED_NOT_RECOMPUTED');return
    if out.exists() and any(out.iterdir()):raise RuntimeError('OUTPUT_NOT_EMPTY')
    out.mkdir(parents=True,exist_ok=True)
    log=out/'logs';log.mkdir()
    jobs=[
      ('proof_verify',[ROOT/'certificate/CERTIFY.py','--input',ROOT/'certificate/INPUT_EVIDENCE.zip','--verify',ROOT/'certificate/CERTIFICATES.json'],[]),
      ('certificates',[ROOT/'certificate/CERTIFY.py','--input',ROOT/'certificate/INPUT_EVIDENCE.zip','--output',out/'certificate'],[('certificate/CERTIFICATES.json','certificate/CERTIFICATES.json'),('certificate/SUMMARY.csv','certificate/SUMMARY.csv')]),
      ('mechanism',[ROOT/'mechanism/RECOMPUTE.py','--evidence',ROOT/'mechanism/INPUT_RESULTS.zip','--out',out/'mechanism'],[('mechanism/RESULTS.json','mechanism/RESULTS.json'),('mechanism/ANALYSIS_TABLES.csv','mechanism/ANALYSIS_TABLES.csv')]),
      ('objectives',[ROOT/'objectives/REANALYZE.py','--input',ROOT/'objectives/INPUT_EVIDENCE.zip','--output',out/'objectives'],[('objectives/RESULTS.json','objectives/RESULTS.json'),('objectives/SEED_COUNTS.csv','objectives/SEED_COUNTS.csv'),('objectives/THRESHOLD_PROFILE.csv','objectives/THRESHOLD_PROFILE.csv')]),
      ('composition',[ROOT/'composition/REANALYZE.py','--input',ROOT/'composition/INPUT_EVIDENCE.zip','--output',out/'composition'],[('composition/RESULTS.json','composition/RESULTS.json'),('composition/GROUP_PROFILE.csv','composition/GROUP_PROFILE.csv'),('composition/SUBSET_RESULTS.csv','composition/SUBSET_RESULTS.csv')]),
      ('count_floor',[ROOT/'count_floor/RECOMPUTE_COUNT_COMPONENTS.py','--source-dir',ROOT/'inputs','--out',out/'count_floor'],[('count_floor/COUNT_ERROR_COMPONENTS.csv','count_floor/COUNT_ERROR_COMPONENTS.csv'),('count_floor/CALCULATION_RECORD.json','count_floor/CALCULATION_RECORD.json')]),
      ('fixed_readout',[ROOT/'fixed_readout/05_REANALYZE_STORED_OUTPUTS.py','--source',ROOT/'fixed_readout/07_SOURCE_RESULTS.zip','--out',out/'fixed_readout'],[('fixed_readout/04_ANALYSIS_SUMMARY.json','fixed_readout/04_ANALYSIS_SUMMARY.json'),('fixed_readout/03_COMPARISON_TABLES.csv','fixed_readout/03_COMPARISON_TABLES.csv')]),
    ]
    compared=[];done=[]
    for name,command,tests in jobs:
        proc=subprocess.run([sys.executable,'-I','-B']+[str(v) for v in command],cwd=str(ROOT),capture_output=True,text=True,errors='replace',timeout=180)
        (log/(name+'.stdout.txt')).write_text(proc.stdout,encoding='utf-8')
        (log/(name+'.stderr.txt')).write_text(proc.stderr,encoding='utf-8')
        if proc.returncode:
            raise RuntimeError('ANALYSIS_JOB_FAILED:'+name+';SEE_LOGS')
        for reference,generation in tests:
            if (ROOT/reference).read_bytes()!=(out/generation).read_bytes():
                raise RuntimeError('BYTE_PARITY_FAILED:'+generation)
            compared.append({'reference':reference,'generated':generation,'sha256':digest(out/generation)})
        done.append(name);print('ANALYSIS_JOB='+name+';PASS',flush=True)
    generated=[{'path':p.relative_to(out).as_posix(),'sha256':digest(p)} for p in sorted(out.rglob('*')) if p.is_file()]
    result={'status':'PASS_STORED_OUTPUT_REPRODUCTION','scope':'STORED_OUTPUTS_ONLY_NOT_RAW_EAB_COMPLETE',
        'python_version':sys.version.split()[0], 'candidate_files':len(rows),'jobs':done,
        'directed_certificates':32,'byte_identical_outputs':compared,'generated_files':generated,
        'verification_scope':'NUMERICAL_REPRODUCTION_NOT_RIGHTS_CERTIFICATION','network_or_install_or_original_training_calls':0}
    report.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('REPRODUCTION=PASS_STORED_OUTPUTS_ONLY')
    print('DIRECTED_CERTIFICATES=32;BYTE_IDENTICAL_OUTPUTS='+str(len(compared)))
    print('NO_NEW_QUERIES_TRAINING_PACKING_INSTALL_NETWORK=1')
if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('REPRODUCTION=HOLD_'+str(exc),file=sys.stderr)
        sys.exit(1)
