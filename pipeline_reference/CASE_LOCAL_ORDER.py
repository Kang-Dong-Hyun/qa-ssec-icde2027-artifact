"""Extracted frozen computation definitions. No source discovery, file writes, network or automatic experiment.
See SOURCE_FUNCTION_PROVENANCE.json.
"""
import hashlib

def local_rank(case_key,metric_id,seed):
    # 선택 함수는 불변 case key와 metric ID만 사용합니다. gold/측정값/질의 정답은 인수가 아닙니다.
    text='QA_SSEC_ICDE_CPU_V4|20261028|SEED_%02d|CASE_%s|%s' % (seed,case_key,str(metric_id))
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
