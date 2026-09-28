#!/usr/bin/env python3
"""Real transformed-image imports after installation, before model allocation."""
import sys
sys.dont_write_bytecode = True
import _diagnostics as diagnostics
import hashlib,json,os,subprocess
from pathlib import Path

PROBE=r'''
import hashlib,json,pathlib
import vllm.model_executor.layers.quantization.exl3
import vllm.distributed.device_communicators.cuda_communicator
from vllm.distributed.device_communicators import jspark3_triar_runtime,jspark3_triar_dual,jspark3_triar_kernel
import vllm.v1.worker.gpu.cudagraph_utils as graphs
import b5_prefix_verify
assert jspark3_triar_dual._INSTALLED,'dual-bank graph hook was not installed'
assert hasattr(graphs,'ModelCudaGraphManager'),'wrong graph phase'
ok,report=b5_prefix_verify.hash_gate()
assert ok,report
paths=[jspark3_triar_runtime.__file__,jspark3_triar_dual.__file__,jspark3_triar_kernel.__file__,graphs.__file__,b5_prefix_verify.__file__]
print('FINAL_INSTALLED '+json.dumps(dict(status='PASS',files={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths},b5_hash_gate=report)))
'''

def main():
 rank=int(os.environ['NODE_RANK']);assert rank in (0,1,2)
 out=Path('/evidence/final-installed-preflight');out.mkdir(exist_ok=False)
 receipt=dict(status='FAULT',rank=rank,phase='after-source-install-before-model-load')
 try:
  p=subprocess.run(['python3','-B','-c',PROBE],env=dict(os.environ,B5_OUT=str(out)),stdin=subprocess.DEVNULL,text=True,capture_output=True,timeout=180)
  diagnostics.retain(p.stdout + p.stderr, out/'result.json')
  rows=[json.loads(x[16:]) for x in p.stdout.splitlines() if x.startswith('FINAL_INSTALLED ')]
  assert p.returncode==0 and len(rows)==1 and rows[0]['status']=='PASS','transformed-image import/graph-source smoke failed'
  receipt.update(status='PASS',installed_sha256=diagnostics.fingerprint(rows[0]),graph_capture_executed=False)
 finally:
  # Runtime module evidence stays local; shared receipt binds its bytes only.
  receipt['runtime_evidence_sha256']=diagnostics.fingerprint({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and not p.name.endswith(diagnostics.PRIVATE_SUFFIX)})
  (out/'result.json').write_text(json.dumps(receipt,sort_keys=True)+'\n')
 print('FINAL-INSTALLED-IMPORTS-PASS '+json.dumps(dict(rank=rank,receipt_sha256=hashlib.sha256((out/'result.json').read_bytes()).hexdigest()),sort_keys=True),flush=True)
 return 0

if __name__=='__main__':
 diagnostics.install_exception_hook()
 raise SystemExit(main())
