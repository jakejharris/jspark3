"""Pre-captured target FULL-graph OFF/ON banks. No recapture on an epoch switch.

Both banks share vLLM's ordered graph pool and output buffers. Only target FULL
capture uses TRIAR: eager, piecewise prefill and the DFlash graphs retain NCCL.
The original profile_memory captures and discards both banks, so both are charged.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import weakref

_CAPTURE_ON = False
_EPOCH_PATH = Path('/evidence/triar-epoch.json')
_SIGNATURE = None
_EPOCH = {'schema':'triar-epoch/1','epoch':'boot-off','mode':'off'}
_SEEN = False
_REPLAYS = {'off':0,'on':0}
_INSTALLED = False
_TRIANGLES = weakref.WeakSet()

def register(triangle):
 if triangle is not None:_TRIANGLES.add(triangle)

def fast_calls():return sum(t.fast_calls for t in _TRIANGLES)


def capturing_on():return _CAPTURE_ON


@contextmanager
def capture_mode(on):
 global _CAPTURE_ON
 previous=_CAPTURE_ON
 _CAPTURE_ON=bool(on)
 try:yield
 finally:_CAPTURE_ON=previous


def validate_epoch(value):
 if (set(value)!={'schema','epoch','mode'} or value['schema']!='triar-epoch/1'
     or value['mode'] not in ('off','on') or not isinstance(value['epoch'],str)
     or not 1<=len(value['epoch'])<=160):
  raise ValueError('invalid TRIAR runtime epoch')
 return value


def poll():
 """Owner writes identical files on all idle ranks before allowing any request."""
 global _SIGNATURE,_EPOCH,_SEEN,_REPLAYS
 try:
  info=_EPOCH_PATH.lstat()
 except FileNotFoundError:
  if _SIGNATURE is not None:raise RuntimeError('TRIAR epoch disappeared')
  signature=None;value=_EPOCH
 else:
  if not stat.S_ISREG(info.st_mode) or info.st_size>4096:raise RuntimeError('unsafe TRIAR epoch file')
  signature=(info.st_ino,info.st_mtime_ns,info.st_size)
  if _SEEN and signature==_SIGNATURE:return _EPOCH['mode']
  value=validate_epoch(json.loads(_EPOCH_PATH.read_text()))
 if _SEEN and signature==_SIGNATURE:return _EPOCH['mode']
 import torch
 import torch.distributed as dist
 from vllm.distributed.parallel_state import get_tp_group
 group=get_tp_group()
 if group.world_size!=3:raise RuntimeError('TRIAR dual requires TP3')
 torch.cuda.synchronize()
 rows=[None]*3
 dist.all_gather_object(rows,value,group=group.cpu_group)
 if any(row!=value for row in rows):raise RuntimeError('TRIAR epoch disagrees across ranks')
 _SIGNATURE,_EPOCH,_SEEN=signature,value,True
 _REPLAYS={'off':0,'on':0}
 print(json.dumps(dict(triar='dual-epoch',rank=group.rank_in_group,**value),sort_keys=True),flush=True)
 return value['mode']


class DualGraph:
 def __init__(self,off,on):self.off,self.on=off,on
 def replay(self):
  mode=poll()
  (self.on if mode=='on' else self.off).replay()
  _REPLAYS[mode]+=1
  # First execution is a native proof that the requested bank really replayed.
  if _REPLAYS[mode]==1:
   from vllm.distributed.parallel_state import get_tp_group
   print(json.dumps(dict(triar='dual-replay',rank=get_tp_group().rank_in_group,
       epoch=_EPOCH['epoch'],mode=mode,count=1),sort_keys=True),flush=True)


def install_manager(module):
 if hasattr(module,'__file__') and hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()!='6b44f24e65e51a0a43c7a5d7d93ef5def8b880cff9cb671ba1e1302a79c757aa':
  raise RuntimeError('TRIAR dual GPU graph source drift')
 cls=module.CudaGraphManager
 if getattr(cls,'_triar_dual_installed',False):return
 original=cls.capture
 def capture(self,create_forward_fn,progress_bar_desc='Capturing CUDA graphs'):
  if not isinstance(self,module.ModelCudaGraphManager):
   with capture_mode(False):return original(self,create_forward_fn,progress_bar_desc)
  if self.graphs:raise RuntimeError('TRIAR dual refuses repeated capture over live graphs')
  descriptors=self._capture_descs
  full=module.CUDAGraphMode.FULL
  if not descriptors.get(full):raise RuntimeError('TRIAR dual requires target FULL graphs')
  with capture_mode(False):original(self,create_forward_fn,progress_bar_desc+' TRIAR OFF')
  off=self.graphs
  self.graphs={}
  self._capture_descs={full:descriptors[full]}
  try:
   qualified={}
   def on_factory(desc,warmup):
    fn=create_forward_fn(desc,warmup)
    def forward(mode):
     before=fast_calls()
     result=fn(mode)
     if not warmup and desc.num_tokens<=64:
      count=fast_calls()-before
      if count<=0:raise RuntimeError('TRIAR ON graph contains no qualified collective')
      qualified[str((desc.num_tokens,desc.num_reqs,desc.uniform_token_count))]=count
     return result
    return forward
   with capture_mode(True):original(self,on_factory,progress_bar_desc+' TRIAR ON')
   on=self.graphs
   if set(off)!=set(on) or not on:raise RuntimeError('TRIAR dual graph coverage differs')
   self.graphs={desc:DualGraph(off[desc],on[desc]) for desc in off}
  except BaseException:
   self.graphs.clear();off.clear();self._graphs_captured=False
   raise
  finally:self._capture_descs=descriptors
  from vllm.distributed.parallel_state import get_tp_group
  shapes=sorted((d.num_tokens,d.num_reqs,d.uniform_token_count) for d in off)
  print(json.dumps(dict(triar='dual-captured',rank=get_tp_group().rank_in_group,
      graphs_per_bank=len(off),shapes=shapes,qualified=qualified,
      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),sort_keys=True),flush=True)
 cls.capture=capture
 cls._triar_dual_installed=True


def install():
 """Compose with B5's existing import hook; never bypass its hash or narrow-graph hook."""
 global _INSTALLED
 if _INSTALLED:return
 import b5_prefix_verify as b5
 name='vllm.v1.worker.gpu.cudagraph_utils'
 original=b5.TARGETS[name]
 def combined(module):
  original(module)
  install_manager(module)
 b5.TARGETS[name]=combined
 module=sys.modules.get(name)
 # A cyclic import may expose the module before its classes exist; B5's loader
 # calls the composed hook after exec_module completes.
 if module is not None and hasattr(module,'ModelCudaGraphManager'):install_manager(module)
 _INSTALLED=True
