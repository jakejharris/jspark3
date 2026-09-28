#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only proof of the resident TRIAR code's inactive Cadence graph path."""
import sys
sys.dont_write_bytecode = True
import _diagnostics as diagnostics
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def need(value, reason):
    if not value:
        raise ValueError(reason)


PROBE=r'''
import json,hashlib,os,sys,time
from pathlib import Path
root=Path('/tmp/b45/graphs');site=Path('/usr/local/lib/python3.12/dist-packages')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources={name:sha(site/name) for name in json.loads(sys.argv[1])}
files={};rows={}
for kind in ('activation','captures','adaptive-capture'):
 rows[kind]=[]
 for p in sorted(root.glob(kind+'-*.jsonl')):
  files[p.name]=sha(p)
  for line in p.read_text().splitlines():
   value=json.loads(line);value['_file']=p.name;rows[kind].append(value)
dots={}
for row in rows['captures']:
 p=root/row['dot'];assert p.parent==root and p.is_file()
 text=p.read_text();dots[p.name]={'sha256':sha(p),'triar_marker':bool(__import__('re').search(r'triar|triangle_all_reduce',text,__import__('re').I))}
epoch=Path('/evidence/triar-epoch.json')
print(json.dumps(dict(observed_at=time.time(),rank=int(os.environ['NODE_RANK']),sources=sources,files=files,rows=rows,dots=dots,epoch=json.loads(epoch.read_text()) if epoch.exists() else None)))
'''

def expected_sources(recipe):
    cadence = json.loads((recipe / 'config/cadence-contract.json').read_text())
    result = {'/opt/b45/' + name: digest for name, digest in cadence['modules'].items()}
    result[cadence['import_owner']['file']] = cadence['import_owner']['sha256']
    result['vllm/v1/worker/gpu/cudagraph_utils.py'] = '6b44f24e65e51a0a43c7a5d7d93ef5def8b880cff9cb671ba1e1302a79c757aa'
    result['vllm/v1/worker/gpu/model_runner.py'] = 'f84255d75435e84f44972d3fd25e53447f9d4d2edd8bff4f8c19dfb793448415'
    contract = json.loads((recipe / 'overlays/v17/triar/INSTALL_CONTRACT.json').read_text())
    result.update({row['path']: row['after_sha256'] for row in contract['transforms']['apply_triar.py']['targets']})
    return result


def evaluate(value, rank, sources, started_at):
 need(value['rank']==rank and value['sources']==sources,'OFF capture source/rank drift')
 begin=datetime.fromisoformat(started_at.replace('Z','+00:00')).timestamp()
 ready=value['observed_at']
 for group in value['rows'].values():
  need(all(int(r['rank'])==rank and begin<float(r['time'])<ready for r in group),'B4 evidence outside this boot')
 epoch=value['epoch'];need(epoch is None or (epoch.get('schema')=='triar-epoch/1' and epoch.get('mode')=='off'),'TRIAR ON epoch is forbidden')
 acts=value['rows']['activation'];need(len(acts)==1,'one complete B4 activation required')
 active=acts[0];need(active['active'] is True and active['hash_gate'] is True,'B4 activation/hash gate absent')
 need(set(active['banks'])=={'bf16_0','int8_0'},'wrong B4 banks')
 pid=active['_file'].removeprefix('activation-').removesuffix('.jsonl')
 caps=value['rows']['captures'];need(caps and all(r['_file']=='captures-'+pid+'.jsonl' and int(r['rank'])==rank for r in caps),'capture PID/rank mismatch')
 phases={phase:[r for r in caps if r['phase']==phase] for phase in ('profile','serving')}
 need(sum(map(len,phases.values()))==len(caps),'unknown capture phase')
 census=[]
 for phase,rows in phases.items():
  need(rows,'missing '+phase+' B4 capture')
  keys=[];objects=[];counts={bank:0 for bank in active['banks']}
  for row in rows:
   need(row['bank'] in counts and row['capture_order']==['bf16_0','int8_0'],'capture order/bank drift')
   key=(row['bank'],json.dumps(row['desc'],sort_keys=True));keys.append(key);objects.append(row['object']);counts[row['bank']]+=1
   dot=value['dots'][row['dot']];need(dot['sha256']==row['dot_sha256'] and not dot['triar_marker'],'graph dump missing/drifted or TRIAR graph')
   calls=row['calls'];need(len(calls)==34 and len({c['prefix'] for c in calls})==34,'partial B4 projection capture')
   desc=row['desc'];need(desc['mode']=='FULL' and desc['tokens']>0,'invalid native descriptor')
   want='int8' if row['bank']=='int8_0' else ('bf16' if desc['tokens']<=8 else 'parent')
   need(all(c['path']==want and c['weight_ptr']>0 and c['control_ptr']>0 for c in calls),'B4 projection path/pointer drift')
  need(len(set(keys))==len(keys) and len(set(objects))==len(objects),'duplicate/aliased B4 graph')
  if phase=='serving':need(counts==active['banks'],'activation has incomplete graph banks')
  census.append(sorted(keys))
 need(census[0]==census[1],'profile/live B4 descriptor mismatch')
 adaptive=value['rows']['adaptive-capture'];need(len(adaptive)==2 and {r['phase'] for r in adaptive}=={'profile','serving'} and all(r['status']=='PASS' and r['_file']=='adaptive-capture-'+pid+'.jsonl' for r in adaptive),'adaptive graph coverage incomplete')
 for row in adaptive:
  need(row['descriptors']==[dict(requests=r,query_width=4,physical_rows=4*r) for r in range(1,9)],'adaptive physical descriptor coverage differs')
 return dict(rank=rank,status='PASS',worker_pid=int(pid),banks=active['banks'],triar_active=False,capture='completed original B45 banks; no dual wrapper')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--started-at', required=True, help='bound Docker State.StartedAt')
    args = parser.parse_args()
    expected = expected_sources(Path(__file__).resolve().parents[1])
    proc = subprocess.run([sys.executable, '-B', '-S', '-c', PROBE, json.dumps(sorted(expected))],
                          check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    diagnostics.retain(proc.stdout + proc.stderr)
    value = json.loads(proc.stdout)
    result = evaluate(value, int(__import__('os').environ['NODE_RANK']), expected, args.started_at)
    print(json.dumps({'schema': 'jspark3-triar-inactive/1', 'verdict': 'PASS', 'triar': 'INACTIVE_NCCL_B45',
                      'attestation': result, 'raw_sha256': diagnostics.fingerprint(value)}, sort_keys=True))


if __name__ == '__main__':
    diagnostics.install_exception_hook()
    main()
