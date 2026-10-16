# SPDX-License-Identifier: Apache-2.0
"""Bounded original-release routing LoRA/head adaptation, separated from review-gate weights."""
import argparse, dataclasses, hashlib, json, os, shutil, sys, threading, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
from routing_data import sha
from routing_policy import questions,MODEL_ID,POLICY_VERSION
from contract import sha as contract_sha


def digest_params(named):
 import torch
 h=hashlib.sha256()
 for name,p in named:
  h.update(name.encode());h.update(str(p.dtype).encode());h.update(str(tuple(p.shape)).encode());flat=p.detach().reshape(-1)
  for start in range(0,p.numel(),1024*1024):h.update(flat[start:start+1024*1024].to('cpu').contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()


def verify_protocol(data):
 protocol=json.loads((data/'protocol.json').read_text())
 assert protocol['questions_sha256']==sha(questions())
 for name,digest in protocol['dataset_hashes'].items():assert hashlib.sha256((data/name).read_bytes()).hexdigest()==digest
 return protocol


def export(source,dest,meta,step):
 from safetensors.torch import save_file
 dest.mkdir(parents=True,exist_ok=False)
 for name in ('adapter_config.json','adapter_model.safetensors'):shutil.copy2(source/name,dest/name)
 save_file({k:v.detach().cpu().contiguous() for k,v in meta.head.items()},dest/'head.safetensors')
 config={k:v for k,v in meta.to_dict().items() if k in ('base','base_revision','lora','head_dim','option_isolation','special_embeddings','temperature','weights')}
 config.update(base_revision='1001bb4d826a52d1f399e183466143f4da7b741b',tokenizer_revision='1001bb4d826a52d1f399e183466143f4da7b741b',
  model_id=MODEL_ID,policy_version=POLICY_VERSION,questions_sha256=contract_sha(questions()),selected_step=step,weights_dtype='bf16',
  inference_backend='mlx',backbone_dtype='bfloat16',head_dtype='float32',max_state=384,max_branch=512,max_packed=512)
 (dest/'model_config.json').write_text(json.dumps(config,indent=2)+'\n')
 manifest={'model':MODEL_ID,'policy_version':POLICY_VERSION,'public_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.iterdir() if p.is_file()}}
 (dest/'routing_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 return manifest


def main():
 p=argparse.ArgumentParser();p.add_argument('--base-dir',required=True);p.add_argument('--original-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--data-dir',default=str(ROOT/'routing/data'));p.add_argument('--target',type=int,choices=[0,2,25,50],required=True);a=p.parse_args()
 data=Path(a.data_dir);protocol=verify_protocol(data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 target=a.target;start=0 if target in (0,2,25) else 25
 stage=out/f'stage-{target:04d}';stage.mkdir(exist_ok=False)
 from isolation import isolate
 isolate();sys.path.insert(0,str(ROOT/'vendor'))
 import torch,psutil,signal
 from kev.checkpoint import Checkpoint,write_meta
 from kev.model import DecisionModel,load_tokenizer,encode
 from kev.data import materialize
 ck=Checkpoint(a.original_dir if not start else str(out/f'stage-{start:04d}'/'checkpoint'));meta=ck.meta
 if target==0:
  export(Path(a.original_dir),stage/'weights',meta,0);(stage/'report.json').write_text(json.dumps({'status':'completed','step':0,'initialization':'pinned_original_untrained_on_routing'}));return
 if not torch.backends.mps.is_available():raise ValueError('MPS_required_no_CPU_fallback')
 torch.set_num_threads(2);torch.mps.set_per_process_memory_fraction(.6);torch.manual_seed(321)
 report={'status':'running','seed':321,'start':start,'target':target,'steps':[],'protocol_sha256':hashlib.sha256((data/'protocol.json').read_bytes()).hexdigest()};began=time.perf_counter();swap=psutil.swap_memory().used;stop=threading.Event()
 def memory():return {'rss':psutil.Process().memory_info().rss,'available':psutil.virtual_memory().available,'swap_delta':psutil.swap_memory().used-swap,'mps':torch.mps.driver_allocated_memory()}
 def write():
  report['elapsed_seconds']=round(time.perf_counter()-began,3);report['memory']=memory();(stage/'report.json').write_text(json.dumps(report,indent=2))
 def monitor():
  while not stop.wait(2):
   m=memory();report['memory']=m
   if m['available']<4*1024**3 or m['swap_delta']>512*1024**2 or time.perf_counter()-began>900:
    report['status']='safety_stop';write();os.kill(os.getpid(),signal.SIGINT);return
 write();threading.Thread(target=monitor,daemon=True).start()
 try:
  tok=load_tokenizer(a.base_dir)
  # Check every split before training; final labels are never used in optimization or selection.
  for name in ('train','dev','final_test'):
   rows=json.loads((data/f'{name}.json').read_text())['cases'];lengths=[]
   for c in rows:
    q=questions();q['route']['label']=c['expected'];q['route']['src']='public_synthetic_routing';r=materialize({'state':c['state'],'questions':q});enc=encode(tok,r,max_state=384,max_branch=512,strict=True);lengths.append(len(enc['ids']))
   report[name+'_max_packed_tokens']=max(lengths)
  model=DecisionModel(a.base_dir,tok,'mps',lora=meta.lora,head_dim=meta.head_dim,option_isolation=meta.option_isolation,special_embeddings=meta.special_embeddings,dtype=torch.bfloat16,direct_load=True)
  model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.lm.config.use_cache=False
  ours=dataclasses.replace(meta,head=None,weights_dtype='bf16',extra={});report['warm_start']=ck.warm_start(model,ours);model.head.temperature=meta.temperature
  named=list(model.named_parameters());base=[(n,p) for n,p in named if not p.requires_grad];adapter=[(n,p) for n,p in named if p.requires_grad and 'lora_' in n];head=list(model.head.named_parameters());trainable=list(model.trainable_parameters())
  assert {id(p) for p in trainable}=={id(p) for _,p in adapter+head}
  before=digest_params(base);abefore=digest_params(adapter);hbefore=digest_params(head)
  optimizer=torch.optim.AdamW(trainable,lr=2e-5,weight_decay=.01)
  if start:
   resume=torch.load(out/f'stage-{start:04d}'/'resume.pt',map_location='cpu',weights_only=True);optimizer.load_state_dict(resume['optimizer']);torch.set_rng_state(resume['cpu_rng']);torch.mps.set_rng_state(resume['mps_rng'])
  cases=json.loads((data/'train.json').read_text())['cases']
  from kev.train import question_loss
  for step in range(start+1,target+1):
   t=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
   for g in optimizer.param_groups:g['lr']=2e-5*min(1,step/10)
   c=cases[protocol['training_order'][step-1]];q=questions();q['route']['label']=c['expected'];q['route']['src']='public_synthetic_routing';rec=materialize({'state':c['state'],'questions':q});enc=model.encode(tok,rec,max_state=384,max_branch=512,strict=True)
   loss=question_loss(model(enc)[0],rec['questions'][0],'mps',ord_w=0);loss.backward();torch.mps.synchronize()
   def norm(params):return float(torch.stack([v.grad.detach().float().square().sum() for _,v in params if v.grad is not None]).sum().sqrt().cpu())
   ga,gh=norm(adapter),norm(head);assert ga>=0 and gh>=0 and all(v.grad is None for _,v in base)
   torch.nn.utils.clip_grad_norm_(trainable,1,error_if_nonfinite=True);optimizer.step();torch.mps.synchronize()
   row={'step':step,'loss':float(loss.detach().cpu()),'adapter_grad_norm':ga,'head_grad_norm':gh,'seconds':round(time.perf_counter()-t,3),'packed_tokens':len(enc['ids'])};report['steps'].append(row);write();print(json.dumps(row),flush=True);torch.mps.empty_cache()
  assert any(x['adapter_grad_norm']>0 for x in report['steps']) and any(x['head_grad_norm']>0 for x in report['steps'])
  after=digest_params(base);assert before==after
  report.update(base_frozen=True,base_sha256_before=before,base_sha256_after=after,adapter_changed=abefore!=digest_params(adapter),head_changed=hbefore!=digest_params(head));assert report['adapter_changed'] and report['head_changed']
  checkpoint=stage/'checkpoint';model.lm.save_pretrained(checkpoint);ours.head={k:v.detach().cpu() for k,v in model.head.state_dict().items()};write_meta(checkpoint,ours)
  def cpu(x):
   if isinstance(x,torch.Tensor):return x.detach().cpu()
   if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
   if isinstance(x,list):return [cpu(v) for v in x]
   if isinstance(x,tuple):return tuple(cpu(v) for v in x)
   return x
  report['initialization']='original_pinned_release' if not start else 'resume_same_run_optimizer_and_rng';torch.save({'optimizer':cpu(optimizer.state_dict()),'cpu_rng':torch.get_rng_state(),'mps_rng':torch.mps.get_rng_state()},stage/'resume.pt')
  report['export']=export(checkpoint,stage/'weights',ours,target);report['status']='completed';write()
 except BaseException:
  if report['status']!='safety_stop':report['status']='failed'
  write();raise
 finally:stop.set()

if __name__=='__main__':main()
