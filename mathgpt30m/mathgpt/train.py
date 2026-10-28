import argparse, contextlib, json, math, os, random, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .model import GPT,Config
from .data import PackedTokens

def wsd_lr(step,total,warmup,decay,peak,min_ratio=.1):
    if warmup>0 and step<warmup:return peak*(step+1)/warmup
    start=max(warmup,total-decay)
    if step<start:return peak
    u=min(1.,max(0.,(step-start)/max(1,total-start-1)))
    return peak*(min_ratio+(1-min_ratio)*(1-u))

def make_optimizer(model,lr,weight_decay=.1):
    decay=[];no_decay=[]
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        # tied embedding is 2D but intentionally excluded from weight decay
        (no_decay if p is model.embed.weight or p.ndim<2 else decay).append(p)
    return torch.optim.AdamW([{'params':decay,'weight_decay':weight_decay},{'params':no_decay,'weight_decay':0.}],lr=lr,betas=(.9,.95),eps=1e-8)

def evaluate(model,data,batch,steps,device,autocast):
    model.eval();losses=[]
    with torch.no_grad(),autocast:
        for _ in range(steps):
            x,y=data.batch(batch,device);_,loss,_=model(x,y);losses.append(loss.item())
    model.train();return sum(losses)/len(losses)

def run(a):
    torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    device='cuda' if torch.cuda.is_available() and not a.cpu else 'cpu'
    bf16=device=='cuda' and torch.cuda.is_bf16_supported()
    ctx=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()
    cfg=Config(vocab_size=a.vocab,dim=a.dim,layers=a.layers,heads=a.heads,kv_heads=a.kv_heads,hidden=a.hidden,max_seq=a.seq,position=a.position,norm_type=a.norm_type,ffn_type=a.ffn_type,tie_embeddings=not a.untie,residual_scale=not a.no_residual_scale)
    model=GPT(cfg).to(device);opt=make_optimizer(model,a.lr)
    train=PackedTokens(Path(a.data)/'train.bin',a.seq,a.seed)
    val=PackedTokens(Path(a.data)/'val.bin',a.seq,a.seed+1)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    start=0;seen=0
    if a.resume:
        ck=torch.load(a.resume,map_location=device,weights_only=False)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['step']+1;seen=ck.get('tokens_seen',0)
        torch.set_rng_state(ck['rng_cpu'])
        if device=='cuda' and ck.get('rng_cuda') is not None:torch.cuda.set_rng_state(ck['rng_cuda'])
    params=sum(p.numel() for p in model.parameters())
    print('parameters',params,'device',device,'bf16',bf16,flush=True)
    (out/'model_config.json').write_text(json.dumps(asdict(cfg),indent=2))
    with (out/'metrics.jsonl').open('a',encoding='utf-8') as log:
        for step in range(start,a.steps):
            tic=time.perf_counter();opt.zero_grad(set_to_none=True);loss_sum=0.
            for _ in range(a.accum):
                x,y=train.batch(a.batch,device)
                with ctx():_,loss,_=model(x,y);loss=loss/a.accum
                loss.backward();loss_sum+=loss.item()
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            lr=wsd_lr(step,a.steps,a.warmup,a.decay,a.lr)
            for g in opt.param_groups:g['lr']=lr
            opt.step();seen+=a.batch*a.accum*a.seq
            if device=='cuda':torch.cuda.synchronize()
            duration=time.perf_counter()-tic
            row={'step':step,'train_loss':loss_sum,'lr':lr,'grad_norm':grad,'tokens_seen':seen,'tok_per_s':a.batch*a.accum*a.seq/duration,'parameters':params,'elapsed_step_s':duration}
            if step%a.eval_every==0 or step==a.steps-1:row['val_loss']=evaluate(model,val,a.batch,a.eval_batches,device,ctx())
            log.write(json.dumps(row)+'\n');log.flush()
            if step%a.log_every==0:print(row,flush=True)
            if (step+1)%a.save_every==0 or step==a.steps-1:
                torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'config':asdict(cfg),'step':step,'tokens_seen':seen,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state() if device=='cuda' else None},out/'last.pt')
    (out/'run_config.json').write_text(json.dumps(vars(a),indent=2))

def parser():
    p=argparse.ArgumentParser()
    for name,default in [('data','data/processed'),('out','runs/base'),('resume','')]:p.add_argument('--'+name,default=default)
    for name,default in [('vocab',32000),('dim',256),('layers',8),('heads',8),('kv_heads',2),('hidden',704),('seq',512),('batch',4),('accum',8),('steps',2000),('warmup',100),('decay',300),('eval_every',100),('eval_batches',10),('save_every',250),('log_every',25),('seed',42)]:p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--position',choices=['rope','nope'],default='rope')
    p.add_argument('--norm_type',choices=['rms','layer'],default='rms')
    p.add_argument('--ffn_type',choices=['swiglu','gelu'],default='swiglu')
    p.add_argument('--untie',action='store_true')
    p.add_argument('--no_residual_scale',action='store_true')
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--cpu',action='store_true')
    return p
if __name__=='__main__':run(parser().parse_args())
