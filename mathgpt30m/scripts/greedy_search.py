"""Six-round sequential greedy ablation with promotion and resumable ledger.
Each round: compare the CURRENT baseline to a single changed component.
If candidate wins by min_delta, promote it. Otherwise keep the baseline.
No claim of global optimality or statistical significance.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
# Allow both `python scripts/name.py` and `python -m scripts.name`.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0,str(_Path(__file__).resolve().parents[1]))
from mathgpt.experiment import read_final,choose

BASE={'dim':128,'layers':4,'heads':4,'kv_heads':2,'hidden':352,'vocab':32000,
      'seq':128,'position':'rope','norm_type':'rms','ffn_type':'swiglu',
      'untie':False,'no_residual_scale':False}
ROUNDS=[
    {'label':'attention','key':'kv_heads','values':[4,2,1]},
    {'label':'position','key':'position','values':['rope','nope']},
    {'label':'normalization','key':'norm_type','values':['rms','layer']},
    {'label':'feedforward','key':'ffn_type','values':['swiglu','gelu']},
    {'label':'embedding','key':'untie','values':[False,True]},
    {'label':'initialization','key':'no_residual_scale','values':[False,True]},
]

def cmd(config,args,output,seed):
    result=[sys.executable,'-m','mathgpt.train','--data',args.data,'--out',str(output),
            '--steps',str(args.steps),'--batch',str(args.batch),'--accum',str(args.accum),
            '--warmup',str(args.warmup),'--decay',str(args.decay),'--eval_every',str(args.eval_every),
            '--eval_batches',str(args.eval_batches),'--save_every',str(args.steps),
            '--seed',str(seed),'--lr',str(args.lr)]
    for k,v in config.items():
        if isinstance(v,bool):
            if v: result.append('--'+k)
        else:result.extend(['--'+k,str(v)])
    if args.cpu:result.append('--cpu')
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',default='data/processed');ap.add_argument('--out',default='runs/greedy')
    ap.add_argument('--steps',type=int,default=150);ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--accum',type=int,default=1);ap.add_argument('--warmup',type=int,default=10)
    ap.add_argument('--decay',type=int,default=30);ap.add_argument('--eval_every',type=int,default=50)
    ap.add_argument('--eval_batches',type=int,default=64);ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--min_delta',type=float,default=.002)
    ap.add_argument('--max_hours',type=float,default=5.);ap.add_argument('--cpu',action='store_true')
    ap.add_argument('--dry-run',action='store_true');ap.add_argument('--restart',action='store_true')
    a=ap.parse_args();root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    ledger=root/'decisions.json'; current=dict(BASE);history=[]
    if ledger.exists() and not a.restart:
        history=json.loads(ledger.read_text())
        if history:current=history[-1]['selected_config']
    start=time.monotonic()
    for index,round_ in enumerate(ROUNDS):
        if index<len(history):continue
        candidates=[current]
        for value in round_['values']:
            if value!=current[round_['key']]:
                variant=dict(current);variant[round_['key']]=value;candidates.append(variant)
        observations=[]
        for ci,variant in enumerate(candidates):
            out=root/f'round_{index+1:02d}_{round_["label"]}'/f'candidate_{ci}'
            command=cmd(variant,a,out,a.seed)
            print(' '.join(command),flush=True)
            if a.dry_run:continue
            if (out/'metrics.jsonl').exists():
                result=read_final(out/'metrics.jsonl')
                if result['step'] != a.steps-1:raise RuntimeError(f'Incomplete run {out}; use fresh --out')
            else:
                if (time.monotonic()-start)/3600>a.max_hours:raise TimeoutError('Search wall-time budget exceeded')
                subprocess.run(command,check=True)
                result=read_final(out/'metrics.jsonl')
            observations.append({'config':variant,'metrics':result,'path':str(out)})
        if a.dry_run:continue
        baseline=observations[0];best=baseline
        for cand in observations[1:]:
            if choose(best['metrics'],cand['metrics'],a.min_delta):best=cand
        current=dict(best['config'])
        decision={'round':index+1,'component':round_['label'],'baseline':baseline,
                  'candidates':observations,'promoted':best is not baseline,
                  'selected_config':current,'selected_metrics':best['metrics'],
                  'selection_rule':f'validation loss improvement >= {a.min_delta}; single seed'}
        history.append(decision);ledger.write_text(json.dumps(history,indent=2,ensure_ascii=False))
    if not a.dry_run:
        (root/'selected_architecture.json').write_text(json.dumps(current,indent=2))
        print('SELECTED',json.dumps(current,indent=2))
        print('WARNING: preliminary single-seed screening; rerun finalists with 3 seeds')
if __name__=='__main__':main()
