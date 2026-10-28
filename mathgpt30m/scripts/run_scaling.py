"""Nine independent proxy training runs; export empirical (N,D,loss) points.
The scaling optimum is theoretical; it never changes the 30M training target.
"""
import argparse,json,math,subprocess,sys,time
from pathlib import Path
# Allow both `python scripts/name.py` and `python -m scripts.name`.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0,str(_Path(__file__).resolve().parents[1]))
from mathgpt.experiment import read_final
from mathgpt.model import Config
from scripts.scale_to_30m import count

SIZES=[(128,4),(192,6),(256,8)]
TOKENS=[5_000_000,15_000_000,40_000_000]

def main():
    p=argparse.ArgumentParser();p.add_argument('--selected',default='runs/greedy/selected_architecture.json')
    p.add_argument('--data',default='data/processed');p.add_argument('--out',default='runs/scaling')
    p.add_argument('--seq',type=int,default=256);p.add_argument('--batch',type=int,default=8)
    p.add_argument('--accum',type=int,default=2);p.add_argument('--max_hours',type=float,default=8.)
    p.add_argument('--cpu',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();selected=json.loads(Path(a.selected).read_text());root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    rows=[];start=time.monotonic()
    for dim,layers in SIZES:
        hidden=round(dim*2.75/32)*32
        heads=selected['heads']
        if dim%heads:raise ValueError('Proxy dim must be divisible by selected heads')
        cfg=Config(vocab_size=selected['vocab'],dim=dim,layers=layers,heads=heads,
                   kv_heads=selected['kv_heads'],hidden=hidden,max_seq=a.seq,
                   position=selected['position'],norm_type=selected['norm_type'],
                   ffn_type=selected['ffn_type'],tie_embeddings=not selected['untie'],
                   residual_scale=not selected.get('no_residual_scale',False))
        n=count(cfg)
        for budget in TOKENS:
            steps=math.ceil(budget/(a.seq*a.batch*a.accum))
            out=root/f'N{n}_D{budget}'
            cmd=[sys.executable,'-m','mathgpt.train','--data',a.data,'--out',str(out),
                 '--dim',str(dim),'--layers',str(layers),'--heads',str(heads),
                 '--kv_heads',str(selected['kv_heads']),'--hidden',str(hidden),
                 '--vocab',str(selected['vocab']),'--seq',str(a.seq),'--batch',str(a.batch),
                 '--accum',str(a.accum),'--steps',str(steps),'--warmup',str(max(1,steps//20)),
                 '--decay',str(max(1,steps//10)),'--eval_every',str(max(1,steps//5)),
                 '--eval_batches','32','--save_every',str(steps),
                 '--position',cfg.position,'--norm_type',cfg.norm_type,'--ffn_type',cfg.ffn_type]
            if selected['untie']:cmd.append('--untie')
            if selected.get('no_residual_scale'):cmd.append('--no_residual_scale')
            if a.cpu:cmd.append('--cpu')
            print(' '.join(cmd),flush=True)
            if a.dry_run:continue
            if (out/'metrics.jsonl').exists():
                metrics=read_final(out/'metrics.jsonl')
                if metrics['step']!=steps-1:raise RuntimeError(f'Incomplete run: {out}')
            else:
                if (time.monotonic()-start)/3600>a.max_hours:raise TimeoutError('Scaling budget exceeded')
                subprocess.run(cmd,check=True);metrics=read_final(out/'metrics.jsonl')
            rows.append({'N':metrics['parameters'],'D':metrics['tokens_seen'],
                         'loss':metrics['val_loss'],'path':str(out)})
            (root/'points.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    if not a.dry_run:print(f'Wrote {len(rows)} empirical points to {root}/points.jsonl')
if __name__=='__main__':main()
