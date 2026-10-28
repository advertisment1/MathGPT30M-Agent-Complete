"""Freeze selected architecture and choose dimensions closest to 30M params."""
import argparse,json
from pathlib import Path
# Allow both `python scripts/name.py` and `python -m scripts.name`.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0,str(_Path(__file__).resolve().parents[1]))
from mathgpt.model import Config,GPT


def count(c):
    # Exact for this codebase: tied/untied embeddings, norms, projections, FFN.
    d,h,k,l,f,v=c.dim,c.heads,c.kv_heads,c.layers,c.hidden,c.vocab_size
    attn=2*d*d+2*d*(k*(d//h))
    ff=(3 if c.ffn_type=='swiglu' else 2)*d*f
    norm=(2*d if c.norm_type=='rms' else 4*d)
    final=d if c.norm_type=='rms' else 2*d
    return v*d*(1 if c.tie_embeddings else 2)+l*(attn+ff+norm)+final


def find(base,target=30_000_000):
    choices=[]
    for d in [192,224,256,288,320,352,384,416,448,480,512]:
        if d%base['heads']:continue
        for layers in [4,6,8,10,12,14,16]:
            for factor in [2.5,2.667,3.0,3.5,4.0]:
                hidden=round(d*factor/32)*32
                c=Config(vocab_size=base.get('vocab',32000),dim=d,layers=layers,
                         heads=base['heads'],kv_heads=base['kv_heads'],hidden=hidden,
                         max_seq=1024,position=base['position'],norm_type=base['norm_type'],
                         ffn_type=base['ffn_type'],tie_embeddings=not base['untie'],
                         residual_scale=not base.get('no_residual_scale',False))
                n=count(c)
                choices.append((abs(n-target),n,c))
    return min(choices,key=lambda x:x[0])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selected',default='runs/greedy/selected_architecture.json')
    p.add_argument('--out',default='configs/final_30m.json');p.add_argument('--target',type=int,default=30_000_000)
    a=p.parse_args();base=json.loads(Path(a.selected).read_text());_,n,c=find(base,a.target)
    from dataclasses import asdict
    obj=asdict(c);obj['actual_parameters']=n;obj['target_parameters']=a.target
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(obj,indent=2))
    print(json.dumps(obj,indent=2))
