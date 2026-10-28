"""Sequential controlled experiments, with independent output folders and summary.
Run after preparing data; does not require network access once token files exist.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/ablation_variants.json')
    p.add_argument('--data',default='data/processed')
    p.add_argument('--out',default='runs/ablations')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--cpu',action='store_true')
    args=p.parse_args()
    config=json.loads(Path(args.config).read_text())
    root=Path(args.out)
    root.mkdir(parents=True,exist_ok=True)
    summary=[]
    for variant in config['variants']:
        name=variant['name']
        options={**config['shared'],**{k:v for k,v in variant.items() if k!='name'}}
        cmd=[sys.executable,'-m','mathgpt.train','--data',args.data,'--out',str(root/name)]
        for key,value in options.items():
            if isinstance(value,bool):
                if value:cmd.append('--'+key)
            else:cmd.extend(['--'+key,str(value)])
        if args.cpu:cmd.append('--cpu')
        print(name,' '.join(cmd),flush=True)
        if not args.dry_run:
            subprocess.run(cmd,check=True)
            metrics=[json.loads(line) for line in (root/name/'metrics.jsonl').read_text().splitlines()]
            last=next((m for m in reversed(metrics) if 'val_loss' in m),None)
            if last is None:raise RuntimeError('missing validation metric for '+name)
            summary.append({'name':name,'val_loss':last['val_loss'],'tokens_seen':last['tokens_seen'],'last_tok_per_s':last['tok_per_s']})
            (root/'summary.json').write_text(json.dumps(summary,indent=2))
    if not args.dry_run:
        print('Ranked by validation loss (not parameter-normalized):')
        for row in sorted(summary,key=lambda r:r['val_loss']):print(row)

if __name__=='__main__':main()
