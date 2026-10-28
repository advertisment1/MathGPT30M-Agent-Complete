"""Train the ~30M config selected by scale_to_30m.py, without editing model code."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/final_30m.json')
    p.add_argument('--data',default='data/processed');p.add_argument('--out',default='runs/final_30m')
    p.add_argument('--steps',type=int,default=5000);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--accum',type=int,default=8);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();c=json.loads(Path(a.config).read_text())
    cmd=[sys.executable,'-m','mathgpt.train','--data',a.data,'--out',a.out,
         '--steps',str(a.steps),'--batch',str(a.batch),'--accum',str(a.accum),
         '--warmup',str(max(1,a.steps//20)),'--decay',str(max(1,a.steps//10))]
    mapping={'vocab_size':'vocab','dim':'dim','layers':'layers','heads':'heads',
             'kv_heads':'kv_heads','hidden':'hidden','max_seq':'seq',
             'position':'position','norm_type':'norm_type','ffn_type':'ffn_type'}
    for source,dest in mapping.items():cmd+=['--'+dest,str(c[source])]
    if not c['tie_embeddings']:cmd.append('--untie')
    if not c['residual_scale']:cmd.append('--no_residual_scale')
    print(' '.join(cmd),flush=True)
    if not a.dry_run:subprocess.run(cmd,check=True)
if __name__=='__main__':main()
