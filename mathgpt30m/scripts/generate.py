import argparse, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenizers import Tokenizer
from mathgpt.model import GPT,Config
p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/base/last.pt');p.add_argument('--tokenizer',default='data/tokenizer.json');p.add_argument('--prompt',default='The theorem states that');p.add_argument('--tokens',type=int,default=80);p.add_argument('--no_cache',action='store_true');a=p.parse_args()
dev='cuda' if torch.cuda.is_available() else 'cpu';ck=torch.load(a.checkpoint,map_location=dev,weights_only=False)
m=GPT(Config(**ck['config'])).to(dev);m.load_state_dict(ck['model']);tok=Tokenizer.from_file(a.tokenizer)
ids=torch.tensor([tok.encode(a.prompt).ids],device=dev)
print(tok.decode(m.generate(ids,max_new_tokens=a.tokens,use_cache=not a.no_cache)[0].tolist()))
