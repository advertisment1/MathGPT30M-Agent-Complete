import argparse, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mathgpt.model import GPT,Config
from mathgpt.data import PackedTokens
p=argparse.ArgumentParser();p.add_argument('--data',default='data/processed/train.bin');p.add_argument('--steps',type=int,default=250);a=p.parse_args()
dev='cuda' if torch.cuda.is_available() else 'cpu'
c=Config(vocab_size=32000,dim=64,layers=2,heads=4,kv_heads=2,hidden=176,max_seq=64)
m=GPT(c).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=1e-3)
x,y=PackedTokens(a.data,64).batch(2,dev)
first=None
for i in range(a.steps):
    opt.zero_grad();_,loss,_=m(x,y);loss.backward();opt.step()
    if first is None:first=loss.item()
    if i%25==0 or i==a.steps-1:print(i,loss.item())
assert loss.item()<first*.5,'Overfit test failed: loss not reduced enough'
