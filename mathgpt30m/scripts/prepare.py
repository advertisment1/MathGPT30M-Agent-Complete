import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mathgpt.data import train_tokenizer,build_tokens
p=argparse.ArgumentParser();p.add_argument('--tokenizer',default='data/tokenizer.json');p.add_argument('--out',default='data/processed');p.add_argument('--config',default='finemath-4plus');p.add_argument('--tokenizer_docs',type=int,default=30000);p.add_argument('--docs',type=int,default=200000);p.add_argument('--vocab',type=int,default=32000)
a=p.parse_args()
train_tokenizer(a.tokenizer,a.tokenizer_docs,a.vocab,a.config)
print(build_tokens(a.tokenizer,a.out,a.config,a.docs))
