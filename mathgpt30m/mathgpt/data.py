import hashlib, json, random
from pathlib import Path
import numpy as np

DATASET='HuggingFaceTB/finemath'

def documents(config='finemath-4plus', seed=42, buffer=1000, limit=None):
    from datasets import load_dataset
    ds=load_dataset(DATASET, config, split='train', streaming=True)
    ds=ds.shuffle(seed=seed, buffer_size=buffer)
    for i, row in enumerate(ds):
        if limit is not None and i>=limit: break
        text=row.get('text','')
        if isinstance(text,str) and len(text)>80:
            yield text

def split_doc(text, val_fraction=.02):
    digest=int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:12],16)
    return 'val' if digest%100000 < int(val_fraction*100000) else 'train'

def train_tokenizer(path, samples=30000, vocab_size=32000, config='finemath-4plus'):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers, decoders
    tok=Tokenizer(models.BPE(unk_token='[UNK]'))
    tok.normalizer=normalizers.NFKC()
    tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder=decoders.ByteLevel()
    trainer=trainers.BpeTrainer(vocab_size=vocab_size,min_frequency=2,special_tokens=['[UNK]','[EOS]'],initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(documents(config=config,limit=samples),trainer=trainer)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    tok.save(str(path)); return tok

def build_tokens(tokenizer_path, out_dir, config='finemath-4plus', docs=200000, val_fraction=.02):
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(tokenizer_path)); out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    # Disk-backed binary streams: no full dataset materialization in RAM.
    counts={'train':0,'val':0}; f={s:open(out/f'{s}.bin','wb') for s in counts}
    try:
        for text in documents(config=config,limit=docs):
            split=split_doc(text,val_fraction)
            ids=tok.encode(text).ids+[tok.token_to_id('[EOS]')]
            if max(ids)>=65536: raise ValueError('uint16 requires vocab_size <= 65536')
            np.asarray(ids,dtype=np.uint16).tofile(f[split]);counts[split]+=len(ids)
    finally:
        for handle in f.values():handle.close()
    (out/'manifest.json').write_text(json.dumps({'tokens':counts,'config':config,'documents_max':docs,'split':'sha256 document 2% validation','dtype':'uint16','tokenizer':str(tokenizer_path)},indent=2))
    return counts

class PackedTokens:
    def __init__(self,path,seq_len,seed=0):
        self.a=np.memmap(path,dtype=np.uint16,mode='r');self.seq_len=seq_len;self.rng=np.random.default_rng(seed)
        if len(self.a)<seq_len+2:raise ValueError(f'Not enough tokens in {path}')
    def batch(self,batch_size,device):
        import torch
        starts=self.rng.integers(0,len(self.a)-self.seq_len-1,size=batch_size)
        chunks=np.stack([np.asarray(self.a[i:i+self.seq_len+1],dtype=np.int64) for i in starts])
        t=torch.from_numpy(chunks).to(device)
        return t[:,:-1],t[:,1:]
