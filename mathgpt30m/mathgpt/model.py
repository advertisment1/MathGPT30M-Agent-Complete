from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class Config:
    vocab_size: int = 32000
    dim: int = 256
    layers: int = 8
    heads: int = 8
    kv_heads: int = 2
    hidden: int = 704
    max_seq: int = 1024
    rope_base: float = 10000.0
    position: str = "rope"  # rope | nope
    norm_type: str = "rms"  # rms | layer
    ffn_type: str = "swiglu"  # swiglu | gelu
    tie_embeddings: bool = True
    residual_scale: bool = True

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight

def rope(x, positions, base):
    d = x.size(-1)
    freq = 1.0 / (base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    angle = positions.float().unsqueeze(-1) * freq
    cos, sin = angle.cos()[None, None], angle.sin()[None, None]
    a, b = x.float()[..., ::2], x.float()[..., 1::2]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2).to(x.dtype)

class GELUFFN(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.up=nn.Linear(c.dim,c.hidden,bias=False)
        self.down=nn.Linear(c.hidden,c.dim,bias=False)
    def forward(self,x):
        return self.down(F.gelu(self.up(x)))

def make_norm(c):
    return RMSNorm(c.dim) if c.norm_type=="rms" else nn.LayerNorm(c.dim)

class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        assert c.dim % c.heads == 0 and c.heads % c.kv_heads == 0
        assert c.position in ("rope", "nope")
        self.c = c
        self.d = c.dim // c.heads
        self.q = nn.Linear(c.dim, c.heads * self.d, bias=False)
        self.k = nn.Linear(c.dim, c.kv_heads * self.d, bias=False)
        self.v = nn.Linear(c.dim, c.kv_heads * self.d, bias=False)
        self.out = nn.Linear(c.dim, c.dim, bias=False)
    def forward(self, x, cache=None):
        b, t, _ = x.shape
        past = 0 if cache is None else cache[0].shape[-2]
        assert past + t <= self.c.max_seq
        positions = torch.arange(past, past+t, device=x.device)
        q = self.q(x).view(b,t,self.c.heads,self.d).transpose(1,2)
        k = self.k(x).view(b,t,self.c.kv_heads,self.d).transpose(1,2)
        if self.c.position == "rope":
            q = rope(q, positions, self.c.rope_base)
            k = rope(k, positions, self.c.rope_base)
        v = self.v(x).view(b,t,self.c.kv_heads,self.d).transpose(1,2)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=-2)
            v = torch.cat([cache[1], v], dim=-2)
        new_cache = (k, v)
        k = k.repeat_interleave(self.c.heads // self.c.kv_heads, dim=1)
        v = v.repeat_interleave(self.c.heads // self.c.kv_heads, dim=1)
        scores = (q @ k.transpose(-1,-2)) / math.sqrt(self.d)
        keys = torch.arange(past+t, device=x.device)
        mask = keys[None, :] <= positions[:, None]
        probs = F.softmax(scores.float().masked_fill(~mask, float('-inf')), dim=-1).to(q.dtype)
        y = (probs @ v).transpose(1,2).contiguous().view(b,t,self.c.dim)
        return self.out(y), new_cache

class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Linear(c.dim,c.hidden,bias=False)
        self.up = nn.Linear(c.dim,c.hidden,bias=False)
        self.down = nn.Linear(c.hidden,c.dim,bias=False)
    def forward(self,x):
        return self.down(F.silu(self.gate(x))*self.up(x))

class Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.n1,self.n2=make_norm(c),make_norm(c)
        self.attn=Attention(c)
        self.ffn=SwiGLU(c) if c.ffn_type=="swiglu" else GELUFFN(c)
    def forward(self,x,cache=None):
        y,cache=self.attn(self.n1(x),cache)
        x=x+y
        return x+self.ffn(self.n2(x)),cache

class GPT(nn.Module):
    def __init__(self,c=Config()):
        super().__init__()
        self.c=c
        self.embed=nn.Embedding(c.vocab_size,c.dim)
        self.blocks=nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm=make_norm(c)
        self.lm_head=nn.Linear(c.dim,c.vocab_size,bias=False)
        if c.tie_embeddings:
            self.lm_head.weight=self.embed.weight
        assert c.norm_type in ("rms", "layer") and c.ffn_type in ("swiglu", "gelu")
        self.apply(self._init)
        if c.residual_scale:
            for block in self.blocks:
                nn.init.normal_(block.attn.out.weight,std=0.02/math.sqrt(2*c.layers))
                nn.init.normal_(block.ffn.down.weight,std=0.02/math.sqrt(2*c.layers))
    @staticmethod
    def _init(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            nn.init.normal_(m.weight,mean=0,std=0.02)
    def forward(self,ids,targets=None,caches=None):
        x=self.embed(ids)
        updated=[]
        for i,block in enumerate(self.blocks):
            x,kv=block(x,None if caches is None else caches[i])
            updated.append(kv)
        logits=self.lm_head(self.norm(x))
        loss=None if targets is None else F.cross_entropy(logits.reshape(-1,self.c.vocab_size).float(),targets.reshape(-1))
        return logits,loss,updated
    @torch.no_grad()
    def generate(self,ids,max_new_tokens=32,temperature=1.,top_k=None,top_p=1.,use_cache=True):
        self.eval()
        caches=None
        if ids.size(1) + max_new_tokens > self.c.max_seq:
            raise ValueError("generation exceeds max_seq; shorten prompt or output")
        for _ in range(max_new_tokens):
            if use_cache and caches is not None:
                inp=ids[:,-1:]
            else:
                inp=ids[:,-self.c.max_seq:]
                if not use_cache: caches=None
            logits,_,new_caches=self(inp,caches=caches)
            if use_cache: caches=new_caches
            scores=logits[:,-1,:].float()
            if temperature<=0:
                nxt=scores.argmax(-1,keepdim=True)
            else:
                scores=scores/temperature
                if top_k is not None:
                    threshold=scores.topk(min(top_k,scores.size(-1)),dim=-1).values[:,-1:]
                    scores=scores.masked_fill(scores<threshold,float('-inf'))
                if top_p<1:
                    sorted_scores,indices=torch.sort(scores,descending=True,dim=-1)
                    cumulative=sorted_scores.softmax(-1).cumsum(-1)
                    drop=(cumulative-sorted_scores.softmax(-1))>=top_p
                    sorted_scores=sorted_scores.masked_fill(drop,float('-inf'))
                    scores=torch.full_like(scores,float('-inf')).scatter(-1,indices,sorted_scores)
                nxt=torch.multinomial(scores.softmax(-1),1)
            ids=torch.cat((ids,nxt),dim=1)
        return ids

if __name__=='__main__':
    m=GPT()
    n=sum(p.numel() for p in m.parameters())
    print('parameters:',n, 'million:',round(n/1e6,2))
    x=torch.randint(0,m.c.vocab_size,(2,16))
    logits,loss,_=m(x,x)
    loss.backward()
    print('logits:',tuple(logits.shape),'loss:',loss.item())
