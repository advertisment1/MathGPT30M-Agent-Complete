import torch
from mathgpt.model import GPT,Config
from mathgpt.train import wsd_lr
from mathgpt.scaling import fit,optimum

def test_wsd():
    assert wsd_lr(0,100,10,20,1.)==.1
    assert wsd_lr(20,100,10,20,1.)==1.
    assert abs(wsd_lr(99,100,10,20,1.)-.1)<1e-9

def test_model_and_cache():
    c=Config(vocab_size=128,dim=32,layers=2,heads=4,kv_heads=2,hidden=64,max_seq=32)
    m=GPT(c).eval();x=torch.randint(0,128,(1,7))
    with torch.no_grad():
        full,_,_=m(x)
        _,_,cache=m(x[:,:5]);partial,_,_=m(x[:,5:],caches=cache)
    assert torch.allclose(full[:,5:],partial,atol=1e-4,rtol=1e-4)
    _,loss,_=m(x,x);loss.backward();assert torch.isfinite(loss)

def test_scaling_synthetic():
    rows=[]
    for n in [5,10,20,40]:
        for d in [5,10,20,40]:rows.append({'N':n*1e6,'D':d*1e6,'loss':1.5+2*n**(-.4)+3*d**(-.3)})
    r=fit(rows);assert abs(r['alpha']-.4)<.01 and abs(r['beta']-.3)<.01
    assert optimum(r,1e18)['N']>0
