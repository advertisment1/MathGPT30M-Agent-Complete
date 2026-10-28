import json
from mathgpt.model import Config,GPT
from mathgpt.experiment import choose
from scripts.scale_to_30m import count,find
from scripts.greedy_search import BASE,ROUNDS

def test_exact_parameter_formula():
    for cfg in [Config(dim=128,layers=2,heads=4,kv_heads=1,hidden=352),
                Config(dim=128,layers=2,heads=4,kv_heads=4,hidden=352,ffn_type='gelu',norm_type='layer',tie_embeddings=False)]:
        assert count(cfg)==sum(p.numel() for p in GPT(cfg).parameters())

def test_30m_search():
    _,n,c=find(BASE)
    assert abs(n-30_000_000)/30_000_000 < .02
    assert count(c)==n

def test_greedy_rounds():
    assert len(ROUNDS)==6
    assert choose({'val_loss':2.5},{'val_loss':2.48})
    assert not choose({'val_loss':2.5},{'val_loss':2.499})
