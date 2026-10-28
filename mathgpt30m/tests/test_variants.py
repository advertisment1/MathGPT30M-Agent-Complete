import unittest
import torch
from mathgpt.model import Config,GPT

class TestVariants(unittest.TestCase):
    def test_all_interfaces(self):
        torch.manual_seed(42)
        for kv in [1,2,4]:
            for position,norm,ffn,tie in [('rope','rms','swiglu',True),('nope','layer','gelu',False)]:
                cfg=Config(vocab_size=96,dim=32,layers=2,heads=4,kv_heads=kv,hidden=64,max_seq=32,position=position,norm_type=norm,ffn_type=ffn,tie_embeddings=tie)
                model=GPT(cfg)
                x=torch.randint(0,96,(2,6))
                logits,loss,caches=model(x,x)
                self.assertEqual(tuple(logits.shape),(2,6,96))
                self.assertEqual(len(caches),2)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                model.eval()
                with torch.no_grad():
                    full,_,_=model(x)
                    _,_,cache=model(x[:,:5])
                    step,_,_=model(x[:,5:],caches=cache)
                self.assertTrue(torch.allclose(full[:,-1],step[:,-1],atol=1e-4,rtol=1e-4))

if __name__=='__main__':unittest.main()
