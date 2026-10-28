import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

def fit(rows):
    # L(N,D) = E + A/N^alpha + B/D^beta ; log parameters for positivity
    N=np.array([r['N'] for r in rows],float);D=np.array([r['D'] for r in rows],float);L=np.array([r['loss'] for r in rows],float)
    if len(rows)<8 or len(set(N))<3 or len(set(D))<3:raise ValueError('Need >=8 runs, >=3 unique model sizes and >=3 token budgets')
    def pred(z):
        e,a,b,alpha,beta=np.exp(z)
        return e+a*(N/1e6)**(-alpha)+b*(D/1e6)**(-beta)
    initial=np.log([max(.1,min(L)*.7),1.,1.,.3,.3])
    res=least_squares(lambda z:pred(z)-L,initial,max_nfev=20000,bounds=(np.log([1e-5,1e-6,1e-6,.01,.01]),np.log([100,1e5,1e5,2.,2.])))
    e,a,b,alpha,beta=np.exp(res.x)
    return {'E':e,'A':a,'B':b,'alpha':alpha,'beta':beta,'rmse':float(np.sqrt(np.mean((pred(res.x)-L)**2))),'success':bool(res.success),'units':'N and D measured in millions'}

def optimum(fit,compute_flops,min_n=1e6,max_n=1e9):
    # approximate transformer train FLOPs = 6ND; this omits attention quadratic and hardware efficiency
    n=np.geomspace(min_n,max_n,3000);d=compute_flops/(6*n)
    loss=fit['E']+fit['A']*(n/1e6)**(-fit['alpha'])+fit['B']*(d/1e6)**(-fit['beta'])
    i=np.argmin(loss);return {'N':float(n[i]),'D':float(d[i]),'predicted_loss':float(loss[i]),'approx_flops':compute_flops}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('jsonl');p.add_argument('--flops',type=float,default=1e18);a=p.parse_args()
    rows=[json.loads(x) for x in Path(a.jsonl).read_text().splitlines() if x.strip()]
    f=fit(rows);print(json.dumps({'fit':f,'compute_optimum':optimum(f,a.flops)},indent=2))
