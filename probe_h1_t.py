"""Does the rho=0 total-energy constraint move the eta-threshold for certifying
K=1 insufficient?  Compares the polytope WITHOUT T (v3 as-run) vs WITH the
s(0)=T slab added.  Noiseless curve; sweep eta.  Pure numpy/scipy.
"""
import numpy as np
from scipy.optimize import linprog

d=49; Kmax=d; alpha=0.05
rhos = np.linspace(0.5,0.97,8)
a_true=np.zeros(Kmax+1); a_true[1],a_true[2],a_true[3]=1.0,0.35,0.04
T=a_true[1:].sum()
s=lambda r: sum(a_true[j]*(1-r**j) for j in range(1,Kmax+1))
s_grid=np.array([s(r) for r in rhos])

nvar=Kmax
W=np.array([[1-r**j for j in range(1,Kmax+1)] for r in rhos])
full=np.ones(nvar); tail1=np.zeros(nvar); tail1[1:]=1.0
obj=tail1-alpha*full
bounds=[(0,None)]*nvar
T_row=np.ones((1,nvar))  # sum_{j>=1} a_j  == s(0)

def minh1(eta, withT):
    A=np.vstack([W,-W]); b=np.concatenate([s_grid+eta,-(s_grid-eta)])
    if withT:
        A=np.vstack([A,T_row,-T_row]); b=np.concatenate([b,[T+eta],[-(T-eta)]])
    r=linprog(obj,A_ub=A,b_ub=b,bounds=bounds,method="highs")
    return r.fun

print(f"{'eta':>8} {'min h1 (no T)':>14} {'min h1 (+rho=0 T)':>18}")
for eta in [1e-3,3e-3,5e-3,8e-3,0.012,0.019,0.03,0.05]:
    a=minh1(eta,False); b=minh1(eta,True)
    print(f"{eta:>8.4f} {a:>+14.4f} {b:>+18.4f}")
print("\n(>0 => K=1 certified INSUFFICIENT. Rightmost column with T is the v3+rho=0 case.)")
print("Threshold = largest eta still giving +ve min h1.")