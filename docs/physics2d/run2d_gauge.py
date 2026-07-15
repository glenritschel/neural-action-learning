"""The real fix: gauge-shift the Lagrangian by c >= Vmax so every edge cost is
non-negative. Adding a constant to L adds c*(total time) to EVERY fixed-endpoint,
fixed-step path equally -> the optimal path is UNCHANGED, but now the report's
standard non-negative-cost machinery (multiplicative bound, no thrashing) applies."""
import numpy as np, heapq
m=1.0
def grid(G,half=1.2): return np.linspace(-half,half,G)
def moves(v): return [(a,b) for a in range(-v,v+1) for b in range(-v,v+1)]
def V_dw(x,y,a=8.0,d=0.8): return a*(((x*x-d*d)**2)+((y*y-d*d)**2))
def V_cp(x,y,w=np.pi,lam=10.0): return 0.5*w*w*(x*x+y*y)+lam*x*x*y*y
def ec(Vf,xs,i,j,i2,j2,dt,shift):   # shifted edge cost = dt*(T - V + shift)
    x1,y1,x2,y2=xs[i],xs[j],xs[i2],xs[j2]
    return dt*(0.5*m*((x2-x1)**2+(y2-y1)**2)/dt**2 - Vf(0.5*(x1+x2),0.5*(y1+y2)) + shift)

def dp_fwd(Vf,xs,N,dt,s,g,v,shift):
    G=len(xs); INF=np.inf; cost=np.full((N,G,G),INF); cost[0,s[0],s[1]]=0; mv=moves(v); exp=0
    for k in range(N-1):
        fi,fj=np.where(np.isfinite(cost[k]))
        for i,j in zip(fi,fj):
            exp+=1; base=cost[k,i,j]
            for di,dj in mv:
                i2,j2=i+di,j+dj
                if 0<=i2<G and 0<=j2<G:
                    c=base+ec(Vf,xs,i,j,i2,j2,dt,shift)
                    if c<cost[k+1,i2,j2]: cost[k+1,i2,j2]=c
    return float(cost[N-1,g[0],g[1]]),exp

def dp_value(Vf,xs,N,dt,g,v,shift):
    G=len(xs); INF=np.inf; val=np.full((N,G,G),INF); val[N-1,g[0],g[1]]=0.0; mv=moves(v)
    for k in range(N-2,-1,-1):
        for i in range(G):
            for j in range(G):
                b=INF
                for di,dj in mv:
                    i2,j2=i+di,j+dj
                    if 0<=i2<G and 0<=j2<G and np.isfinite(val[k+1,i2,j2]):
                        c=ec(Vf,xs,i,j,i2,j2,dt,shift)+val[k+1,i2,j2]
                        if c<b: b=c
                val[k,i,j]=b
    return val

def focal_mult(Vf,xs,N,dt,s,g,v,h_adm,h_guide,W,shift):
    """Non-negative-cost focal: FOCAL={f_adm<=W*f_min}; expand min h_guide. cost<=W*C*."""
    G=len(xs); mv=moves(v); start=(s[0],s[1],0); goal=(g[0],g[1],N-1)
    bestg={start:0.0}; open_set={start}; exp=0; hA={}; hG={}
    def gA(st):
        if st not in hA: hA[st]=h_adm(*st)
        return hA[st]
    def gG(st):
        if st not in hG: hG[st]=h_guide(*st)
        return hG[st]
    while open_set:
        fmin=min(bestg[st]+gA(st) for st in open_set)
        thr=W*fmin
        focal=[st for st in open_set if bestg[st]+gA(st)<=thr]
        cur=min(focal,key=gG); open_set.discard(cur); exp+=1
        if cur==goal: return bestg[cur],exp
        i,j,k=cur
        if k==N-1: continue
        gc=bestg[cur]
        for di,dj in mv:
            i2,j2=i+di,j+dj
            if 0<=i2<G and 0<=j2<G:
                ns=(i2,j2,k+1); ng=gc+ec(Vf,xs,i,j,i2,j2,dt,shift)
                if ng<bestg.get(ns,np.inf): bestg[ns]=ng; open_set.add(ns)
    return np.inf,exp

G=17; xs=grid(G); v=2; N=12; T=0.3; dt=T/(N-1); s=(3,3); goals=[(13,13),(13,4),(6,14)]
Gc=9; xsc=grid(Gc); vc=1; f2c=np.array([np.argmin(np.abs(xsc-x)) for x in xs])
for name,Vf in [("DOUBLE WELL",V_dw),("COUPLED",V_cp)]:
    Vg=np.array([[Vf(xs[i],xs[j]) for j in range(G)] for i in range(G)]); Vmax=Vg.max()
    shift=Vmax   # makes -V+shift >= 0
    print(f"==== {name} (shift c=Vmax={Vmax:.2f}) ====")
    # confirm optimal path unchanged: shifted C* should equal unshifted C* + shift*(N-1)*dt
    c_un,_=dp_fwd(Vf,xs,N,dt,s,goals[0],v,0.0)
    c_sh,_=dp_fwd(Vf,xs,N,dt,s,goals[0],v,shift)
    print(f"   gauge check: C*_shift - shift*T = {c_sh - shift*(N-1)*dt:+.4f}  vs  C*_unshift = {c_un:+.4f}  (match: {abs((c_sh-shift*(N-1)*dt)-c_un)<1e-6})")
    for gname in ["admissible-A*(W=1)","coarse-guided(W=1.5)"]:
        exps=[]; subs=[]; oks=[]
        for g in goals:
            gx,gy=xs[g[0]],xs[g[1]]
            def h_adm(i,j,k):
                tr=(N-1-k)*dt
                if tr<=0: return 0.0
                return 0.5*m*((gx-xs[i])**2+(gy-xs[j])**2)/tr + 0.0  # kinetic LB (>=0, since -V+shift>=0)
            Cst,edp=dp_fwd(Vf,xs,N,dt,s,g,v,shift)
            if gname.startswith("admissible"):
                c,e=focal_mult(Vf,xs,N,dt,s,g,v,h_adm,h_adm,1.0,shift)
            else:
                val=dp_value(Vf,xsc,N,dt,(int(f2c[g[0]]),int(f2c[g[1]])),vc,shift)
                def h_c(i,j,k,val=val):
                    x=val[k,f2c[i],f2c[j]]; return x if np.isfinite(x) else 1e9
                c,e=focal_mult(Vf,xs,N,dt,s,g,v,h_adm,h_c,1.5,shift)
                oks.append(c<=1.5*Cst+1e-6)
            exps.append(e); subs.append((c-Cst))
        extra="" if gname.startswith("admissible") else f"  realized_subopt={np.mean(subs):+.3f}  bound(cost<=1.5C*)_ok={all(oks)}"
        print(f"   {gname:22s}: mean_exp={np.mean(exps):6.1f}{extra}")
    print()
