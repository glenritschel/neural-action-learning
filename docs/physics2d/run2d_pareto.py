"""Additive-bound focal search (cost <= C* + eps) for a signed action, and a
speed/quality Pareto over cheap guidance heuristics. Pure numpy."""
import numpy as np, heapq, time
m=1.0
def grid(G,half=1.2): return np.linspace(-half,half,G)
def moves(v): return [(a,b) for a in range(-v,v+1) for b in range(-v,v+1)]
def V_harm(x,y,w=np.pi): return 0.5*w*w*(x*x+y*y)
def V_dw(x,y,a=8.0,d=0.8): return a*(((x*x-d*d)**2)+((y*y-d*d)**2))
def V_cp(x,y,w=np.pi,lam=10.0): return 0.5*w*w*(x*x+y*y)+lam*x*x*y*y
def ec(Vf,xs,i,j,i2,j2,dt):
    x1,y1,x2,y2=xs[i],xs[j],xs[i2],xs[j2]
    return dt*(0.5*m*((x2-x1)**2+(y2-y1)**2)/dt**2 - Vf(0.5*(x1+x2),0.5*(y1+y2)))

def dp_fwd(Vf,xs,N,dt,s,g,v):
    G=len(xs); INF=np.inf; cost=np.full((N,G,G),INF); cost[0,s[0],s[1]]=0; mv=moves(v)
    for k in range(N-1):
        fi,fj=np.where(np.isfinite(cost[k]))
        for i,j in zip(fi,fj):
            base=cost[k,i,j]
            for di,dj in mv:
                i2,j2=i+di,j+dj
                if 0<=i2<G and 0<=j2<G:
                    c=base+ec(Vf,xs,i,j,i2,j2,dt)
                    if c<cost[k+1,i2,j2]: cost[k+1,i2,j2]=c
    return float(cost[N-1,g[0],g[1]])

def dp_value(Vf,xs,N,dt,g,v):
    G=len(xs); INF=np.inf; val=np.full((N,G,G),INF); val[N-1,g[0],g[1]]=0.0; mv=moves(v)
    for k in range(N-2,-1,-1):
        for i in range(G):
            for j in range(G):
                b=INF
                for di,dj in mv:
                    i2,j2=i+di,j+dj
                    if 0<=i2<G and 0<=j2<G and np.isfinite(val[k+1,i2,j2]):
                        c=ec(Vf,xs,i,j,i2,j2,dt)+val[k+1,i2,j2]
                        if c<b: b=c
                val[k,i,j]=b
    return val

def focal_additive(Vf,xs,N,dt,s,g,v,h_adm,h_guide,eps):
    """OPEN by admissible f_adm=g+h_adm; FOCAL={f_adm<=f_min+eps}; expand min h_guide.
    Guarantees returned cost <= C* + eps for ANY sign of cost (time-layered DAG)."""
    G=len(xs); mv=moves(v); start=(s[0],s[1],0); goal=(g[0],g[1],N-1)
    bestg={start:0.0}; open_set={start}; closed=set(); exp=0
    hA={}; hG={}
    def gethA(st):
        if st not in hA: hA[st]=h_adm(*st)
        return hA[st]
    def gethG(st):
        if st not in hG: hG[st]=h_guide(*st)
        return hG[st]
    while open_set:
        f_min=min(bestg[st]+gethA(st) for st in open_set)
        thr=f_min+eps
        focal=[st for st in open_set if bestg[st]+gethA(st)<=thr]
        cur=min(focal, key=gethG)
        open_set.discard(cur); closed.add(cur); exp+=1
        if cur==goal: return bestg[cur],exp
        i,j,k=cur
        if k==N-1: continue
        gc=bestg[cur]
        for di,dj in mv:
            i2,j2=i+di,j+dj
            if 0<=i2<G and 0<=j2<G:
                ns=(i2,j2,k+1); ng=gc+ec(Vf,xs,i,j,i2,j2,dt)
                if ng<bestg.get(ns,np.inf):
                    bestg[ns]=ng
                    open_set.add(ns); closed.discard(ns)
    return np.inf,exp

# ---- setup (kept small so the O(n) focal sweep is fast) ----
G=17; xs=grid(G); v=2; N=12; T=0.3; dt=T/(N-1); w=np.pi; s=(3,3)
goals=[(13,13),(13,4),(6,14)]
Gc=9; xsc=grid(Gc); vc=1; f2c=np.array([np.argmin(np.abs(xsc-x)) for x in xs])
eps_grid=[0.0,0.5,1.0,2.0,4.0,8.0]
print(f"fine {G}x{G} v={v} N_t={N}; coarse {Gc}x{Gc}; eps={eps_grid}\n")

results={}
for name,Vf in [("HARMONIC",V_harm),("DOUBLE WELL",V_dw),("COUPLED",V_cp)]:
    Vg=np.array([[Vf(xs[i],xs[j]) for j in range(G)] for i in range(G)]); Vmax=Vg.max()
    Cstar={}; val={}
    for g in goals:
        Cstar[g]=dp_fwd(Vf,xs,N,dt,s,g,v)
        val[g]=dp_value(Vf,xsc,N,dt,(int(f2c[g[0]]),int(f2c[g[1]])),vc)
    def make_hadm(g):
        gx,gy=xs[g[0]],xs[g[1]]
        def h(i,j,k):
            tr=(N-1-k)*dt
            if tr<=0: return 0.0
            return 0.5*m*((gx-xs[i])**2+(gy-xs[j])**2)/tr - Vmax*tr
        return h
    def make_hcoarse(g):
        vv=val[g]
        def h(i,j,k):
            x=vv[k,f2c[i],f2c[j]]; return x if np.isfinite(x) else 1e6
        return h
    def make_hclass(g):
        gx,gy=xs[g[0]],xs[g[1]]
        def h(i,j,k):
            tr=(N-1-k)*dt
            if tr<=0: return 0.0
            sn=np.sin(w*tr); cs=np.cos(w*tr)
            def Sax(qa,qb): return (m*w/(2*sn))*((qa*qa+qb*qb)*cs-2*qa*qb)
            return Sax(xs[i],gx)+Sax(xs[j],gy)
        return h
    guides=[("coarse-DP",make_hcoarse)]
    if name=="HARMONIC": guides.append(("classical",make_hclass))
    guides.append(("admissible-as-guide",make_hadm))
    print(f"==== {name} ====")
    for gname,mk in guides:
        pts=[]
        for eps in eps_grid:
            exps=[]; subs=[]; viol=0
            for g in goals:
                c,e=focal_additive(Vf,xs,N,dt,s,g,v,make_hadm(g),mk(g),eps)
                exps.append(e); subs.append(c-Cstar[g])
                if c>Cstar[g]+eps+1e-6: viol+=1
            pts.append((np.mean(exps),np.mean(subs)))
            print(f"   {gname:20s} eps={eps:4.1f}: exp={np.mean(exps):6.1f}  realized_subopt={np.mean(subs):+.3f}  bound_ok={viol==0}")
        results[(name,gname)]=pts
    print()

import json
json.dump({f"{k[0]}|{k[1]}":v for k,v in results.items()}, open("pareto_pts.json","w"))
print("saved pareto_pts.json")
