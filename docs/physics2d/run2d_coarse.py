"""Coarse-DP control: is a cheap precomputed value function (DP on a coarse grid,
used as a heuristic on the fine grid) already enough to collapse the headroom?
This is the strongest NON-learned competitor a learned cost-to-go must beat."""
import numpy as np, heapq
m=1.0
def grid(G,half=1.2): return np.linspace(-half,half,G)
def moves(v): return [(a,b) for a in range(-v,v+1) for b in range(-v,v+1)]
def V_harm(x,y,w=np.pi): return 0.5*w*w*(x*x+y*y)
def V_dw(x,y,a=8.0,d=0.8): return a*(((x*x-d*d)**2)+((y*y-d*d)**2))
def V_cp(x,y,w=np.pi,lam=10.0): return 0.5*w*w*(x*x+y*y)+lam*x*x*y*y
def ec(Vf,xs,i,j,i2,j2,dt):
    x1,y1,x2,y2=xs[i],xs[j],xs[i2],xs[j2]
    return dt*(0.5*m*((x2-x1)**2+(y2-y1)**2)/dt**2 - Vf(0.5*(x1+x2),0.5*(y1+y2)))

def dp_fwd(Vf,xs,N,dt,s,g,v):   # forward min action from start (ground truth) + expansions
    G=len(xs); INF=np.inf; cost=np.full((N,G,G),INF); back=np.full((N,G,G,2),-1,int)
    cost[0,s[0],s[1]]=0; mv=moves(v); exp=0
    for k in range(N-1):
        fi,fj=np.where(np.isfinite(cost[k]))
        for i,j in zip(fi,fj):
            exp+=1; base=cost[k,i,j]
            for di,dj in mv:
                i2,j2=i+di,j+dj
                if 0<=i2<G and 0<=j2<G:
                    c=base+ec(Vf,xs,i,j,i2,j2,dt)
                    if c<cost[k+1,i2,j2]: cost[k+1,i2,j2]=c; back[k+1,i2,j2]=(i,j)
    gi,gj=g; path=[(gi,gj)]; ci,cj=gi,gj
    for k in range(N-1,0,-1): pi,pj=back[k,ci,cj]; path.append((pi,pj)); ci,cj=pi,pj
    return float(cost[N-1,gi,gj]),path[::-1],exp

def dp_value_to_goal(Vf,xs,N,dt,g,v):   # backward cost-to-go value function
    G=len(xs); INF=np.inf; val=np.full((N,G,G),INF); val[N-1,g[0],g[1]]=0.0; mv=moves(v)
    for k in range(N-2,-1,-1):
        for i in range(G):
            for j in range(G):
                best=INF
                for di,dj in mv:
                    i2,j2=i+di,j+dj
                    if 0<=i2<G and 0<=j2<G and np.isfinite(val[k+1,i2,j2]):
                        c=ec(Vf,xs,i,j,i2,j2,dt)+val[k+1,i2,j2]
                        if c<best: best=c
                val[k,i,j]=best
    return val

def astar(Vf,xs,N,dt,s,g,v,hf):
    G=len(xs); mv=moves(v); tie=0; start=(s[0],s[1],0)
    oh=[(hf(*start),tie,0.0,start,None)]; tie+=1
    best={start:0.0}; par={start:None}; closed=set(); exp=0; goal=(g[0],g[1],N-1)
    while oh:
        f,_,gg,st,p=heapq.heappop(oh)
        if st in closed or gg>best.get(st,np.inf): continue
        closed.add(st); par[st]=p; exp+=1
        i,j,k=st
        if st==goal:
            path=[]; c=st
            while c is not None: path.append((c[0],c[1])); c=par[c]
            return gg,path[::-1],exp
        if k==N-1: continue
        for di,dj in mv:
            i2,j2=i+di,j+dj
            if 0<=i2<G and 0<=j2<G:
                ns=(i2,j2,k+1); ng=gg+ec(Vf,xs,i,j,i2,j2,dt)
                if ng<best.get(ns,np.inf):
                    best[ns]=ng; heapq.heappush(oh,(ng+hf(i2,j2,k+1),tie,ng,ns,st)); tie+=1
    return np.inf,[],exp

# fine + coarse setup
Gf=25; xs=grid(Gf); v=2; N=16; T=0.3; dt=T/(N-1); w=np.pi; s=(4,4)
goals=[(20,20),(20,4),(12,20),(18,10),(6,16)]
Gc=13; xsc=grid(Gc); vc=1                         # coarse: 13x13, matched physical velocity
f2c=np.array([np.argmin(np.abs(xsc-x)) for x in xs])   # fine index -> coarse index
Vmax_global={}
print(f"fine {Gf}x{Gf} (v={v}), coarse {Gc}x{Gc} (v={vc}), N_t={N}, T={T}\n")

for name,Vf in [("HARMONIC",V_harm),("DOUBLE WELL",V_dw),("COUPLED x2y2",V_cp)]:
    Vg=np.array([[Vf(xs[i],xs[j]) for j in range(Gf)] for i in range(Gf)]); Vmax=Vg.max()
    lazy_e=[]; coarse_e=[]; gap=[]; dpe=[]; pathlen=N
    for g in goals:
        sdp,pdp,edp=dp_fwd(Vf,xs,N,dt,s,g,v); dpe.append(edp)
        def h_lazy(i,j,k,Vmax=Vmax):
            tr=(N-1-k)*dt
            if tr<=0: return 0.0
            dx=xs[g[0]]-xs[i]; dy=xs[g[1]]-xs[j]
            return 0.5*m*(dx*dx+dy*dy)/tr - Vmax*tr
        _,_,el=astar(Vf,xs,N,dt,s,g,v,h_lazy); lazy_e.append(el)
        # coarse value function to the coarse image of the goal
        gc=(int(f2c[g[0]]),int(f2c[g[1]]))
        val=dp_value_to_goal(Vf,xsc,N,dt,gc,vc)
        def h_coarse(i,j,k,val=val):
            vv=val[k,f2c[i],f2c[j]]
            return vv if np.isfinite(vv) else 1e6
        sc,pc,ec2=astar(Vf,xs,N,dt,s,g,v,h_coarse); coarse_e.append(ec2)
        gap.append(0.0 if sdp==0 else (sc-sdp)/abs(sdp)*100.0)
    import numpy as _np
    print(f"==== {name} ====  DP_exp={_np.mean(dpe):.0f}")
    print(f"   A* lazy-admissible : exp={_np.mean(lazy_e):7.0f}  A*/path={_np.mean(lazy_e)/pathlen:6.1f}x  (optimal)")
    print(f"   A* coarse-DP heur  : exp={_np.mean(coarse_e):7.0f}  A*/path={_np.mean(coarse_e)/pathlen:6.1f}x  cost gap vs optimal={_np.mean(gap):+.2f}%")
    print()
