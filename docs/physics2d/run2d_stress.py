"""Headroom stress test: does the DP->A* gap survive a SMARTER admissible heuristic,
and does it generalize beyond the isotropic oscillator? Pure numpy."""
import numpy as np, heapq

def make_grid(G, half=1.2): return np.linspace(-half,half,G)
def moves(vmax): return [(di,dj) for di in range(-vmax,vmax+1) for dj in range(-vmax,vmax+1)]
m=1.0

# ---------------- potentials ----------------
def V_harmonic(x,y,w=np.pi):        return 0.5*w*w*(x*x+y*y)
def V_doublewell(x,y,a=8.0,d=0.8):  return a*(((x*x-d*d)**2)+((y*y-d*d)**2))
def V_coupled(x,y,w=np.pi,lam=10.0):return 0.5*w*w*(x*x+y*y)+lam*(x*x)*(y*y)

def edge_cost(Vf, xs, i,j,i2,j2, dt):
    x1,y1,x2,y2=xs[i],xs[j],xs[i2],xs[j2]
    v2=((x2-x1)**2+(y2-y1)**2)/dt**2
    return dt*(0.5*m*v2 - Vf(0.5*(x1+x2),0.5*(y1+y2)))

def dp_min_action(Vf, xs, N_t, dt, s_ij, g_ij, vmax):
    G=len(xs); INF=np.inf
    cost=np.full((N_t,G,G),INF); back=np.full((N_t,G,G,2),-1,dtype=int)
    cost[0,s_ij[0],s_ij[1]]=0.0; mv=moves(vmax); exp=0
    for k in range(N_t-1):
        fi,fj=np.where(np.isfinite(cost[k]))
        for i,j in zip(fi,fj):
            exp+=1; base=cost[k,i,j]
            for di,dj in mv:
                i2,j2=i+di,j+dj
                if 0<=i2<G and 0<=j2<G:
                    c=base+edge_cost(Vf,xs,i,j,i2,j2,dt)
                    if c<cost[k+1,i2,j2]: cost[k+1,i2,j2]=c; back[k+1,i2,j2]=(i,j)
    gi,gj=g_ij; total=cost[N_t-1,gi,gj]
    path=[(gi,gj)]; ci,cj=gi,gj
    for k in range(N_t-1,0,-1):
        pi,pj=back[k,ci,cj]; path.append((pi,pj)); ci,cj=pi,pj
    path.reverse(); return float(total),path,exp

def astar(Vf, xs, N_t, dt, s_ij, g_ij, vmax, hfun):
    G=len(xs); mv=moves(vmax); tie=0
    start=(s_ij[0],s_ij[1],0)
    openh=[(hfun(start[0],start[1],0),tie,0.0,start,None)]; tie+=1
    best={start:0.0}; parent={start:None}; closed=set(); exp=0
    goal=(g_ij[0],g_ij[1],N_t-1)
    while openh:
        f,_,g,s,par=heapq.heappop(openh)
        if s in closed or g>best.get(s,np.inf): continue
        closed.add(s); parent[s]=par; exp+=1
        i,j,k=s
        if s==goal:
            path=[]; c=s
            while c is not None: path.append((c[0],c[1])); c=parent[c]
            path.reverse(); return g,path,exp
        if k==N_t-1: continue
        for di,dj in mv:
            i2,j2=i+di,j+dj
            if 0<=i2<G and 0<=j2<G:
                ns=(i2,j2,k+1); ng=g+edge_cost(Vf,xs,i,j,i2,j2,dt)
                if ng<best.get(ns,np.inf):
                    best[ns]=ng
                    heapq.heappush(openh,(ng+hfun(i2,j2,k+1),tie,ng,ns,s)); tie+=1
    return np.inf,[],exp

# ---------------- admissible heuristics (lower bounds on remaining action) ----------------
def make_heuristics(Vf, xs, N_t, dt, g_ij, vmax, w=None):
    G=len(xs); Vgrid=np.array([[Vf(xs[i],xs[j]) for j in range(G)] for i in range(G)])
    Vmax_global=Vgrid.max()
    gx,gy=xs[g_ij[0]],xs[g_ij[1]]
    def kin_lb(i,j,k):
        t_rem=(N_t-1-k)*dt
        if t_rem<=0: return 0.0,0.0
        dx=gx-xs[i]; dy=gy-xs[j]
        return 0.5*m*(dx*dx+dy*dy)/t_rem, t_rem
    def h_lazy(i,j,k):
        kl,tr=kin_lb(i,j,k)
        return kl - Vmax_global*tr
    def h_cone(i,j,k):
        kl,tr=kin_lb(i,j,k)
        if tr<=0: return 0.0
        r=vmax*(N_t-1-k)
        i0,i1=max(0,i-r),min(G,i+r+1); j0,j1=max(0,j-r),min(G,j+r+1)
        Vmax_box=Vgrid[i0:i1,j0:j1].max()   # V can't exceed this on any continuation
        return kl - Vmax_box*tr
    def h_classical(i,j,k):   # exact continuous SHO cost-to-go, per axis (oscillator only)
        t_rem=(N_t-1-k)*dt
        if t_rem<=0: return 0.0
        s=np.sin(w*t_rem); c=np.cos(w*t_rem)
        def Sax(qa,qb): return (m*w/(2*s))*((qa*qa+qb*qb)*c - 2*qa*qb)
        return Sax(xs[i],gx)+Sax(xs[j],gy)
    return dict(lazy=h_lazy, cone=h_cone, classical=h_classical)

# ---------------- run ----------------
G=25; xs=make_grid(G,1.2); vmax=2; N_t=16; T=0.3; dt=T/(N_t-1); w=np.pi
s_ij=(4,4); goals=[(20,20),(20,4),(12,20),(18,10),(6,16)]
print(f"grid {G}x{G} N_t={N_t} states={G*G*N_t} moves={(2*vmax+1)**2} T={T} (omega*T={w*T/np.pi:.2f}pi)\n")

configs=[("HARMONIC (separable, closed-form exists)", V_harmonic, ["lazy","cone","classical"]),
         ("DOUBLE WELL (anharmonic, no closed form)", V_doublewell, ["lazy","cone"]),
         ("COUPLED x^2 y^2 (non-separable)",           V_coupled,    ["lazy","cone"])]
for name,Vf,hkeys in configs:
    print(f"==================== {name} ====================")
    agg={hk:[] for hk in hkeys}; dpexp=[]; optok=True
    for g in goals:
        sdp,pdp,edp=dp_min_action(Vf,xs,N_t,dt,s_ij,g,vmax); dpexp.append(edp)
        H=make_heuristics(Vf,xs,N_t,dt,g,vmax,w=w)
        for hk in hkeys:
            sa,pa,ea=astar(Vf,xs,N_t,dt,s_ij,g,vmax,H[hk])
            agg[hk].append(ea); optok=optok and abs(sa-sdp)<1e-6
    dpm=np.mean(dpexp)
    print(f"  DP mean expansions={dpm:.0f}   (A* returns exact optimum: {optok})")
    for hk in hkeys:
        em=np.mean(agg[hk]); print(f"    A* [{hk:9s}]  mean_exp={em:7.0f}   A*/path={em/N_t:6.1f}x   DP/A*={dpm/max(em,1):5.1f}x")
    print()
