"""
2D discrete least-action baseline: exact DP vs admissible A*.
Measures whether there is HEADROOM for a learned cost-to-go heuristic BEFORE
anyone trains a network. Pure numpy. Two-point BVP, fixed endpoints, fixed T.

Config space q=(x,y). State for search = (i, j, k) = (x-index, y-index, time layer).
Edge cost is the discrete Lagrangian (midpoint rule):
    L_d = dt * [ 0.5 m |(q2-q1)/dt|^2  -  V((q1+q2)/2) ]
Pure L = T - V, so the edge cost depends only on the two positions -> (i,j,k) is Markov.
"""
import numpy as np, heapq

def make_grid(G, half=1.2):
    xs = np.linspace(-half, half, G)
    return xs

def V_iso(sys, x, y):
    if sys["omega"] == 0.0: return 0.0
    return 0.5*sys["m"]*sys["omega"]**2*(x*x + y*y)

def free_particle(m=1.0):   return dict(name="free particle", m=m, omega=0.0)
def harmonic(m=1.0, omega=np.pi): return dict(name="harmonic oscillator", m=m, omega=omega)

def moves(vmax):
    return [(di,dj) for di in range(-vmax,vmax+1) for dj in range(-vmax,vmax+1)]

def edge_cost(sys, xs, i, j, i2, j2, dt):
    x1,y1,x2,y2 = xs[i],xs[j],xs[i2],xs[j2]
    v2 = ((x2-x1)**2 + (y2-y1)**2)/dt**2
    T = 0.5*sys["m"]*v2
    V = V_iso(sys, 0.5*(x1+x2), 0.5*(y1+y2))
    return dt*(T - V)

# ------------------------------------------------------------------ exact DP
def dp_min_action(sys, xs, N_t, dt, start_ij, goal_ij, vmax):
    G = len(xs); INF = np.inf
    cost = np.full((N_t, G, G), INF); back = np.full((N_t, G, G, 2), -1, dtype=int)
    cost[0, start_ij[0], start_ij[1]] = 0.0
    mv = moves(vmax); expanded = 0
    for k in range(N_t-1):
        fi, fj = np.where(np.isfinite(cost[k]))
        for i, j in zip(fi, fj):
            expanded += 1
            base = cost[k, i, j]
            for di, dj in mv:
                i2, j2 = i+di, j+dj
                if 0<=i2<G and 0<=j2<G:
                    c = base + edge_cost(sys, xs, i, j, i2, j2, dt)
                    if c < cost[k+1, i2, j2]:
                        cost[k+1, i2, j2] = c; back[k+1, i2, j2] = (i, j)
    gi, gj = goal_ij
    total = cost[N_t-1, gi, gj]
    # backtrack
    path = [(gi, gj)]; ci, cj = gi, gj
    for k in range(N_t-1, 0, -1):
        pi, pj = back[k, ci, cj]; path.append((pi, pj)); ci, cj = pi, pj
    path.reverse()
    return float(total), path, expanded

# ------------------------------------------------------------ admissible A*
def h_admissible(sys, xs, i, j, k, goal_ij, N_t, dt, Rmax):
    t_rem = (N_t-1-k)*dt
    if t_rem <= 0: return 0.0
    dx = xs[goal_ij[0]]-xs[i]; dy = xs[goal_ij[1]]-xs[j]
    kin_lb = 0.5*sys["m"]*(dx*dx+dy*dy)/t_rem        # straight-line kinetic lower bound
    if sys["omega"] == 0.0:
        return kin_lb                                 # exact for the free particle
    Vmax = 0.5*sys["m"]*sys["omega"]**2*Rmax**2       # most reward per unit time
    return kin_lb - Vmax*t_rem                         # subtract max potential reward -> admissible

def astar_min_action(sys, xs, N_t, dt, start_ij, goal_ij, vmax):
    G = len(xs); Rmax = np.sqrt(2)*xs.max()
    mv = moves(vmax); tie = 0
    start = (start_ij[0], start_ij[1], 0)
    h0 = h_admissible(sys, xs, start[0], start[1], 0, goal_ij, N_t, dt, Rmax)
    openh = [(h0, tie, 0.0, start, None)]; tie += 1
    best = {start: 0.0}; parent = {start: None}; closed = set(); expanded = 0
    goal_state = (goal_ij[0], goal_ij[1], N_t-1)
    while openh:
        f, _, g, s, par = heapq.heappop(openh)
        if s in closed or g > best.get(s, np.inf): continue
        closed.add(s); parent[s] = par; expanded += 1
        i, j, k = s
        if s == goal_state:
            path=[]; cur=s
            while cur is not None: path.append((cur[0],cur[1])); cur=parent[cur]
            path.reverse(); return g, path, expanded
        if k == N_t-1: continue
        for di, dj in mv:
            i2, j2 = i+di, j+dj
            if 0<=i2<G and 0<=j2<G:
                ns = (i2, j2, k+1)
                ng = g + edge_cost(sys, xs, i, j, i2, j2, dt)
                if ng < best.get(ns, np.inf):
                    best[ns] = ng
                    h = h_admissible(sys, xs, i2, j2, k+1, goal_ij, N_t, dt, Rmax)
                    heapq.heappush(openh, (ng+h, tie, ng, ns, s)); tie += 1
    return np.inf, [], expanded

# ---------------------------------------------------------- physical fidelity
def energy_and_residual(sys, xs, path_ij, dt):
    q = np.array([[xs[i], xs[j]] for i, j in path_ij])
    v = (q[1:]-q[:-1])/dt
    qmid = 0.5*(q[1:]+q[:-1])
    E = 0.5*sys["m"]*np.sum(v**2, axis=1) + (0.0 if sys["omega"]==0 else 0.5*sys["m"]*sys["omega"]**2*np.sum(qmid**2,axis=1))
    return E

def analytic_axis(sys, a, b, T, ts):
    if sys["omega"]==0.0: return a + (b-a)*ts/T
    w=sys["omega"]; s=np.sin(w*T)
    return a*np.cos(w*ts) + (b-a*np.cos(w*T))/s*np.sin(w*ts)
