"""
Discrete least-action 1D validation (free particle + harmonic oscillator).

Pure numpy. A standalone, honest first step for the physics track, kept SEPARATE
from the grid-planner cost model. It does three things:

  1. Builds a discrete Lagrangian L_d on a (position x time) trellis with fixed
     endpoints (a two-point boundary value problem), using the midpoint rule.
  2. Finds the exact minimum-discrete-action path by dynamic programming (Viterbi
     over the layered DAG) and compares it to the analytic classical trajectory.
  3. Reports energy behaviour and the discrete Euler-Lagrange (DEL) residual.

Crucial honesty point, made explicit by the experiments below:
  A least-cost solver MINIMIZES the action. Classical mechanics requires a
  STATIONARY action, which is a minimum ONLY for short time intervals. For the
  harmonic oscillator the classical path minimizes the action only while the
  total time T is below half a period (omega*T < pi); past the first conjugate
  point the classical path is a SADDLE, and a minimizer will not find it.
"""
import numpy as np

# ----------------------------------------------------------------------------
# Physical systems: potential V(q) and its derivative V'(q). L = 0.5 m v^2 - V.
# ----------------------------------------------------------------------------
def free_particle(m=1.0):
    return dict(name="free particle", m=m, V=lambda q: 0.0*q, dV=lambda q: 0.0*q, omega=0.0)

def harmonic(m=1.0, omega=2*np.pi):
    return dict(name="harmonic oscillator", m=m, omega=omega,
                V=lambda q: 0.5*m*omega**2*q**2, dV=lambda q: m*omega**2*q)

# ----------------------------------------------------------------------------
# Discrete Lagrangian on an edge (q_a at t_k) -> (q_b at t_{k+1}), midpoint rule.
#   L_d(a,b) = dt * [ 0.5 m ((b-a)/dt)^2 - V((a+b)/2) ]
# ----------------------------------------------------------------------------
def Ld(sys, a, b, dt):
    v = (b - a) / dt
    return dt * (0.5*sys["m"]*v**2 - sys["V"](0.5*(a+b)))

def discrete_action(sys, path, dt):
    return float(sum(Ld(sys, path[k], path[k+1], dt) for k in range(len(path)-1)))

# ----------------------------------------------------------------------------
# Exact minimiser: DP / Viterbi over the position x time trellis, fixed ends.
# ----------------------------------------------------------------------------
def dp_min_action(sys, qgrid, N_t, dt, q_start, q_goal):
    Nq = len(qgrid)
    si = int(np.argmin(np.abs(qgrid - q_start)))
    gi = int(np.argmin(np.abs(qgrid - q_goal)))
    INF = np.inf
    cost = np.full((N_t, Nq), INF)
    back = np.full((N_t, Nq), -1, dtype=int)
    cost[0, si] = 0.0
    # precompute edge cost matrix E[i,j] = Ld(q_i, q_j)
    A = qgrid[:, None]; B = qgrid[None, :]
    v = (B - A) / dt
    E = dt * (0.5*sys["m"]*v**2 - sys["V"](0.5*(A+B)))
    nodes_expanded = 0
    for k in range(N_t - 1):
        active = np.where(np.isfinite(cost[k]))[0]
        for i in active:
            nodes_expanded += 1
            cand = cost[k, i] + E[i, :]
            improve = cand < cost[k+1]
            cost[k+1, improve] = cand[improve]
            back[k+1, improve] = i
    # force goal at final layer
    total = cost[N_t-1, gi]
    path_idx = [gi]
    j = gi
    for k in range(N_t-1, 0, -1):
        j = back[k, j]
        path_idx.append(j)
    path_idx.reverse()
    path = qgrid[np.array(path_idx)]
    return path, float(total), nodes_expanded

# ----------------------------------------------------------------------------
# Analytic classical solutions of the two-point BVP q(0)=qa, q(T)=qb.
# ----------------------------------------------------------------------------
def analytic_path(sys, qa, qb, T, ts):
    if sys["omega"] == 0.0:          # free particle: straight line
        return qa + (qb - qa) * (ts / T)
    w = sys["omega"]
    s = np.sin(w*T)
    A = qa
    B = (qb - qa*np.cos(w*T)) / s    # valid when sin(wT) != 0
    return A*np.cos(w*ts) + B*np.sin(w*ts)

def energy_along(sys, q, dt):
    # midpoint velocity + midpoint potential, per interval -> E_k
    v = (q[1:] - q[:-1]) / dt
    qmid = 0.5*(q[1:] + q[:-1])
    return 0.5*sys["m"]*v**2 + sys["V"](qmid)

def del_residual(sys, q, dt):
    # discrete Euler-Lagrange residual at interior nodes (should be ~0 for a
    # stationary path; O(dt^2) for a sampled analytic solution).
    m = sys["m"]
    res = []
    for k in range(1, len(q)-1):
        term = -m*(q[k+1] - 2*q[k] + q[k-1])/dt \
               - 0.5*dt*(sys["dV"](0.5*(q[k-1]+q[k])) + sys["dV"](0.5*(q[k]+q[k+1])))
        res.append(term)
    return np.array(res)
