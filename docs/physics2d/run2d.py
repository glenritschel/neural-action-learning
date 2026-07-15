import numpy as np, time
from physics2d import (make_grid, free_particle, harmonic, dp_min_action,
                       astar_min_action, energy_and_residual, analytic_axis)

def rms_to_analytic(sys, xs, path_ij, start_ij, goal_ij, T, N_t):
    ts=np.linspace(0,T,N_t)
    ax=analytic_axis(sys, xs[start_ij[0]], xs[goal_ij[0]], T, ts)
    ay=analytic_axis(sys, xs[start_ij[1]], xs[goal_ij[1]], T, ts)
    q=np.array([[xs[i],xs[j]] for i,j in path_ij])
    return float(np.sqrt(np.mean((q[:,0]-ax)**2+(q[:,1]-ay)**2)))

G=25; xs=make_grid(G, half=1.2); vmax=2
w=np.pi; T=0.3; N_t=16; dt=T/(N_t-1)   # omega*T=0.3pi, minimising regime
start_ij=(4,4)   # x=-0.8,y=-0.8
goals=[(20,20),(20,4),(12,20),(18,10),(6,16)]   # a few reachable goals
print(f"grid {G}x{G}, N_t={N_t}, states={G*G*N_t}, moves={(2*vmax+1)**2}, omega*T={w*T/np.pi:.2f}pi\n")

for sysf,label in [(free_particle(1.0),"FREE PARTICLE"),(harmonic(1.0,w),"HARMONIC OSC")]:
    print(f"==================== {label} ====================")
    dpE=[]; asE=[]; ratios=[]; head=[]; rmss=[]; optok=[]
    for g in goals:
        sdp,pdp,edp = dp_min_action(sysf,xs,N_t,dt,start_ij,g,vmax)
        sas,pas,eas = astar_min_action(sysf,xs,N_t,dt,start_ij,g,vmax)
        dpE.append(edp); asE.append(eas); ratios.append(edp/max(eas,1))
        head.append(eas/N_t)                 # expansions per optimal-path node (1.0 = perfect heuristic)
        rmss.append(rms_to_analytic(sysf,xs,pas,start_ij,g,T,N_t))
        optok.append(abs(sdp-sas)<1e-6)
        print(f"  goal {str(g):9s}  S={sas:8.3f}  DP_exp={edp:5d}  A*_exp={eas:5d}  DP/A*={edp/max(eas,1):5.1f}x  A*/path={eas/N_t:5.1f}  RMS_analytic={rmss[-1]:.3f}  opt={optok[-1]}")
    print(f"  ---- means: DP_exp={np.mean(dpE):.0f}  A*_exp={np.mean(asE):.0f}  DP/A*={np.mean(ratios):.1f}x  A*/path(headroom)={np.mean(head):.1f}x  RMS={np.mean(rmss):.3f}  A*=DP optimal: {all(optok)}")
    print()
