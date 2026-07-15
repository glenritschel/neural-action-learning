import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from discrete_least_action import (free_particle, harmonic, dp_min_action,
    analytic_path, discrete_action, energy_along, del_residual)

plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.3,"figure.dpi":150})
C_DP="#38a169"; C_AN="#2b6cb0"; C_E="#dd6b20"; C_BAD="#c53030"
OUT="/sessions/sweet-confident-ritchie/mnt/outputs/physics"
def rms(a,b): return float(np.sqrt(np.mean((a-b)**2)))

# ===================================================== EXP 1: free particle
sys=free_particle(1.0); qa,qb,T,Nt=-1.0,1.0,1.0,21
dt=T/(Nt-1); ts=np.linspace(0,T,Nt); qgrid=np.linspace(-1.5,1.5,301)
dp,S_dp,ne=dp_min_action(sys,qgrid,Nt,dt,qa,qb); an=analytic_path(sys,qa,qb,T,ts)
E=energy_along(sys,dp,dt)
print("=== EXP1 free particle ===")
print(f"  RMS(DP,analytic)={rms(dp,an):.3e}  S_dp={S_dp:.5f}  S_an={discrete_action(sys,an,dt):.5f}  E spread={E.max()-E.min():.2e}")
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
ax[0].plot(ts,an,"-",color=C_AN,lw=2,label="analytic (straight line)")
ax[0].plot(ts,dp,"o",color=C_DP,ms=5,label="DP min-action path")
ax[0].set_xlabel("t"); ax[0].set_ylabel("q"); ax[0].set_title("Free particle: DP recovers the classical path"); ax[0].legend(fontsize=8)
ax[1].plot(ts[:-1]+dt/2,E,"-o",color=C_E,ms=4); ax[1].set_ylim(E.mean()-1,E.mean()+1)
ax[1].set_xlabel("t"); ax[1].set_ylabel("energy E"); ax[1].set_title("Energy along DP path is conserved")
plt.tight_layout(); plt.savefig(f"{OUT}/phys_fig1_free_particle.png"); plt.close()

# ===================================================== EXP 2: SHO short T (clean minimising regime)
w=np.pi; P=2*np.pi/w; sys=harmonic(1.0,w); qa,qb=1.0,0.5
T=0.3*P; Nt=25; dt=T/(Nt-1); ts=np.linspace(0,T,Nt); qgrid=np.linspace(-1.5,1.5,201)
dp,S_dp,ne=dp_min_action(sys,qgrid,Nt,dt,qa,qb); an=analytic_path(sys,qa,qb,T,ts)
E_dp=energy_along(sys,dp,dt); E_an=energy_along(sys,an,dt)
print("\n=== EXP2 SHO short T (omega*T=0.6pi, minimising regime) ===")
print(f"  RMS(DP,analytic)={rms(dp,an):.3e}  S_dp={S_dp:.4f}  S_an={discrete_action(sys,an,dt):.4f}  E_an spread={E_an.max()-E_an.min():.2e}")
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
tf=np.linspace(0,T,300); ax[0].plot(tf,analytic_path(sys,qa,qb,T,tf),"-",color=C_AN,lw=2,label="analytic sinusoid")
ax[0].plot(ts,dp,"o",color=C_DP,ms=5,label="DP min-action path")
ax[0].set_xlabel("t"); ax[0].set_ylabel("q"); ax[0].set_title("SHO, short T (omega*T=0.6π): DP tracks classical"); ax[0].legend(fontsize=8)
ax[1].plot(ts[:-1]+dt/2,E_an,"-",color=C_AN,lw=2,label="analytic (constant)")
ax[1].plot(ts[:-1]+dt/2,E_dp,"-o",color=C_E,ms=4,label="DP path")
ax[1].set_xlabel("t"); ax[1].set_ylabel("energy E"); ax[1].set_title("Energy stays bounded / conserved"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/phys_fig2_sho_short.png"); plt.close()

# ===================================================== EXP 3: regime scan RMS vs omega*T
print("\n=== EXP3 regime scan: RMS(DP, analytic) vs omega*T ===")
fracs=[0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.55,0.6,0.65,0.7,0.75,0.8,0.85]
xs=[]; ys=[]
for fr in fracs:
    T=fr*P; Nt=25; dt=T/(Nt-1); ts=np.linspace(0,T,Nt)
    dp,_,_=dp_min_action(sys,qgrid,Nt,dt,qa,qb); an=analytic_path(sys,qa,qb,T,ts)
    xs.append(w*T/np.pi); ys.append(rms(dp,an))
    print(f"  omega*T={w*T/np.pi:.2f}pi  RMS={rms(dp,an):.3e}")
fig,ax=plt.subplots(figsize=(7,4.2))
ax.semilogy(xs,ys,"o-",color=C_DP)
ax.axvline(1.0,color=C_BAD,ls="--",lw=1.5,label="conjugate point (omega*T=π)")
ax.axvspan(0,0.65,color="#e6fffa",label="clean minimising regime")
ax.set_xlabel("omega*T / π"); ax.set_ylabel("RMS(DP path, classical path)")
ax.set_title("Least = stationary only below the conjugate point")
ax.legend(fontsize=8,loc="lower right")
plt.tight_layout(); plt.savefig(f"{OUT}/phys_fig3_regime.png"); plt.close()

# ===================================================== EXP 4: SHO long T (saddle) visual
T=0.75*P; Nt=41; dt=T/(Nt-1); ts=np.linspace(0,T,Nt)
dp,S_dp,ne=dp_min_action(sys,qgrid,Nt,dt,qa,qb); an=analytic_path(sys,qa,qb,T,ts)
print("\n=== EXP4 SHO long T (omega*T=1.5pi, beyond conjugate point) ===")
print(f"  RMS(DP,analytic classical)={rms(dp,an):.3e}  S_dp={S_dp:.4f}  S_an(classical)={discrete_action(sys,an,dt):.4f}")
fig,ax=plt.subplots(figsize=(6.4,4.2))
tf=np.linspace(0,T,400); ax.plot(tf,analytic_path(sys,qa,qb,T,tf),"-",color=C_AN,lw=2.2,label="analytic classical path (a saddle here)")
ax.plot(ts,dp,"o-",color=C_BAD,ms=4,lw=1.2,label="DP min-action path (unphysical)")
ax.set_xlabel("t"); ax.set_ylabel("q"); ax.set_ylim(-1.65,1.65)
ax.set_title("SHO, long T (omega*T=1.5π): least ≠ stationary")
ax.legend(fontsize=8,loc="lower left")
plt.tight_layout(); plt.savefig(f"{OUT}/phys_fig4_sho_long_saddle.png"); plt.close()

# ===================================================== EXP 5: DEL (force) residual convergence
print("\n=== EXP5 force residual (m q'' + V') of analytic SHO, should be O(dt^2) ===")
Nts=[9,13,19,25,37,49,73,97]; xs=[]; ys=[]
Tc=0.3*P
for Nt in Nts:
    dt=Tc/(Nt-1); ts=np.linspace(0,Tc,Nt); an=analytic_path(sys,qa,qb,Tc,ts)
    force_res=np.max(np.abs(del_residual(sys,an,dt)/dt))   # divide by dt -> equation-of-motion residual
    xs.append(dt); ys.append(force_res); print(f"  N_t={Nt:3d} dt={dt:.4f} max|m q''+V'|={force_res:.3e}")
fig,ax=plt.subplots(figsize=(5.8,4))
ax.loglog(xs,ys,"o-",color=C_DP,label="max |m q'' + V'|")
ref=[ys[0]*(d/xs[0])**2 for d in xs]; ax.loglog(xs,ref,"--",color="#718096",label="O(dt$^2$) reference")
ax.set_xlabel("time step dt"); ax.set_ylabel("force residual"); ax.set_title("Discrete Lagrangian is 2nd-order consistent"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/phys_fig5_del_convergence.png"); plt.close()
print("\nDONE. figures in", OUT)
