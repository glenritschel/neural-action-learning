# Physics track — 1D discrete least-action starter

A standalone, honest first step for the variational / stationary-action track,
kept **separate** from the grid-planner cost model. Pure numpy + matplotlib.

Run:
```
python run_experiment.py    # writes phys_fig1..5 and prints metrics
```

## What it does
Builds a discrete Lagrangian `L_d(q_k, q_{k+1}) = dt * [ 0.5 m v^2 - V((q_k+q_{k+1})/2) ]`
(midpoint rule) on a (position × time) trellis with fixed endpoints, finds the
**exact minimum-discrete-action path by dynamic programming**, and compares it to
the analytic classical trajectory, with energy and discrete-Euler–Lagrange checks.

## Findings (all reproduced by `run_experiment.py`)

1. **Free particle — exact.** DP recovers the classical straight line to machine
   precision (RMS ≈ 1e-16), action = 2.0 exactly, energy constant. Validates the
   discrete Lagrangian and the DP solver.

2. **Harmonic oscillator, short T (ω·T = 0.6π) — tracks.** DP follows the analytic
   sinusoid to RMS ≈ 5e-3 (grid-limited), energy conserved to ≈ 7e-3. This is the
   regime where the report's focal-search + learned-heuristic machinery applies
   directly, now with a physically meaningful cost.

3. **The least-vs-stationary boundary (the key result).** A least-cost solver
   *minimizes* the action; classical mechanics needs a *stationary* action, which
   is a minimum only for short times. The RMS-vs-ω·T scan (fig 3) stays ≈ 5e-3
   while ω·T ≲ 0.65π, then blows up — and it departs **before** the continuum
   conjugate point at ω·T = π, because the bounded-grid global minimum gives way
   first.

4. **Beyond the conjugate point (ω·T = 1.5π) — least ≠ stationary.** The classical
   path is now a *saddle* of the action, and the DP minimizer abandons it entirely,
   hugging the domain boundary to exploit the −V reward (fig 4). To recover the
   physical path here you must **root-find the discrete Euler–Lagrange equations**
   (a variational integrator), not minimize.

5. **The discrete Lagrangian is 2nd-order consistent.** The force residual
   `m q'' + V'` of the analytic solution scales as O(dt²) (fig 5), confirming the
   midpoint scheme.

## Two honest paths forward
- **Option A — minimizing regime.** Restrict to systems/intervals with ω·T well
  below π. The learned cost-to-go + focal search from the report transfer directly;
  add energy-drift and EL-residual as physical-fidelity metrics alongside node
  expansions. Immediately reusable, honestly limited.
- **Option B — general stationary action.** Build a discrete-EL root-finder
  (Störmer–Verlet style). This is the real "Lagrangian engine" but the
  node-expansion / learned-search story does **not** transfer, since you are no
  longer minimizing a cost over a graph. Keep it a separate object.

## Files
- `discrete_least_action.py` — the module (systems, discrete Lagrangian, DP, analytic, metrics)
- `run_experiment.py` — the five experiments + figures
- `phys_fig1_free_particle.png`, `phys_fig2_sho_short.png`, `phys_fig3_regime.png`,
  `phys_fig4_sho_long_saddle.png`, `phys_fig5_del_convergence.png`
