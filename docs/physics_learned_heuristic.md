# Learned Cost-to-Go for Discrete Least-Action Planning

### A matched-quality benchmark on a physical action (companion to the grid-planner report)

**Glen Ritschel** · [`neural-action-learning`](https://github.com/glenritschel/neural-action-learning) (Apache-2.0)

> Companion to [`docs/report.md`](report.md). That report benchmarked a learned heuristic on an *engineered* grid cost. This note asks the same question for a *physical* least-action cost, and finds a cleaner win — but only when the learned value is used as a search heuristic directly (weighted A\*), not inside the bounded focal search.

## Summary

A small MLP is trained to predict the exact cost-to-go of a 2D discrete least-action problem (a particle in a potential, minimising the discrete action `S = sum dt*(T_kin - V)`). Used as the heuristic in **weighted A\***, it reduces node expansions by **~5-21x at ≤1% suboptimality**, across two potentials and two grid sizes, with a held-out value-prediction R² of 0.999. The effect is robust but comes with three honest caveats: it is **unbounded** (the provably-bounded focal search did *not* benefit), it is measured in **node expansions, not wall-clock**, and it is a small-grid / single-seed / minimising-regime result.

## Setup

Config space is a 2D grid; state is `(x, y, t)` on a time-layered trellis with fixed endpoints and fixed total time (a two-point boundary value problem in the short-time *minimising* regime, `omega*T = 0.3π`). The transition cost is the discrete Lagrangian `L_d = dt*(0.5 m |v|^2 - V(q_mid))` (midpoint rule).

**Gauge shift (the enabling fix).** A physical action can be negative, which breaks the standard multiplicative `cost ≤ W*C*` bound and destabilises the search. Adding a constant `c = Vmax` to the Lagrangian makes every edge cost non-negative; because every fixed-endpoint, fixed-step path gains the *same* constant `c*(N-1)*dt`, the optimal path is unchanged (verified to machine precision). With non-negative costs, the report's standard machinery applies directly.

Three potentials are used: harmonic (separable, has a closed-form heuristic — omitted here as a solved case), a **double well** (anharmonic, separable), and a **coupled `x²y²`** potential (non-separable). Exact dynamic programming gives both the ground-truth optimum `C*` and the training labels; train and test goals are a disjoint, leakage-guarded split.

## Methods

- **Admissible A\*** — optimal baseline, admissible kinetic lower bound.
- **Learned focal (bounded)** — A\*ε with the network as focal guidance; provably `cost ≤ W*C*`.
- **Learned weighted A\*** — the network used as the search heuristic itself (`f = g + Wh·h_learned`); fast, but **no** guarantee.

## Result

The network learns the value surface essentially perfectly (held-out R² = 0.999 in every configuration). The **bounded focal search does not beat admissible A\***: on the small double well it expanded ~650 nodes versus admissible's ~432 — the loose admissible bound leaves focal little to exploit. The win comes entirely from **weighted A\***, reported below at **matched quality** (the fastest inflation `Wh` that stays within 1% of optimal — the fair comparison, since a fixed `Wh` hides quality differences):

| Potential | Size | R² | C\* | admissible A\* (exp) | learned wA\* (exp) | speedup | realized suboptimality |
|---|---|---:|---:|---:|---:|---:|---:|
| Double well | small | 0.999 | 6.9 | 432 | 21 | **21×** | 0.67% |
| Coupled | small | 0.999 | 13.5 | 1140 | 178 | **6.4×** | 0.27% |
| Double well | large | 0.999 | 8.8 | 841 | 52 | **16×** | 0.78% |
| Coupled | large | 0.999 | 2034 | 393 | **5.2×** | 0.24% |

<p align="center">
  <img src="figures/matched_quality_speedup.png" width="640" alt="Matched-quality speedup">
</p>

*Figure 1. Node expansions of learned weighted A\* (held to ≤1% suboptimality) versus optimal admissible A\*, across two potentials and two grid sizes. Speedup annotated above each bar.*

## Reading it honestly

- **The win is real but modest, and much smaller than a naive run suggests.** At a fixed inflation `Wh = 2`, these same configs showed 20-125× speedups — but at 2-9% suboptimality. Held to a fair ≤1% quality, they collapse to **5-21×**. The matched-quality number is the one that survives scrutiny.
- **The advantage is larger on *smoother* potentials, not harder ones.** The double well (16-21×) beats the coupled potential (5-6×). On the rougher coupled value surface the network's small errors force a safe `Wh = 1`, capping the gain — the opposite of the fixed-`Wh` reading. Speedup is roughly flat across grid size.
- **No bound.** The provably-bounded focal method did not help; the speedup requires the unbounded weighted-A\* use of the network. That trade — dropping the guarantee for a well-trained value function — is standard in the learned-heuristic literature and is exactly what this reproduces.
- **Node expansions, not seconds.** As in the main report, this counts search effort; per-node network inference cost is not included, so a wall-clock win is not claimed.

## Limitations

Small grids (21² and 29²), a single seed, six held-out goals per configuration, the short-time minimising regime only, and a value surface learned from a single obstacle-free potential per run. Past the conjugate point (`omega*T > π`) least ≠ stationary and this framing does not apply. R² ≈ 1.0 is strikingly clean and may partly reflect a smooth, low-dimensional value surface that generalises easily at this scale; larger and rougher problems are the real test.

## Reproduce

Everything here is produced by the self-contained Colab notebook `learned_least_action_colab.ipynb` (the final "2×2 matched-quality sweep" cell). The numpy baselines it builds on are in [`docs/physics2d/`](physics2d/PHYSICS2D_NOTES.md).
