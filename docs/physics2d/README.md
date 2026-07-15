# physics2d — 2D discrete least-action baselines

Numpy baselines behind the companion note [`../physics_learned_heuristic.md`](../physics_learned_heuristic.md).
These scripts were written to **de-risk** the "does a learned cost-to-go help?" question
*before* training any network — and to fix the search framework for a physical (signed) action.
Read them in order; each answers one question and motivates the next.

| File | Question it answers |
|---|---|
| `physics2d.py` | The shared environment: 2D grid, discrete Lagrangian `L_d = dt*(T_kin - V)`, exact DP, admissible A\*, energy/EL fidelity. |
| `run2d.py` | Is there headroom for a better heuristic? (exact DP vs admissible A\* node expansions) |
| `run2d_stress.py` | Does the headroom survive a *smarter* hand heuristic, and generalize beyond the oscillator? (velocity-cone + closed-form vs lazy bound; harmonic / double-well / coupled) |
| `run2d_coarse.py` | Does a cheap precomputed value function (coarse-grid DP) already close the gap? |
| `run2d_gauge.py` | The fix: gauge-shift the Lagrangian by `c = Vmax` so costs are non-negative and the optimal path is unchanged, restoring the standard multiplicative bound. |
| `run2d_pareto.py` | An additive-bound focal search and a speed/quality Pareto over cheap guides (shows the tradeoff is badly behaved before the gauge fix). |

Figures: `phys2d_headroom.png` (headroom by potential), `phys2d_stress_headroom.png`
(headroom vs heuristic strength).

## What they concluded
1. Real headroom exists on **anharmonic / non-separable** potentials (not the oscillator, which a closed-form heuristic solves).
2. A signed action breaks the multiplicative bound; a one-line **gauge shift** fixes it and provably preserves the optimum.
3. Cheap non-learned heuristics (velocity-cone, coarse-DP) do **not** beat admissible A\* at matched quality.

That cleared the way for the learned-heuristic experiment. The result — a learned cost-to-go
in weighted A\* giving ~5–21× fewer node expansions at ≤1% suboptimality — is in the Colab
notebook (`learned_least_action.ipynb`) and summarized in the companion note above.

## Run
```
python run2d.py          # headroom
python run2d_stress.py   # headroom vs smarter heuristics + more potentials
python run2d_coarse.py   # coarse-DP control
python run2d_gauge.py    # gauge-shift fix
python run2d_pareto.py   # additive-bound focal + Pareto
```
Pure numpy + matplotlib; no GPU, seconds each.

See also [`PHYSICS2D_NOTES.md`](PHYSICS2D_NOTES.md) for the full narrative with numbers.
