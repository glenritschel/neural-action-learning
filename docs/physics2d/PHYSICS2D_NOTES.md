# 2D physical least-action — headroom check (before training anything)

**Question:** is there room for a learned cost-to-go heuristic to help on a
*physical* least-action problem, or is a good admissible A\* already efficient?
Measured with exact DP vs admissible A\* — pure numpy, no neural net.

Setup: 2D config space, discrete Lagrangian `L_d = dt[0.5 m |v|^2 - V(q_mid)]`,
fixed endpoints, fixed T, short-time minimising regime (omega*T = 0.3π). Grid
25×25 × 16 time layers (10,000 states), local velocity-bounded moves.

## Result — there is headroom, and only where the potential is non-trivial

| System | admissible h | mean A\* expansions | A\* / optimal-path length | vs DP |
|---|---|---:|---:|---:|
| Free particle | exact (straight-line action) | 181 | ~11× | 54× fewer than DP |
| Harmonic oscillator | loose (can't see the −V reward) | 863 | ~54× | 6.5× fewer than DP |

Both return the exact optimum (A\* = DP) and track the analytic classical path
(RMS ~0.08–0.12, grid-limited).

- **Free particle is a solved case.** The physical admissible heuristic is *exact*,
  so A\* is already near-efficient and a learned heuristic has almost nothing to add.
- **The harmonic oscillator is where learning could pay.** The admissible bound
  subtracts a worst-case potential reward, so it is loose; A\* expands ~54× the
  optimal path length.

## And the headroom grows with problem size

Scaling the oscillator (short T, one goal):

| states | DP exp | A\* exp | A\* / path |
|---:|---:|---:|---:|
| 4,332 | 2,193 | 366 | 30× |
| 10,000 | 5,070 | 935 | 58× |
| 19,220 | 9,735 | 1,899 | 95× |
| 32,856 | 16,620 | 3,267 | 136× |

The loose admissible heuristic scales poorly, so the room for a learned heuristic
*increases* with problem size. This is the honest green light for the learned-
heuristic experiment — pointed specifically at non-trivial potentials, not the
free particle.

## What this does and does not show
- **Does:** there is real, growing headroom between admissible A\* and the optimal
  path on a physical least-action problem with a non-trivial potential.
- **Does not:** prove a *learned* heuristic can capture it. That is the next test —
  train a cost-to-go on exact-DP labels (leakage-guarded) and see whether focal
  search closes the gap toward the free-particle-like efficiency, within its bound,
  while keeping the returned path physically faithful (energy drift, EL residual).

## Recommended next step
Run the learned-heuristic experiment on the 2D harmonic oscillator (and richer
potentials — double well, central force) in the minimising regime. That needs
training (torch → Colab/Jules, same loop as the main report); this baseline says
it is worth doing and where to aim it.

## Files
- `physics2d.py` — env, discrete Lagrangian, exact DP, admissible A\*, fidelity
- `run2d.py` — the 5-goal comparison
- `phys2d_headroom.png` — search effort and headroom summary

---

# Stress test — is the headroom fundamental, or a lazy-bound artifact?

The 54x oscillator headroom above used a deliberately loose admissible bound. This
test pits it against smarter hand-designed heuristics and two anharmonic potentials
(grid 25x25, N_t=16, omega*T=0.3pi). All heuristics are admissible; A* stays exact.

| Potential | lazy bound | velocity-cone bound | exact classical action |
|---|---:|---:|---:|
| Harmonic (separable) | 54x | 52x | **8.8x** |
| Double well (anharmonic) | 36x | 34x | n/a (no closed form) |
| Coupled x^2 y^2 (non-separable) | 111x | 105x | n/a (no closed form) |

(values = A* expansions / optimal-path length)

**What this changes:**

1. **The oscillator was a misleading example.** A closed-form classical-action
   heuristic collapses its headroom ~6x (863 -> 141 expansions). Where the system is
   exactly solvable, you don't need a learned heuristic — you write one down. Running
   the learned experiment on the oscillator would be beating a strawman.

2. **My cheap "smarter" bound (velocity cone) barely helped anywhere** (54->52, 36->34,
   111->105). A naive tighter bound is not the answer.

3. **The headroom is real where it matters:** anharmonic / non-separable potentials
   (double well, coupled) have no closed-form action, the cheap bounds stay loose, and
   34x-105x headroom survives. That is the honest case for a learned cost-to-go.

**One more control before training (not yet run):** the strongest *non-learned*
competitor is not a closed form or the cone bound — it is a cheap precomputed value
function (coarse-grid DP used as a heuristic, i.e. a pattern database). A learned
heuristic must beat *that*, not just the lazy bound. If coarse-DP already closes the
gap on the anharmonic potentials, learning is again only competing with cheap
engineering. This is the decisive de-risk to run next.

**Updated recommendation:** point any learned-heuristic experiment at anharmonic /
non-separable potentials (double well, coupled, central force) — never the plain
oscillator — and benchmark it against the coarse-DP value function, not the lazy bound.

---

# Coarse-DP control — and two obstacles it exposed

Strongest non-learned competitor: solve DP on a coarse 13x13 grid, use its cost-to-go
as a heuristic on the fine 25x25 grid (the non-learned analog of a learned value fn).

| Potential | A* lazy-admissible | A* + coarse-DP heuristic |
|---|---|---|
| Harmonic | 863 exp, 54x path, **optimal** | 54 exp, 3.4x path, **+80% cost** |
| Double well | 569 exp, 36x path, **optimal** | 45 exp, 2.8x path, **+85% cost** |
| Coupled | 1772 exp, 111x path, **optimal** | 51 exp, 3.2x path, **+84% cost** |

The coarse-DP heuristic collapses expansions (~3x path) but, used greedily, returns
paths ~80% above optimal. So it is fast-but-wrong with this crude config — not a free
lunch, but also not yet a fair comparison. Two real obstacles surfaced:

1. **The comparison must be at matched quality.** Admissible A* is optimal but slow;
   coarse-DP is fast but far off. The honest question is the speed/quality tradeoff
   (a Pareto curve), which means running every heuristic inside the SAME bounded-
   suboptimality search, then comparing expansions at equal quality.

2. **The bound itself breaks for a physical action.** The report's method guarantees
   `cost <= w * optimal` (multiplicative). But a Lagrangian action can be NEGATIVE, and
   a multiplicative bound is meaningless when C* < 0 (w*C* is *below* C*). Porting the
   focal-search machinery to physics needs an ADDITIVE bound (`cost <= C* + eps`), not
   the multiplicative one. (The search still works despite negative edge costs only
   because the time-layered graph is a DAG.)

## Revised recommendation
Do NOT train yet. The blocker is no longer "is there headroom" (there is, on anharmonic
potentials) — it is that the search framework and the comparison are not yet well-posed
for a signed action. Next honest step, still cheap and runnable without any network:
1. Implement focal search with an ADDITIVE suboptimality bound (`cost <= C* + eps`).
2. Run all heuristics (lazy admissible, coarse-DP guidance, and — oscillator only —
   classical action) inside that bounded search and plot expansions vs realized
   suboptimality (a Pareto curve) on the anharmonic potentials.
3. Only if a cheap value function cannot reach the low-expansion / low-suboptimality
   corner does a learned cost-to-go become justified — and then it is benchmarked on
   that same curve.

---

# Resolution: gauge-shift the Lagrangian (and what the Pareto actually showed)

Two things came out of building the additive-bound focal search and sweeping it:

1. **The additive bound works but the tradeoff is badly behaved.** With a signed action
   and a very loose admissible heuristic, expansions are non-monotone in eps and even a
   near-exact guide thrashes (reopening under an inconsistent, loose bound). This said
   the *framework*, not the heuristic, was the problem.

2. **The clean fix is a gauge shift, not an additive bound.** Adding a constant c >= Vmax
   to the Lagrangian makes every edge cost `dt*(T - V + c) >= 0`. Because every fixed-
   endpoint, fixed-step path has the same number of steps, this adds the SAME constant
   `c*(N-1)*dt` to every path -> the optimal path is unchanged (verified: `C*_shift -
   c*T == C*_unshift` to machine precision). With non-negative costs, the report's
   original machinery applies directly: multiplicative bound valid, no reopening thrash.

## Post-gauge measurements (double well, coupled; 17x17, N_t=12)

| Potential | admissible A* (optimal) | coarse-DP guided, W=1.5 |
|---|---:|---:|
| Double well | 188 exp | 378 exp (+0.52 subopt, bound ok) |
| Coupled | 633 exp | 907 exp (+0.02 subopt, bound ok) |

- The gauge shift restores sane behavior and a valid bound.
- **The cheap non-learned guides I tried (velocity-cone, coarse-DP) do NOT beat admissible
  A*** — coarse-DP guidance actually expands *more*. So they are not the strong
  competitor feared earlier; the earlier "coarse-DP collapses expansions" was only the
  greedy, +80%-suboptimal regime.
- Admissible A* still expands ~190-630 nodes (~17-57x the optimal path) on these
  anharmonic potentials, and no cheap heuristic tried closes that. That gap is the
  genuine, well-posed target for a learned cost-to-go.

## Bottom line of the whole de-risking arc
1. Headroom is real on anharmonic / non-separable potentials (not the oscillator, which
   is solved by a closed-form heuristic).
2. The signed-action framework obstacle is fixed by a one-line gauge shift that provably
   preserves the optimum and restores the multiplicative bound.
3. Cheap non-learned heuristics (cone, coarse-DP) fail to beat admissible A* here.
=> The learned-cost-to-go experiment is now well-posed and motivated. Run it on a
   gauge-shifted anharmonic potential, inside the report's standard focal search,
   benchmarked against admissible A* (and, honestly, a better-tuned coarse value fn).
