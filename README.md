# neural-action-learning

A research platform for learning action functions over discrete trajectories using neural networks, graph search, and latent-space optimization.

## Report

**[Learned Cost-to-Go Heuristics for Discrete Trajectory Optimization](docs/report.md)** — an honest benchmark of a learned heuristic inside a bounded-suboptimal focal search ([PDF](docs/report.pdf)).

Short version: on held-out goals, the learned heuristic used in a focal search (A\*ε, w = 1.5) expands about **21% fewer nodes** than admissible A\* — and 3–6× fewer on reachable goals — while staying inside its provable 1.5× suboptimality bound. It does **not** yet win on wall-clock time: per-node network evaluation currently costs more than the node savings buy back on a grid this small. The method reduces *search*, not yet *seconds*. See the report for methods, figures, and limitations.

## Layout

- `search/` — DP, A\*, focal search (A\*ε), beam search
- `models/`, `autoencoder/`, `gnn/` — the learned cost-to-go network and feature builder
- `environment/`, `action/` — the phase-space grid world and cost model
- `experiments/` — training, benchmark harness, and the persisted train/test goal split
- `tests/` — correctness and bound gates
- `docs/` — the benchmark report and figures
