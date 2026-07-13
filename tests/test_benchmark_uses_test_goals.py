import json
import os
from unittest.mock import patch, MagicMock

import experiments.benchmark as benchmark

def _split_path():
    return os.path.join(os.path.dirname(benchmark.__file__), "goal_split.json")

@patch("experiments.benchmark.torch.load", return_value={})
@patch("experiments.benchmark.ActionMLP")
@patch("os.path.exists", return_value=True)
def test_benchmark_evaluates_on_test_goals(mock_exists, mock_mlp, mock_load):
    mock_model = MagicMock()
    mock_model.input_dim = 11
    mock_mlp.return_value = mock_model

    with open(_split_path()) as f:
        raw_goals = json.load(f)["test_goals"]
        expected = {tuple(g) for g in raw_goals} - {(0, 0)}

    seen = []
    stub = MagicMock(path=[], cost=float("inf"), nodes_expanded=0, runtime=0, memory_footprint=0)

    def capture(world, calculator, *args, **kwargs):
        # DP/A*/beam signature: (world, calculator, [heuristic], start, goal, max_depth, ...)
        goal = next(a for a in args if isinstance(a, tuple) and len(a) == 2)
        seen.append(goal)
        return stub

    with patch("experiments.benchmark.MLPHeuristic"), \
         patch("experiments.benchmark.dynamic_programming_search", side_effect=capture), \
         patch("experiments.benchmark.a_star_search", side_effect=capture), \
         patch("experiments.benchmark.focal_search", side_effect=capture), \
         patch("experiments.benchmark.beam_search", side_effect=capture), \
         patch("experiments.benchmark.pd.DataFrame"), \
         patch("experiments.benchmark.json.load", side_effect=[{"test_goals": raw_goals}, {"mae": 0.0, "r2": 0.0}]), \
         patch("builtins.open", MagicMock()):
        benchmark.run_benchmark()

    assert set(seen) == expected, f"benchmark evaluated on {set(seen)}, expected test_goals {expected}"
