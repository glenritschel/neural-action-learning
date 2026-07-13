import pytest
import torch
import os
from unittest.mock import patch, MagicMock

import experiments.benchmark as benchmark

@patch('torch.load')
@patch('experiments.benchmark.ActionMLP')
@patch('os.path.exists')
def test_benchmark_loads_checkpoint(mock_exists, mock_action_mlp, mock_torch_load):
    # Set up mocks
    mock_exists.return_value = True
    mock_model_instance = MagicMock()
    mock_model_instance.input_dim = 11
    mock_action_mlp.return_value = mock_model_instance
    mock_torch_load.return_value = {"dummy": "state_dict"}

    # Run only up to the setup to avoid running full benchmark loop
    with patch('experiments.benchmark.pd.DataFrame'), \
         patch('experiments.benchmark.dynamic_programming_search', return_value=MagicMock(path=[], cost=float('inf'), nodes_expanded=0, runtime=0, memory_footprint=0)), \
         patch('experiments.benchmark.a_star_search', return_value=MagicMock(path=[], cost=float('inf'), nodes_expanded=0, runtime=0, memory_footprint=0)), \
         patch('experiments.benchmark.focal_search', return_value=MagicMock(path=[], cost=float('inf'), nodes_expanded=0, runtime=0, memory_footprint=0)), \
         patch('experiments.benchmark.beam_search', return_value=MagicMock(path=[], cost=float('inf'), nodes_expanded=0, runtime=0, memory_footprint=0)), \
         patch('experiments.benchmark.json.load', return_value={"mae": 1.0, "r2": 0.5}), \
         patch('builtins.open', MagicMock()):

        benchmark.run_benchmark()

    mock_torch_load.assert_called_once()
    assert 'action_mlp_weights.pth' in mock_torch_load.call_args[0][0]
    mock_model_instance.load_state_dict.assert_called_once_with({"dummy": "state_dict"})
