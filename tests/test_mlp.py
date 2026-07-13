import torch
import pytest
from models.mlp import ActionMLP

def test_mlp_forward_shape():
    input_dim = 10
    output_dim = 2
    batch_size = 32

    model = ActionMLP(input_dim=input_dim, output_dim=output_dim)

    # Create dummy input
    x = torch.randn(batch_size, input_dim)

    # Forward pass
    output = model(x)

    # Check output shape
    assert output.shape == (batch_size, output_dim)

def test_mlp_loss_and_metrics():
    model = ActionMLP(input_dim=10, output_dim=2)

    predictions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    loss = model.compute_loss(predictions, targets)
    assert loss.item() == 0.0

    metrics = model.compute_metrics(predictions, targets)
    assert metrics['mae'] == 0.0
    assert metrics['rmse'] == 0.0
    assert metrics['r2'] == 1.0

    # Test with errors
    predictions_err = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    loss_err = model.compute_loss(predictions_err, targets)
    assert loss_err.item() == 1.0

    metrics_err = model.compute_metrics(predictions_err, targets)
    assert metrics_err['mae'] == 1.0
    assert metrics_err['rmse'] == 1.0
