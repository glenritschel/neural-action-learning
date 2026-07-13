import torch
import pytest
from latent_optimization.optimizer import LatentOptimizer
from autoencoder.autoencoder import TrajectoryAutoencoder
from models.mlp import ActionMLP

def test_latent_optimizer():
    # Setup dummy models
    input_dim = 10
    latent_dim = 8
    output_dim = 2

    autoencoder = TrajectoryAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    action_network = ActionMLP(input_dim=input_dim, output_dim=output_dim)

    optimizer = LatentOptimizer(autoencoder, action_network, learning_rate=0.1)

    # Initialize optimization targets and vectors
    initial_latent = torch.randn(1, latent_dim)
    target_action = torch.tensor([[1.0, -1.0]])

    # Perform optimization
    optimized_latent, loss_history = optimizer.optimize(initial_latent, target_action, steps=10)

    # Verify the output shapes and properties
    assert optimized_latent.shape == (1, latent_dim)
    assert len(loss_history) == 10

    # Verify the latent vector was actually updated (optimization occurred)
    assert not torch.allclose(initial_latent, optimized_latent)

    # Verify loss generally decreases (or at least the final loss is computed)
    assert isinstance(loss_history[-1], float)
