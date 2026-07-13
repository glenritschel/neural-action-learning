import torch
import pytest
from autoencoder.autoencoder import TrajectoryAutoencoder

def test_autoencoder_configurable_latents():
    input_dim = 100
    batch_size = 16

    # Test valid latent dimensions
    for latent_dim in [8, 16, 32, 64]:
        model = TrajectoryAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

        # Test shape
        x = torch.randn(batch_size, input_dim)
        output = model(x)
        assert output.shape == (batch_size, input_dim)

        # Test latent representation shape
        latent = model.encoder(x)
        assert latent.shape == (batch_size, latent_dim)

def test_autoencoder_invalid_latent():
    with pytest.raises(ValueError, match="latent_dim must be one of"):
        TrajectoryAutoencoder(input_dim=100, latent_dim=10)

def test_autoencoder_reconstruction_error():
    model = TrajectoryAutoencoder(input_dim=10)

    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    reconstructed = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    error = model.compute_reconstruction_error(original, reconstructed)
    assert error.item() == 0.0

    reconstructed_err = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    error_err = model.compute_reconstruction_error(original, reconstructed_err)
    assert error_err.item() == 1.0
