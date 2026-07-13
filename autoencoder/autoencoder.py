import torch
import torch.nn as nn

class TrajectoryAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super(TrajectoryAutoencoder, self).__init__()

        if latent_dim not in [8, 16, 32, 64]:
            raise ValueError("latent_dim must be one of [8, 16, 32, 64]")

        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def compute_reconstruction_error(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Calculate the Mean Squared Error (MSE) reconstruction error."""
        return self.criterion(reconstructed, original)
