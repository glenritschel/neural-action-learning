import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

class LatentOptimizer:
    def __init__(self, autoencoder: nn.Module, action_network: nn.Module, learning_rate: float = 0.01):
        self.autoencoder = autoencoder
        self.action_network = action_network
        self.learning_rate = learning_rate

        # Ensure models are in eval mode during optimization to prevent updating their weights
        self.autoencoder.eval()
        self.action_network.eval()

    def optimize(self, initial_latent: torch.Tensor, target_action: torch.Tensor, steps: int = 100) -> Tuple[torch.Tensor, list]:
        """
        Perform gradient descent on a generated latent vector to match a target action.

        Args:
            initial_latent: The starting point in latent space (requires_grad will be set to True internally)
            target_action: The desired output from the action network
            steps: Number of optimization steps

        Returns:
            Optimized latent vector, list of loss values over time
        """
        # Create a new leaf tensor that requires gradients, starting from initial_latent's data
        latent_vector = initial_latent.clone().detach().requires_grad_(True)

        optimizer = optim.Adam([latent_vector], lr=self.learning_rate)
        criterion = nn.MSELoss()

        loss_history = []

        for step in range(steps):
            optimizer.zero_grad()

            # Pass latent through decoder
            reconstructed_trajectory = self.autoencoder.decoder(latent_vector)

            # Evaluate using action network
            predicted_action = self.action_network(reconstructed_trajectory)

            # Compute loss against target action
            loss = criterion(predicted_action, target_action)

            # Backpropagate to update latent vector
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

        return latent_vector.detach(), loss_history
