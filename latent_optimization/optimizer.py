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

    def optimize(self, initial_latent: torch.Tensor, steps: int = 100) -> Tuple[torch.Tensor, list]:
        """
        Perform gradient descent on a generated latent vector to minimize predicted cost.
        Note: This is a small-T track technique. Fixed-length latents do not scale to T=500.

        Args:
            initial_latent: The starting point in latent space (requires_grad will be set to True internally)
            steps: Number of optimization steps

        Returns:
            Optimized latent vector, list of loss values over time
        """
        # Create a new leaf tensor that requires gradients, starting from initial_latent's data
        latent_vector = initial_latent.clone().detach().requires_grad_(True)

        optimizer = optim.Adam([latent_vector], lr=self.learning_rate)

        loss_history = []

        for step in range(steps):
            optimizer.zero_grad()

            # Pass latent through decoder
            reconstructed_trajectory = self.autoencoder.decoder(latent_vector)

            # Evaluate using action network (which predicts cost)
            predicted_cost = self.action_network(reconstructed_trajectory)

            # We want to minimize the predicted cost, so the loss is simply the cost itself.
            # We take the mean across the batch if applicable.
            loss = predicted_cost.mean()

            # Backpropagate to update latent vector
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

        return latent_vector.detach(), loss_history

    def project_to_valid_path(self, latent_vector: torch.Tensor, world, start_state: tuple) -> tuple:
        """
        Decodes the latent vector, rounds to nearest discrete grid coordinates,
        and attempts to repair the path to enforce adjacency and validity.

        Args:
            latent_vector: The optimized latent representation.
            world: The DiscreteWorld instance.
            start_state: The known starting state (x, y, vx, vy, t)

        Returns:
            A tuple (valid_path, is_valid)
        """
        with torch.no_grad():
            reconstructed = self.autoencoder.decoder(latent_vector).squeeze()

        # Assuming the reconstructed trajectory is flattened [nx, ny, nt, nx, ny, nt...]
        # and has length 3 * T.
        # (For this sub-track, we assume a simple small-T fixed-length vector).
        coords = reconstructed.numpy()

        path = [start_state]
        is_valid = True

        # We need to reshape the flat array. Let's assume it maps to 3 values per step (nx, ny, nt)
        # We will step through the decoded sequence, round, and try to make a valid transition.
        num_steps = len(coords) // 3

        current_state = start_state

        for i in range(num_steps):
            idx = i * 3
            if idx + 2 >= len(coords):
                break

            # Un-normalize
            rx = int(round(coords[idx] * world.grid_size_x))
            ry = int(round(coords[idx+1] * world.grid_size_y))

            x, y, vx, vy, t = current_state
            next_t = t + 1

            # Find the valid adjacent action that brings us closest to (rx, ry)
            best_action = None
            best_dist = float('inf')

            from environment.discrete_world import Action
            for action in Action:
                dx, dy = action.value
                nx, ny = x + dx, y + dy
                if world.is_valid_state(nx, ny, next_t):
                    dist = abs(nx - rx) + abs(ny - ry)
                    if dist < best_dist:
                        best_dist = dist
                        best_action = (dx, dy)

            if best_action is None:
                is_valid = False
                break

            dx, dy = best_action
            nx, ny = x + dx, y + dy
            current_state = (nx, ny, dx, dy, next_t)
            path.append(current_state)

        return path, is_valid
