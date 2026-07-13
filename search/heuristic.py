import torch
from typing import Tuple
from models.mlp import ActionMLP
from environment.discrete_world import DiscreteWorld

class MLPHeuristic:
    def __init__(self, model: ActionMLP, world: DiscreteWorld):
        self.model = model
        self.world = world
        self.model.eval()

    def predict_cost(self, current_state: Tuple[int, int, int], goal_state: Tuple[int, int]) -> float:
        """
        Normalizes the state and uses the MLP to predict the remaining cost to the goal.
        """
        x, y, t = current_state
        gx, gy = goal_state

        # Simple normalization based on grid size and max time
        nx = x / float(self.world.grid_size_x)
        ny = y / float(self.world.grid_size_y)
        nt = t / float(self.world.time_steps)

        ngx = gx / float(self.world.grid_size_x)
        ngy = gy / float(self.world.grid_size_y)

        # Distance metrics for heuristic
        dx = ngx - nx
        dy = ngy - ny

        # Prepare input tensor. Assuming MLP expects [nx, ny, nt, dx, dy] for a state-to-goal heuristic
        # If the MLP was trained differently (e.g. on full trajectories), this might need adjustment,
        # but this represents a standard state-based heuristic input
        features = torch.tensor([[nx, ny, nt, dx, dy]], dtype=torch.float32)

        with torch.no_grad():
            predicted_cost = self.model(features)

        return predicted_cost.item()
