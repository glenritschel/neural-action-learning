import torch
from typing import Tuple
from models.mlp import ActionMLP
from environment.discrete_world import DiscreteWorld
from models.dataset_builder import DatasetBuilder

class MLPHeuristic:
    def __init__(self, model: ActionMLP, world: DiscreteWorld):
        self.model = model
        self.world = world
        self.model.eval()
        self.builder = DatasetBuilder(world.grid_size_x, world.grid_size_y, world.time_steps)

    def predict_cost(self, current_state: Tuple[int, int, int, int, int], goal_state: Tuple[int, int]) -> float:
        """
        Uses the shared DatasetBuilder to normalize the state and goal state into a fixed-size
        feature vector, then uses the MLP to predict the remaining cost to the goal.
        """
        # Ensure we are using the exact same feature function as training
        features = self.builder.build_features(current_state, goal_state)

        # Prepare input tensor
        features_tensor = torch.tensor([features], dtype=torch.float32)

        with torch.no_grad():
            predicted_cost = self.model(features_tensor)

        return predicted_cost.item()
