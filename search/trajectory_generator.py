from typing import List, Dict, Any, Tuple
from environment.discrete_world import DiscreteWorld, Action
from action.action_calculator import ActionCalculator

class TrajectoryGenerator:
    def __init__(self, world: DiscreteWorld, calculator: ActionCalculator):
        self.world = world
        self.calculator = calculator

    def generate(self, start_state: Tuple[int, int, int], depth: int) -> List[Dict[str, Any]]:
        """
        Generates all valid trajectories of length `depth` (number of moves) from the start_state.
        Trajectory length in states will be `depth + 1`.
        Returns a list of dictionaries with trajectory, cost, endpoint, and metadata.
        """
        results = []

        def dfs(current_state: Tuple[int, int, int], current_path: List[Tuple[int, int, int]]):
            if len(current_path) == depth + 1:
                cost = self.calculator.calculate_cost(current_path)
                endpoint = current_path[-1]
                results.append({
                    "trajectory": list(current_path),
                    "cost": cost,
                    "endpoint": endpoint,
                    "metadata": {
                        "length": len(current_path),
                        "start": current_path[0]
                    }
                })
                return

            x, y, t = current_state
            next_t = t + 1

            for action in Action:
                dx, dy = action.value
                nx, ny = x + dx, y + dy

                # Check validity
                if self.world.is_valid_state(nx, ny, next_t):
                    current_path.append((nx, ny, next_t))
                    dfs((nx, ny, next_t), current_path)
                    current_path.pop()

        # Check start state validity
        x, y, t = start_state
        if self.world.is_valid_state(x, y, t):
            dfs(start_state, [start_state])

        return results
