from typing import Tuple, Optional, List
from environment.discrete_world import DiscreteWorld, Action
from action.action_calculator import ActionCalculator
import numpy as np

class PathingEnv:
    def __init__(self, world: DiscreteWorld, calculator: ActionCalculator, goal_state: Tuple[int, int]):
        self.world = world
        self.calculator = calculator
        self.goal_state = goal_state
        self.current_state: Optional[Tuple[int, int, int, int, int]] = None
        self.path: List[Tuple[int, int, int, int, int]] = []

    def get_observation(self) -> np.ndarray:
        x, y, vx, vy, t = self.current_state
        # Normalize
        nx = x / float(self.world.grid_size_x)
        ny = y / float(self.world.grid_size_y)
        nt = t / float(self.world.time_steps)
        nvx = float(vx)
        nvy = float(vy)
        gx = self.goal_state[0] / float(self.world.grid_size_x)
        gy = self.goal_state[1] / float(self.world.grid_size_y)
        return np.array([nx, ny, nvx, nvy, nt, gx, gy], dtype=np.float32)

    def reset(self, start_state: Tuple[int, int, int, int, int]) -> np.ndarray:
        self.current_state = start_state
        self.path = [start_state]
        return self.get_observation()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        action_list = list(Action)
        if action_idx < 0 or action_idx >= len(action_list):
            raise ValueError(f"Invalid action index {action_idx}")

        action = action_list[action_idx]
        x, y, vx, vy, t = self.current_state

        dx, dy = action.value
        next_t = t + 1
        nx, ny = x + dx, y + dy

        done = False
        reward = 0.0
        info = {}

        if next_t >= self.world.time_steps:
            done = True
            info['reason'] = 'timeout'
            reward = -100.0 # penalty for timeout
            return self.get_observation(), reward, done, info

        if not self.world.is_valid_state(nx, ny, next_t):
            # Invalid state (e.g. out of bounds or obstacle)
            done = True
            info['reason'] = 'invalid_move'
            reward = -100.0
            return self.get_observation(), reward, done, info

        # Valid move
        next_state = (nx, ny, dx, dy, next_t)

        # Calculate cost increment
        step_cost = self.calculator.calculate_transition_cost(self.current_state, next_state)
        self.path.append(next_state)

        reward = -step_cost # Reward is negative cost

        self.current_state = next_state

        if (nx, ny) == self.goal_state:
            done = True
            info['reason'] = 'goal_reached'
            # Could add a large positive reward for reaching the goal if desired,
            # but usually minimizing cost handles this. We might add a completion bonus.
            reward += 1000.0

        return self.get_observation(), reward, done, info
