from typing import Tuple, Dict, Any
from environment.discrete_world import DiscreteWorld, Action
from action.action_calculator import ActionCalculator

class DiscreteWorldEnv:
    def __init__(self, world: DiscreteWorld, calculator: ActionCalculator, goal_state: Tuple[int, int], max_steps: int = 100):
        self.world = world
        self.calculator = calculator
        self.goal_state = goal_state
        self.max_steps = max_steps

        self.current_state = None
        self.path = []
        self.steps_taken = 0

    def reset(self, start_state: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Resets the environment to the start state."""
        if not self.world.is_valid_state(start_state[0], start_state[1], start_state[2]):
            raise ValueError("Invalid start state")

        self.current_state = start_state
        self.path = [start_state]
        self.steps_taken = 0
        return self.current_state

    def step(self, action: Action) -> Tuple[Tuple[int, int, int], float, bool, Dict[str, Any]]:
        """
        Takes a step in the environment.
        Returns: next_state, reward, done, info
        """
        if self.current_state is None:
            raise RuntimeError("Environment must be reset before taking steps.")

        x, y, t = self.current_state
        dx, dy = action.value

        next_x, next_y = x + dx, y + dy
        next_t = t + 1

        next_state = (next_x, next_y, next_t)

        # Check if valid
        if not self.world.is_valid_state(next_x, next_y, next_t):
            # Invalid move (hit wall or obstacle or time out)
            # Stay in place but time progresses, heavily penalize
            reward = -100.0
            done = True
            info = {"reason": "invalid_state"}
            return self.current_state, reward, done, info

        # Add to path to calculate cost
        temp_path = self.path + [next_state]

        # In RL, we usually define reward as the negative of the cost incurred in this step.
        # However, our ActionCalculator computes cost over the whole path (including turning/accel history).
        # We can compute the incremental cost:
        cost_before = self.calculator.calculate_cost(self.path)
        cost_after = self.calculator.calculate_cost(temp_path)
        step_cost = cost_after - cost_before

        reward = -step_cost

        self.current_state = next_state
        self.path = temp_path
        self.steps_taken += 1

        done = False
        info = {"reason": "step"}

        # Check if reached goal (ignoring time)
        if (next_x, next_y) == self.goal_state:
            reward += 100.0 # Goal bonus
            done = True
            info = {"reason": "goal_reached"}

        elif self.steps_taken >= self.max_steps or next_t >= self.world.time_steps:
            done = True
            info = {"reason": "max_steps_or_time"}

        return self.current_state, reward, done, info
