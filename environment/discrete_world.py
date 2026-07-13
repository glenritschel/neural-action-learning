import enum
from typing import List, Tuple, Set

class Action(enum.Enum):
    UP = (0, 1)
    DOWN = (0, -1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    STAY = (0, 0)

class DiscreteWorld:
    def __init__(self, grid_size_x: int = 20, grid_size_y: int = 20, time_steps: int = 40):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.time_steps = time_steps
        # Obstacles are represented as a set of (x, y) tuples
        self.obstacles: Set[Tuple[int, int]] = set()
        # Trajectories are stored as a list of lists of states (x, y, t)
        self.trajectories: List[List[Tuple[int, int, int]]] = []

    def set_obstacle(self, x: int, y: int):
        """Adds an obstacle at the given (x, y) coordinate if within bounds."""
        if self.is_within_bounds(x, y):
            self.obstacles.add((x, y))
        else:
            raise ValueError(f"Coordinate ({x}, {y}) is out of bounds.")

    def is_obstacle(self, x: int, y: int) -> bool:
        """Checks if there is an obstacle at the given (x, y) coordinate."""
        return (x, y) in self.obstacles

    def is_within_bounds(self, x: int, y: int) -> bool:
        """Checks if the given (x, y) coordinate is within the grid."""
        return 0 <= x < self.grid_size_x and 0 <= y < self.grid_size_y

    def add_trajectory(self, trajectory: List[Tuple[int, int, int, int, int]]):
        """Adds a trajectory to the world."""
        # Simple validation
        for state in trajectory:
            if len(state) != 5:
                raise ValueError("State must be a tuple of (x, y, vx, vy, t).")
            x, y, vx, vy, t = state
            if not self.is_within_bounds(x, y):
                raise ValueError(f"State ({x}, {y}, {vx}, {vy}, {t}) is out of bounds.")
            if not (0 <= t < self.time_steps):
                raise ValueError(f"Time step {t} is out of bounds (0 to {self.time_steps - 1}).")

        self.trajectories.append(trajectory)

    def is_valid_state(self, x: int, y: int, t: int) -> bool:
        """Checks if a state (x, y, t) is valid (within bounds, correct time). Allows stepping onto obstacles (soft obstacles)."""
        return self.is_within_bounds(x, y) and (0 <= t < self.time_steps)
