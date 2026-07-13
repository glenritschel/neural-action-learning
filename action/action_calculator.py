from typing import List, Tuple
from environment.discrete_world import DiscreteWorld, Action

class ActionCalculator:
    def __init__(self, world: DiscreteWorld, move_weight: float = 1.0, turn_weight: float = 0.25, obstacle_weight: float = 100.0, acceleration_weight: float = 1.0):
        self.world = world
        self.move_weight = move_weight
        self.turn_weight = turn_weight
        self.obstacle_weight = obstacle_weight
        self.acceleration_weight = acceleration_weight

    def calculate_transition_cost(self, current_state: Tuple[int, int, int, int, int], next_state: Tuple[int, int, int, int, int]) -> float:
        """
        Calculates the incremental cost of a transition from current_state to next_state.
        States are represented as (x, y, vx, vy, t).
        """
        x1, y1, vx1, vy1, t1 = current_state
        x2, y2, vx2, vy2, t2 = next_state

        cost = 0.0

        # Obstacle cost (incurred if entering an obstacle)
        if self.world.is_obstacle(x2, y2):
            cost += self.obstacle_weight

        # Movement cost (incurred if there is any movement)
        if vx2 != 0 or vy2 != 0:
            cost += self.move_weight

        # Turning cost: incurred when velocity vector changes direction and both are non-zero
        if (vx1 != 0 or vy1 != 0) and (vx2 != 0 or vy2 != 0):
            if (vx1, vy1) != (vx2, vy2):
                cost += self.turn_weight

        # Acceleration cost: squared change in velocity
        dvx = vx2 - vx1
        dvy = vy2 - vy1
        accel_cost = (dvx**2 + dvy**2) * self.acceleration_weight
        cost += accel_cost

        return cost

    def calculate_cost(self, trajectory: List[Tuple[int, int, int, int, int]]) -> float:
        """
        Calculates the total action cost for a given trajectory.
        A trajectory is a list of states (x, y, vx, vy, t).
        """
        if not trajectory:
            return 0.0

        total_cost = 0.0

        # Initial obstacle cost for the start state
        x0, y0, vx0, vy0, t0 = trajectory[0]
        if self.world.is_obstacle(x0, y0):
            total_cost += self.obstacle_weight

        for i in range(len(trajectory) - 1):
            total_cost += self.calculate_transition_cost(trajectory[i], trajectory[i+1])

        return total_cost
