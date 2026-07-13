from typing import List, Tuple
from environment.discrete_world import DiscreteWorld, Action

class ActionCalculator:
    def __init__(self, world: DiscreteWorld, move_weight: float = 1.0, turn_weight: float = 0.25, obstacle_weight: float = 100.0, acceleration_weight: float = 1.0):
        self.world = world
        self.move_weight = move_weight
        self.turn_weight = turn_weight
        self.obstacle_weight = obstacle_weight
        self.acceleration_weight = acceleration_weight

    def calculate_cost(self, trajectory: List[Tuple[int, int, int]]) -> float:
        """
        Calculates the total action cost for a given trajectory.
        A trajectory is a list of states (x, y, t).
        """
        if not trajectory:
            return 0.0

        total_cost = 0.0

        # Calculate obstacle cost for all states in trajectory
        for x, y, _ in trajectory:
            if self.world.is_obstacle(x, y):
                total_cost += self.obstacle_weight

        if len(trajectory) < 2:
            return total_cost

        # Calculate movement, turning, and acceleration costs
        # A transition from t to t+1 gives a velocity vector
        velocities = []
        for i in range(len(trajectory) - 1):
            x1, y1, _ = trajectory[i]
            x2, y2, _ = trajectory[i+1]
            dx, dy = x2 - x1, y2 - y1
            velocities.append((dx, dy))

            # Movement cost: incurred if there is any movement
            if dx != 0 or dy != 0:
                total_cost += self.move_weight

        # Turning cost: incurred when velocity vector changes direction
        for i in range(len(velocities) - 1):
            vx1, vy1 = velocities[i]
            vx2, vy2 = velocities[i+1]

            # If moving and changing direction
            if (vx1 != 0 or vy1 != 0) and (vx2 != 0 or vy2 != 0):
                if (vx1, vy1) != (vx2, vy2):
                    total_cost += self.turn_weight

            # Acceleration cost: squared change in velocity
            dvx = vx2 - vx1
            dvy = vy2 - vy1
            accel_cost = (dvx**2 + dvy**2) * self.acceleration_weight
            total_cost += accel_cost

        return total_cost
