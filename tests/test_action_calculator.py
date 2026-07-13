import pytest
from environment.discrete_world import DiscreteWorld
from action.action_calculator import ActionCalculator

@pytest.fixture
def world():
    w = DiscreteWorld(20, 20, 40)
    w.set_obstacle(5, 5)
    return w

@pytest.fixture
def calculator(world):
    return ActionCalculator(world, move_weight=1.0, turn_weight=0.25, obstacle_weight=100.0, acceleration_weight=1.0)

def test_empty_trajectory(calculator):
    assert calculator.calculate_cost([]) == 0.0

def test_single_state_trajectory(calculator):
    # Just standing still, no velocity
    assert calculator.calculate_cost([(0, 0, 0)]) == 0.0

def test_single_state_obstacle(calculator):
    # Single state on obstacle
    assert calculator.calculate_cost([(5, 5, 0)]) == 100.0

def test_straight_movement(calculator):
    # Trajectory: (0,0)->(1,0)->(2,0)
    # Velocities: (1,0), (1,0)
    # Moves: 2 (cost 1.0 each)
    # Turn: 0
    # Accel: velocity diff is (0,0) -> cost 0
    traj = [(0, 0, 0), (1, 0, 1), (2, 0, 2)]
    cost = calculator.calculate_cost(traj)
    assert cost == 2.0

def test_movement_with_turn(calculator):
    # Trajectory: (0,0)->(1,0)->(1,1)
    # Velocities: v1=(1,0), v2=(0,1)
    # Moves: 2
    # Turn: 1 (cost 0.25)
    # Accel: v2 - v1 = (-1, 1). dvx=-1, dvy=1. dvx^2 + dvy^2 = 1 + 1 = 2 (cost 2.0 * 1.0)
    # Total cost = 2.0 (moves) + 0.25 (turn) + 2.0 (accel) = 4.25
    traj = [(0, 0, 0), (1, 0, 1), (1, 1, 2)]
    cost = calculator.calculate_cost(traj)
    assert cost == 4.25

def test_movement_into_obstacle(calculator):
    # Trajectory: (4,5)->(5,5)
    # Moves: 1
    # Obstacles: 1 (cost 100)
    # Accel/Turn: not enough states for turn or accel
    traj = [(4, 5, 0), (5, 5, 1)]
    cost = calculator.calculate_cost(traj)
    assert cost == 101.0

def test_stop_and_go(calculator):
    # Trajectory: (0,0)->(1,0)->(1,0)->(2,0)
    # Velocities: v1=(1,0), v2=(0,0), v3=(1,0)
    # Moves: 2 (v1, v3) -> cost 2.0
    # Turn: 0 (changing to/from 0,0 doesn't incur turn cost per logic: "if (vx1 != 0 or vy1 != 0) and (vx2 != 0 or vy2 != 0)")
    # Accel:
    # v2 - v1 = (-1, 0) -> accel_cost1 = 1.0
    # v3 - v2 = (1, 0) -> accel_cost2 = 1.0
    # Total accel = 2.0
    # Total cost = 2.0 (move) + 0.0 (turn) + 2.0 (accel) = 4.0
    traj = [(0, 0, 0), (1, 0, 1), (1, 0, 2), (2, 0, 3)]
    cost = calculator.calculate_cost(traj)
    assert cost == 4.0
