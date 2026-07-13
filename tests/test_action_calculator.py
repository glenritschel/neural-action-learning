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
    assert calculator.calculate_cost([(0, 0, 0, 0, 0)]) == 0.0

def test_single_state_obstacle(calculator):
    # Single state on obstacle
    assert calculator.calculate_cost([(5, 5, 0, 0, 0)]) == 100.0

def test_straight_movement(calculator):
    # Trajectory: (0,0)->(1,0)->(2,0)
    # State: (x, y, vx, vy, t)
    # Initial state velocity is typically (0,0)
    traj = [(0, 0, 0, 0, 0), (1, 0, 1, 0, 1), (2, 0, 1, 0, 2)]
    cost = calculator.calculate_cost(traj)
    # Step 1: (0,0,0,0,0) -> (1,0,1,0,1)
    # Move: 1.0, Accel: (1-0)^2 + (0-0)^2 = 1.0 -> step cost = 2.0
    # Step 2: (1,0,1,0,1) -> (2,0,1,0,2)
    # Move: 1.0, Accel: (1-1)^2 + (0-0)^2 = 0.0 -> step cost = 1.0
    # Total = 3.0
    assert cost == 3.0

def test_movement_with_turn(calculator):
    # Trajectory: (0,0)->(1,0)->(1,1)
    traj = [(0, 0, 0, 0, 0), (1, 0, 1, 0, 1), (1, 1, 0, 1, 2)]
    cost = calculator.calculate_cost(traj)
    # Step 1: (0,0,0,0,0) -> (1,0,1,0,1)
    # Move: 1.0, Accel: 1.0 -> 2.0
    # Step 2: (1,0,1,0,1) -> (1,1,0,1,2)
    # Move: 1.0, Turn: 0.25, Accel: (0-1)^2 + (1-0)^2 = 2.0 -> step cost = 3.25
    # Total = 5.25
    assert cost == 5.25

def test_movement_into_obstacle(calculator):
    # Trajectory: (4,5)->(5,5)
    traj = [(4, 5, 0, 0, 0), (5, 5, 1, 0, 1)]
    cost = calculator.calculate_cost(traj)
    # Step 1:
    # Obstacle: 100.0, Move: 1.0, Accel: 1.0 -> 102.0
    assert cost == 102.0

def test_stop_and_go(calculator):
    # Trajectory: (0,0)->(1,0)->(1,0)->(2,0)
    traj = [(0, 0, 0, 0, 0), (1, 0, 1, 0, 1), (1, 0, 0, 0, 2), (2, 0, 1, 0, 3)]
    cost = calculator.calculate_cost(traj)
    # Step 1: (0,0,0,0) -> (1,0,1,0) => Move: 1.0, Accel: 1.0 = 2.0
    # Step 2: (1,0,1,0) -> (1,0,0,0) => Move: 0.0, Accel: 1.0 = 1.0
    # Step 3: (1,0,0,0) -> (2,0,1,0) => Move: 1.0, Accel: 1.0 = 2.0
    # Total = 5.0
    assert cost == 5.0
