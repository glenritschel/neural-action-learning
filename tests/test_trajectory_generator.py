import pytest
from environment.discrete_world import DiscreteWorld
from action.action_calculator import ActionCalculator
from search.trajectory_generator import TrajectoryGenerator

def test_trajectory_generator_depth_1():
    # Setup world and calculator
    world = DiscreteWorld(grid_size_x=5, grid_size_y=5, time_steps=5)
    calculator = ActionCalculator(world)
    generator = TrajectoryGenerator(world, calculator)

    # Start at (2, 2, 0, 0, 0). Depth = 1 move (so 2 states total).
    start_state = (2, 2, 0, 0, 0)
    results = generator.generate(start_state, depth=1)

    # Possible moves: UP, DOWN, LEFT, RIGHT, STAY -> 5 moves.
    # All are within bounds.
    assert len(results) == 5

    for r in results:
        traj = r["trajectory"]
        assert len(traj) == 2
        assert traj[0] == (2, 2, 0, 0, 0)
        assert r["metadata"]["length"] == 2
        assert r["metadata"]["start"] == (2, 2, 0, 0, 0)

        # Check endpoint
        assert r["endpoint"] == traj[-1]

        # Check time increment
        assert traj[-1][4] == 1

def test_trajectory_generator_bounds():
    # Setup world and calculator
    world = DiscreteWorld(grid_size_x=2, grid_size_y=2, time_steps=5)
    calculator = ActionCalculator(world)
    generator = TrajectoryGenerator(world, calculator)

    # Start at corner (0, 0, 0, 0, 0).
    start_state = (0, 0, 0, 0, 0)
    results = generator.generate(start_state, depth=1)

    # Valid moves from (0,0): UP(0,1), RIGHT(1,0), STAY(0,0)
    # DOWN and LEFT go out of bounds.
    assert len(results) == 3
    endpoints = set([(r["endpoint"][0], r["endpoint"][1]) for r in results])
    assert endpoints == {(0, 1), (1, 0), (0, 0)}

def test_trajectory_generator_obstacles():
    # Setup world and calculator
    world = DiscreteWorld(grid_size_x=3, grid_size_y=3, time_steps=5)
    world.set_obstacle(1, 1)
    calculator = ActionCalculator(world)
    generator = TrajectoryGenerator(world, calculator)

    # Start at (0, 1, 0, 0, 0)
    start_state = (0, 1, 0, 0, 0)
    results = generator.generate(start_state, depth=1)

    # Now that we use soft obstacles, all 4 within-bounds actions are valid states
    # Valid from (0,1): UP(0,2), DOWN(0,0), LEFT(-1,1 - OOB), RIGHT(1,1 - OBSTACLE), STAY(0,1)
    assert len(results) == 4
    endpoints = set([(r["endpoint"][0], r["endpoint"][1]) for r in results])
    assert (1, 1) in endpoints

def test_trajectory_generator_cost_integration():
    world = DiscreteWorld(grid_size_x=5, grid_size_y=5, time_steps=5)
    # Setting weights to verify they are computed
    calculator = ActionCalculator(world, move_weight=1.0, turn_weight=0.0, obstacle_weight=0.0, acceleration_weight=0.0)
    generator = TrajectoryGenerator(world, calculator)

    start_state = (2, 2, 0, 0, 0)
    results = generator.generate(start_state, depth=1)

    # For depth 1, moves will cost 1.0, stay will cost 0.0
    costs = [r["cost"] for r in results]
    assert costs.count(1.0) == 4 # 4 moves
    assert costs.count(0.0) == 1 # 1 stay
