import pytest
from environment.discrete_world import DiscreteWorld, Action

def test_discrete_world_initialization():
    world = DiscreteWorld(10, 10, 20)
    assert world.grid_size_x == 10
    assert world.grid_size_y == 10
    assert world.time_steps == 20
    assert len(world.obstacles) == 0
    assert len(world.trajectories) == 0

def test_discrete_world_bounds():
    world = DiscreteWorld(20, 20, 40)
    assert world.is_within_bounds(0, 0)
    assert world.is_within_bounds(19, 19)
    assert not world.is_within_bounds(20, 20)
    assert not world.is_within_bounds(-1, 0)

def test_discrete_world_obstacles():
    world = DiscreteWorld(20, 20, 40)
    world.set_obstacle(5, 5)
    assert world.is_obstacle(5, 5)
    assert not world.is_obstacle(0, 0)

    with pytest.raises(ValueError):
        world.set_obstacle(25, 25)

def test_discrete_world_state_validity():
    world = DiscreteWorld(20, 20, 40)
    world.set_obstacle(2, 2)

    assert world.is_valid_state(0, 0, 0)
    assert not world.is_valid_state(2, 2, 0)  # Obstacle
    assert not world.is_valid_state(20, 0, 0) # Out of bounds x
    assert not world.is_valid_state(0, 0, 40) # Out of bounds t
    assert not world.is_valid_state(0, 0, -1) # Negative t

def test_discrete_world_trajectories():
    world = DiscreteWorld(20, 20, 40)
    traj = [(0, 0, 0), (0, 1, 1), (1, 1, 2)]
    world.add_trajectory(traj)
    assert len(world.trajectories) == 1
    assert world.trajectories[0] == traj

    # Invalid trajectory (out of bounds)
    invalid_traj = [(25, 0, 0)]
    with pytest.raises(ValueError):
        world.add_trajectory(invalid_traj)

    # Invalid trajectory (wrong tuple size)
    invalid_traj_2 = [(0, 0)]
    with pytest.raises(ValueError):
        world.add_trajectory(invalid_traj_2)

def test_action_enum():
    assert Action.UP.value == (0, 1)
    assert Action.DOWN.value == (0, -1)
    assert Action.LEFT.value == (-1, 0)
    assert Action.RIGHT.value == (1, 0)
    assert Action.STAY.value == (0, 0)
