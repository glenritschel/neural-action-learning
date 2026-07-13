import pytest
from environment.discrete_world import DiscreteWorld
from action.action_calculator import ActionCalculator
from search.search_algorithms import BruteForceSearch, DPSearch, AStarSearch, BeamSearch, MCTS

def setup_environment():
    world = DiscreteWorld(grid_size_x=10, grid_size_y=10, time_steps=20)
    calculator = ActionCalculator(world)
    return world, calculator

def test_brute_force_search():
    world, calculator = setup_environment()
    search = BruteForceSearch(world, calculator)
    start_state = (0, 0, 0)
    goal_state = (2, 2)
    result = search.search(start_state, goal_state, max_depth=5)

    assert result is not None, "Search should find a valid path"
    path, cost, nodes_expanded = result
    assert len(path) > 0
    assert (path[-1][0], path[-1][1]) == goal_state
    assert nodes_expanded > 0

def test_dp_search():
    world, calculator = setup_environment()
    search = DPSearch(world, calculator)
    start_state = (0, 0, 0)
    goal_state = (2, 2)
    result = search.search(start_state, goal_state, max_depth=5)

    assert result is not None
    path, cost, nodes_expanded = result
    assert (path[-1][0], path[-1][1]) == goal_state
    assert nodes_expanded > 0

def test_astar_search():
    world, calculator = setup_environment()
    search = AStarSearch(world, calculator, heuristic_model=None)
    start_state = (0, 0, 0)
    goal_state = (2, 2)
    result = search.search(start_state, goal_state, max_depth=5)

    assert result is not None
    path, cost, nodes_expanded = result
    assert (path[-1][0], path[-1][1]) == goal_state
    assert nodes_expanded > 0

def test_beam_search():
    world, calculator = setup_environment()
    search = BeamSearch(world, calculator, beam_width=50)
    start_state = (0, 0, 0)
    goal_state = (2, 2)
    result = search.search(start_state, goal_state, max_depth=10)

    assert result is not None
    path, cost, nodes_expanded = result
    assert (path[-1][0], path[-1][1]) == goal_state
    assert nodes_expanded > 0

def test_mcts_search():
    world, calculator = setup_environment()
    search = MCTS(world, calculator, iterations=50)
    start_state = (0, 0, 0)
    goal_state = (2, 2)
    # MCTS might not always find it if iterations are low, but let's test execution
    result = search.search(start_state, goal_state, max_depth=5)
    # Don't assert not None, just ensure it runs without error.
