import pytest
from environment.discrete_world import DiscreteWorld
from action.action_calculator import ActionCalculator
from search.search_algorithms import brute_force_search, dynamic_programming_search, a_star_search

@pytest.fixture
def world():
    w = DiscreteWorld(5, 5, 8)
    w.set_obstacle(2, 2)
    return w

@pytest.fixture
def calculator(world):
    return ActionCalculator(world)

@pytest.fixture
def mock_heuristic():
    class MockHeuristic:
        def predict_cost(self, state, goal):
            return 0.0 # admissible
    return MockHeuristic()

def test_search_correctness(world, calculator, mock_heuristic):
    # Test on a small instance (T <= 6) to ensure DP == BF == A*
    start_state = (0, 0, 0, 0, 0)
    goal_state = (3, 3) # Solvable in 6 steps
    max_depth = 6

    bf_res = brute_force_search(world, calculator, start_state, goal_state, max_depth)
    dp_res = dynamic_programming_search(world, calculator, start_state, goal_state, max_depth)
    astar_res = a_star_search(world, calculator, mock_heuristic, start_state, goal_state, max_depth)

    assert bf_res.cost != float('inf')
    assert bf_res.cost == dp_res.cost
    assert dp_res.cost == astar_res.cost

def test_search_unsolvable(world, calculator, mock_heuristic):
    start_state = (0, 0, 0, 0, 0)
    goal_state = (4, 4)
    max_depth = 4 # Not solvable in 4 steps

    bf_res = brute_force_search(world, calculator, start_state, goal_state, max_depth)
    dp_res = dynamic_programming_search(world, calculator, start_state, goal_state, max_depth)
    astar_res = a_star_search(world, calculator, mock_heuristic, start_state, goal_state, max_depth)

    assert bf_res.cost == float('inf')
    assert dp_res.cost == float('inf')
    assert astar_res.cost == float('inf')
