import pytest
from environment.discrete_world import DiscreteWorld
from action.action_calculator import ActionCalculator
from search.search_algorithms import dynamic_programming_search, a_star_search, focal_search

@pytest.fixture
def world():
    w = DiscreteWorld(10, 10, 15)
    w.set_obstacle(3, 3)
    w.set_obstacle(3, 4)
    w.set_obstacle(4, 3)
    return w

@pytest.fixture
def calculator(world):
    return ActionCalculator(world, move_weight=1.0, turn_weight=0.25)

class MockInadmissibleHeuristic:
    def predict_cost(self, state, goal):
        # A bad heuristic that aggressively prefers moving right
        x, y, _, _, _ = state
        gx, gy = goal
        return (gx - x) * 0.1 + (gy - y) * 2.0

def test_focal_search_bound(world, calculator):
    start_state = (0, 0, 0, 0, 0)
    goal_state = (5, 5)
    max_depth = 12

    # Get exact optimal cost
    dp_res = dynamic_programming_search(world, calculator, start_state, goal_state, max_depth)
    opt_cost = dp_res.cost
    assert opt_cost != float('inf')

    heuristic = MockInadmissibleHeuristic()

    for w in [1.0, 1.2, 1.5, 2.0]:
        res = focal_search(world, calculator, heuristic, start_state, goal_state, max_depth, weight=w)
        assert res.cost <= w * opt_cost, f"Weight {w}: cost {res.cost} exceeded {w} * {opt_cost}"
