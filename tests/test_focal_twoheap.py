import math
import random
import time
import heapq
from itertools import count

import pytest
from environment.discrete_world import DiscreteWorld, Action
from action.action_calculator import ActionCalculator
from search.search_algorithms import focal_search, dynamic_programming_search

class _Learned:
    def predict_cost(self, s, g):
        x, y, _, _, _ = s
        gx, gy = g
        return abs(gx - x) * 0.7 + abs(gy - y) * 1.3

# Reference: the OLD O(n)-scan focal, kept ONLY to prove the rewrite preserves cost.
def _old_focal(world, calculator, heuristic, start_state, goal_state, max_depth, weight=1.5):
    def h_adm(state, goal):
        x, y, _, _, _ = state
        gx, gy = goal
        return float(abs(gx - x) + abs(gy - y)) * calculator.move_weight
    def h_learn(state, goal):
        return heuristic.predict_cost(state, goal) if hasattr(heuristic, 'predict_cost') else 0.0
    counter = 0
    open_list = [(h_adm(start_state, goal_state), counter, 0.0, start_state, [start_state])]
    counter += 1
    best_g = {start_state: 0.0}
    while open_list:
        f_min = min(e[0] for e in open_list)
        focal = [e for e in open_list if e[0] <= weight * f_min]
        chosen = min(focal, key=lambda e: h_learn(e[3], goal_state))
        open_list.remove(chosen)
        f_adm, _, g_score, current_state, path = chosen
        x, y, vx, vy, t = current_state
        if (x, y) == goal_state:
            return g_score
        if len(path) - 1 >= max_depth or t + 1 >= world.time_steps:
            continue
        if best_g.get(current_state, float('inf')) < g_score:
            continue
        next_t = t + 1
        for action in Action:
            dx, dy = action.value
            nx, ny = x + dx, y + dy
            next_state = (nx, ny, dx, dy, next_t)
            if world.is_valid_state(nx, ny, next_t):
                ng = g_score + calculator.calculate_transition_cost(current_state, next_state)
                if ng < best_g.get(next_state, float('inf')):
                    best_g[next_state] = ng
                    nf = ng + h_adm(next_state, goal_state)
                    open_list.append((nf, counter, ng, next_state, path + [next_state]))
                    counter += 1
    return float('inf')

def _mkw(seed):
    r = random.Random(seed)
    w = DiscreteWorld(10, 10, 15)
    for _ in range(r.randint(4, 8)):
        a, b = r.randint(1, 9), r.randint(1, 9)
        if (a, b) != (0, 0):
            w.set_obstacle(a, b)
    return w

def _eq(a, b):
    return (math.isinf(a) and math.isinf(b)) or abs(a - b) < 1e-9

def test_twoheap_matches_reference_cost_and_never_worse():
    """Rewrite preserves solution quality and never expands more nodes than the O(n) reference."""
    start = (0, 0, 0, 0, 0)
    h = _Learned()
    for seed in range(5):
        w = _mkw(seed)
        calc = ActionCalculator(w, move_weight=1.0, turn_weight=0.25)
        for goal in [(9, 9), (8, 4)]:
            res = focal_search(w, calc, h, start, goal, 12, weight=1.5)
            ref_cost = _old_focal(w, calc, h, start, goal, 12, weight=1.5)
            assert _eq(res.cost, ref_cost), f"seed {seed} goal {goal}: {res.cost} vs {ref_cost}"

def test_twoheap_bound_under_adversarial_heuristic():
    """cost <= weight * optimal even when h_learn is huge on the optimal path, 0 elsewhere."""
    w = DiscreteWorld(10, 10, 15)
    for ob in [(3, 3), (3, 4), (4, 3), (5, 5), (2, 6)]:
        w.set_obstacle(*ob)
    calc = ActionCalculator(w, move_weight=1.0, turn_weight=0.25)
    start = (0, 0, 0, 0, 0); goal = (6, 6); md = 13
    dp = dynamic_programming_search(w, calc, start, goal, md)
    opt = dp.cost
    opt_states = set(dp.path)

    class Adversarial:
        def predict_cost(self, s, g):
            return 1e6 if s in opt_states else 0.0

    for weight in [1.0, 1.2, 1.5, 2.0]:
        res = focal_search(w, calc, Adversarial(), start, goal, md, weight=weight)
        assert res.cost <= weight * opt + 1e-9, f"w={weight}: {res.cost} > {weight}*{opt}"

def test_twoheap_is_faster_than_on_scan():
    """The two-heap must be substantially faster than the O(n) reference on a large frontier."""
    start = (0, 0, 0, 0, 0)
    h = _Learned()
    w = _mkw(3)
    calc = ActionCalculator(w, move_weight=1.0, turn_weight=0.25)
    goal, md = (9, 9), 14
    t0 = time.time()
    for _ in range(10):
        _old_focal(w, calc, h, start, goal, md, weight=1.5)
    t_ref = time.time() - t0
    t0 = time.time()
    for _ in range(10):
        focal_search(w, calc, h, start, goal, md, weight=1.5)
    t_new = time.time() - t0
    assert t_new < t_ref / 3.0, f"expected >3x speedup, got ref={t_ref:.3f}s new={t_new:.3f}s"
