import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable, Tuple, List, Set, Dict
from functools import wraps

from environment.discrete_world import DiscreteWorld, Action
from action.action_calculator import ActionCalculator

@dataclass
class SearchResult:
    path: list
    cost: float
    nodes_expanded: int
    runtime: float
    memory_footprint: int

def track_metrics(search_func: Callable) -> Callable:
    """Decorator to track runtime execution speed, memory footprint, nodes expanded, and total solution quality."""
    @wraps(search_func)
    def wrapper(*args, **kwargs) -> SearchResult:
        tracemalloc.start()
        start_time = time.time()

        # The search function should return (path, cost, nodes_expanded)
        path, cost, nodes_expanded = search_func(*args, **kwargs)

        runtime = time.time() - start_time
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return SearchResult(
            path=path,
            cost=cost,
            nodes_expanded=nodes_expanded,
            runtime=runtime,
            memory_footprint=peak_memory
        )
    return wrapper

@track_metrics
def brute_force_search(world: DiscreteWorld, calculator: ActionCalculator, start_state: Tuple[int, int, int, int, int], goal_state: Tuple[int, int], max_depth: int) -> Tuple[List[Tuple[int, int, int, int, int]], float, int]:
    """Exhaustive DFS to find the optimal path to the goal."""
    best_path = []
    best_cost = float('inf')
    nodes_expanded = 0

    def dfs(current_path: List[Tuple[int, int, int, int, int]]):
        nonlocal best_path, best_cost, nodes_expanded

        current_state = current_path[-1]
        x, y, vx, vy, t = current_state

        nodes_expanded += 1

        if (x, y) == goal_state:
            cost = calculator.calculate_cost(current_path)
            if cost < best_cost:
                best_cost = cost
                best_path = list(current_path)
            return

        if len(current_path) - 1 >= max_depth or t + 1 >= world.time_steps:
            return

        next_t = t + 1
        for action in Action:
            dx, dy = action.value
            nx, ny = x + dx, y + dy

            if world.is_valid_state(nx, ny, next_t):
                next_state = (nx, ny, dx, dy, next_t)
                current_path.append(next_state)
                dfs(current_path)
                current_path.pop()

    dfs([start_state])
    return best_path, best_cost, nodes_expanded

from search.heuristic import MLPHeuristic
import heapq

@track_metrics
def focal_search(world: DiscreteWorld, calculator: ActionCalculator, heuristic: Any,
                 start_state, goal_state, max_depth: int, weight: float = 1.5):
    """
    A*_epsilon (Focal Search). OPEN is ordered by the ADMISSIBLE score f_adm = g + h_adm.
    FOCAL = { n in OPEN : f_adm(n) <= weight * min_OPEN f_adm }. Within FOCAL we expand the
    node with the smallest LEARNED heuristic h_learn. Because every expanded node satisfies
    f_adm <= weight * f_min <= weight * C*, and a goal node has h_adm = 0 (so f_adm = g),
    the returned path cost g <= weight * C*. This is the real bound the summed f never had.
    """
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
    nodes_expanded = 0

    while open_list:
        f_min = min(e[0] for e in open_list)
        focal = [e for e in open_list if e[0] <= weight * f_min]
        chosen = min(focal, key=lambda e: h_learn(e[3], goal_state))  # learned ordering
        open_list.remove(chosen)

        f_adm, _, g_score, current_state, path = chosen
        nodes_expanded += 1
        x, y, vx, vy, t = current_state

        if (x, y) == goal_state:
            return path, g_score, nodes_expanded
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
    return [], float('inf'), nodes_expanded

@track_metrics
def a_star_search(world: DiscreteWorld, calculator: ActionCalculator, heuristic: Any, start_state: Tuple[int, int, int, int, int], goal_state: Tuple[int, int], max_depth: int, weight: float = 1.0) -> Tuple[List[Tuple[int, int, int, int, int]], float, int]:
    """
    Standard A* Search.
    Computes f = g + h_adm.
    """
    # Priority queue stores tuples of (f_score, tie_breaker, g_score, current_state, path)
    counter = 0

    def h_adm(state, goal):
        # True admissible lower bound: min_move_cost * manhattan distance
        x, y, _, _, _ = state
        gx, gy = goal
        return float(abs(gx - x) + abs(gy - y)) * calculator.move_weight

    start_g_score = 0.0
    adm_h = h_adm(start_state, goal_state)

    start_f_score = start_g_score + adm_h

    pq = [(start_f_score, counter, start_g_score, start_state, [start_state])]
    counter += 1

    nodes_expanded = 0
    visited = {} # to track best g_score for the full Markov state

    while pq:
        f_score, _, g_score, current_state, path = heapq.heappop(pq)

        nodes_expanded += 1

        x, y, vx, vy, t = current_state
        if (x, y) == goal_state:
            return path, g_score, nodes_expanded

        if len(path) - 1 >= max_depth or t + 1 >= world.time_steps:
            continue

        state_key = current_state

        if state_key in visited and visited[state_key] <= g_score:
            continue
        visited[state_key] = g_score

        next_t = t + 1
        for action in Action:
            dx, dy = action.value
            nx, ny = x + dx, y + dy
            next_state = (nx, ny, dx, dy, next_t)

            if world.is_valid_state(nx, ny, next_t):
                # Calculate cost of adding this step incrementally
                step_cost = calculator.calculate_transition_cost(current_state, next_state)

                next_g_score = g_score + step_cost
                next_adm_h = h_adm(next_state, goal_state)
                next_f_score = next_g_score + next_adm_h

                heapq.heappush(pq, (next_f_score, counter, next_g_score, next_state, path + [next_state]))
                counter += 1

    return [], float('inf'), nodes_expanded

@track_metrics
def beam_search(world: DiscreteWorld, calculator: ActionCalculator, heuristic: MLPHeuristic, start_state: Tuple[int, int, int, int, int], goal_state: Tuple[int, int], max_depth: int, beam_width: int = 5) -> Tuple[List[Tuple[int, int, int, int, int]], float, int]:
    """Beam Search maintaining only top-K candidates based on path cost and heuristic."""
    # Beam contains tuples of (f_score, path, g_score)
    start_h = heuristic.predict_cost(start_state, goal_state)
    beam = [(start_h, [start_state], 0.0)]
    nodes_expanded = 0

    best_path = []
    best_cost = float('inf')

    for depth in range(max_depth):
        if not beam:
            break

        next_beam = []

        for f_score, path, g_score in beam:
            current_state = path[-1]
            x, y, vx, vy, t = current_state

            nodes_expanded += 1

            if (x, y) == goal_state:
                if g_score < best_cost:
                    best_cost = g_score
                    best_path = list(path)
                continue

            if t + 1 >= world.time_steps:
                continue

            next_t = t + 1
            for action in Action:
                dx, dy = action.value
                nx, ny = x + dx, y + dy
                next_state = (nx, ny, dx, dy, next_t)

                if world.is_valid_state(nx, ny, next_t):
                    step_cost = calculator.calculate_transition_cost(current_state, next_state)

                    next_g_score = g_score + step_cost
                    next_h_score = heuristic.predict_cost(next_state, goal_state)
                    next_f_score = next_g_score + next_h_score

                    next_beam.append((next_f_score, path + [next_state], next_g_score))

        # Sort and prune beam
        next_beam.sort(key=lambda item: item[0])
        beam = next_beam[:beam_width]

    return best_path, best_cost, nodes_expanded

import math
import random

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0 # Will store negative cost (since we want to minimize cost)
        self.untried_actions = list(Action)

    def ucb1(self, exploration_weight=1.414):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

@track_metrics
def mcts_search(world: DiscreteWorld, calculator: ActionCalculator, start_state: Tuple[int, int, int, int, int], goal_state: Tuple[int, int], max_depth: int, num_simulations: int = 1000) -> Tuple[List[Tuple[int, int, int, int, int]], float, int]:
    """Monte Carlo Tree Search for path finding."""
    nodes_expanded = 0
    root = MCTSNode(start_state)

    # Run simulations
    for _ in range(num_simulations):
        node = root

        # 1. Selection
        while not node.untried_actions and node.children:
            node = max(node.children, key=lambda c: c.ucb1())

        # 2. Expansion
        if node.untried_actions:
            action = node.untried_actions.pop(0) # or random.choice(node.untried_actions)
            x, y, vx, vy, t = node.state
            dx, dy = action.value
            next_t = t + 1
            nx, ny = x + dx, y + dy

            if world.is_valid_state(nx, ny, next_t):
                next_state = (nx, ny, dx, dy, next_t)
                child = MCTSNode(next_state, parent=node, action=action)
                node.children.append(child)
                node = child
                nodes_expanded += 1

        # 3. Rollout
        current_rollout_state = node.state
        rollout_path = [current_rollout_state]

        # Backtrack to build the path from root for accurate cost calculation
        temp_node = node
        path_prefix = []
        while temp_node.parent is not None:
            path_prefix.append(temp_node.state)
            temp_node = temp_node.parent
        path_prefix.append(root.state)
        path_prefix.reverse()

        rollout_path = path_prefix.copy()

        depth_in_rollout = len(rollout_path) - 1
        x, y, vx, vy, t = current_rollout_state

        while (x, y) != goal_state and depth_in_rollout < max_depth and t + 1 < world.time_steps:
            valid_actions = []
            next_t = t + 1
            for a in Action:
                dx, dy = a.value
                nx, ny = x + dx, y + dy
                if world.is_valid_state(nx, ny, next_t):
                    valid_actions.append((nx, ny, dx, dy, next_t))

            if not valid_actions:
                break

            next_state = random.choice(valid_actions)
            rollout_path.append(next_state)
            current_rollout_state = next_state
            x, y, vx, vy, t = current_rollout_state
            depth_in_rollout += 1

        # Calculate cost. Reward is negative cost.
        # Add a large penalty if goal not reached.
        if (x, y) == goal_state:
            cost = calculator.calculate_cost(rollout_path)
            reward = -cost
        else:
            # Penalize heavily based on distance to goal
            dist = abs(goal_state[0] - x) + abs(goal_state[1] - y)
            reward = -(1000 + dist * 10)

        # 4. Backpropagation
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    # Extract best path
    best_path = [root.state]
    node = root
    while node.children:
        # Choose child with max visits or max average value
        node = max(node.children, key=lambda c: c.visits)
        best_path.append(node.state)
        if (node.state[0], node.state[1]) == goal_state:
            break

    final_cost = calculator.calculate_cost(best_path)
    return best_path, final_cost, nodes_expanded

@track_metrics
def dynamic_programming_search(world: DiscreteWorld, calculator: ActionCalculator, start_state: Tuple[int, int, int, int, int], goal_state: Tuple[int, int], max_depth: int) -> Tuple[List[Tuple[int, int, int, int, int]], float, int]:
    """Dynamic programming (memoized DFS) over the DAG to find the optimal path to the goal."""
    # Since cost depends on history (velocity/acceleration), we need to memoize based on the full Markov state
    # Memoization table: current_state -> (best_cost_from_here, path_from_here)
    memo = {}
    nodes_expanded = 0

    def dfs(current_path: List[Tuple[int, int, int, int, int]]) -> Tuple[float, List[Tuple[int, int, int, int, int]]]:
        nonlocal nodes_expanded

        current_state = current_path[-1]

        x, y, vx, vy, t = current_state
        nodes_expanded += 1

        if (x, y) == goal_state:
            return 0.0, [current_state]

        if len(current_path) - 1 >= max_depth or t + 1 >= world.time_steps:
            return float('inf'), []

        state_key = current_state
        if state_key in memo:
            return memo[state_key]

        best_cost_from_here = float('inf')
        best_path_from_here = []

        next_t = t + 1
        for action in Action:
            dx, dy = action.value
            nx, ny = x + dx, y + dy

            next_state = (nx, ny, dx, dy, next_t)
            if world.is_valid_state(nx, ny, next_t):
                step_cost = calculator.calculate_transition_cost(current_state, next_state)

                current_path.append(next_state)
                future_cost, future_path = dfs(current_path)
                current_path.pop()

                total_cost = step_cost + future_cost
                if total_cost < best_cost_from_here:
                    best_cost_from_here = total_cost
                    best_path_from_here = [current_state] + future_path

        memo[state_key] = (best_cost_from_here, best_path_from_here)
        return best_cost_from_here, best_path_from_here

    best_cost_from_start, optimal_path = dfs([start_state])

    # Calculate the actual total cost
    final_cost = calculator.calculate_cost(optimal_path) if optimal_path else float('inf')

    return optimal_path, final_cost, nodes_expanded
