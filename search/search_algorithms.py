import heapq
import math
from typing import List, Tuple, Dict, Any, Optional
import torch

from environment.discrete_world import DiscreteWorld, Action
from action.action_calculator import ActionCalculator
from models.mlp import ActionMLP

class BaseSearch:
    def __init__(self, world: DiscreteWorld, calculator: ActionCalculator):
        self.world = world
        self.calculator = calculator

    def _get_neighbors(self, state: Tuple[int, int, int]) -> List[Tuple[Tuple[int, int, int], Action]]:
        x, y, t = state
        next_t = t + 1
        neighbors = []
        for action in Action:
            dx, dy = action.value
            nx, ny = x + dx, y + dy
            if self.world.is_valid_state(nx, ny, next_t):
                neighbors.append(((nx, ny, next_t), action))
        return neighbors

class BruteForceSearch(BaseSearch):
    def search(self, start_state: Tuple[int, int, int], goal_state: Tuple[int, int, int], max_depth: int) -> Optional[Tuple[List[Tuple[int, int, int]], float, int]]:
        """DFS that explores all paths up to max_depth."""
        best_path = None
        best_cost = float('inf')
        nodes_expanded = 0

        def dfs(current_state: Tuple[int, int, int], current_path: List[Tuple[int, int, int]]):
            nonlocal best_path, best_cost, nodes_expanded
            nodes_expanded += 1

            if (current_state[0], current_state[1]) == (goal_state[0], goal_state[1]):
                cost = self.calculator.calculate_cost(current_path)
                if cost < best_cost:
                    best_cost = cost
                    best_path = list(current_path)
                return

            if len(current_path) > max_depth:
                return

            for neighbor_state, _ in self._get_neighbors(current_state):
                current_path.append(neighbor_state)
                dfs(neighbor_state, current_path)
                current_path.pop()

        if self.world.is_valid_state(start_state[0], start_state[1], start_state[2]):
            dfs(start_state, [start_state])

        if best_path:
            return best_path, best_cost, nodes_expanded
        return None

class DPSearch(BaseSearch):
    def search(self, start_state: Tuple[int, int, int], goal_state: Tuple[int, int, int], max_depth: int) -> Optional[Tuple[List[Tuple[int, int, int]], float, int]]:
        """DP using memoization to find min cost path."""
        # For DP to work with action costs, state needs to include the last velocity to compute turn/acceleration costs.
        # This makes the state space large, but we'll approximate/simplify by just storing the min cost to reach each (x,y,t) and the path.

        # We will implement DP via Value Iteration / BFS with a cost table to be more rigorous.

        cost_to_reach = {start_state: 0.0}
        paths = {start_state: [start_state]}

        queue = [start_state]

        best_goal_cost = float('inf')
        best_goal_path = None
        nodes_expanded = 0

        while queue:
            current_state = queue.pop(0)
            nodes_expanded += 1
            current_path = paths[current_state]

            if len(current_path) > max_depth:
                continue

            if (current_state[0], current_state[1]) == (goal_state[0], goal_state[1]):
                c = self.calculator.calculate_cost(current_path)
                if c < best_goal_cost:
                    best_goal_cost = c
                    best_goal_path = current_path
                continue

            for neighbor_state, _ in self._get_neighbors(current_state):
                new_path = current_path + [neighbor_state]
                new_cost = self.calculator.calculate_cost(new_path)

                # We need to distinguish states by their history for turning/accel costs.
                # A simpler DP state for the grid:
                if neighbor_state not in cost_to_reach or new_cost < cost_to_reach[neighbor_state]:
                    cost_to_reach[neighbor_state] = new_cost
                    paths[neighbor_state] = new_path
                    queue.append(neighbor_state)

        if best_goal_path:
            return best_goal_path, best_goal_cost, nodes_expanded
        return None

class AStarSearch(BaseSearch):
    def __init__(self, world: DiscreteWorld, calculator: ActionCalculator, heuristic_model: Optional[ActionMLP] = None):
        super().__init__(world, calculator)
        self.heuristic_model = heuristic_model

    def _heuristic(self, state: Tuple[int, int, int], goal_state: Tuple[int, int, int]) -> float:
        if self.heuristic_model is None:
            # Manhattan distance + time difference as fallback
            return abs(state[0] - goal_state[0]) + abs(state[1] - goal_state[1])

        # If we have an MLP model, use it to predict cost.
        # We need to construct the feature vector expected by MLP.
        # In dataset_builder, it expects normalized [nx, ny, nt] for the whole trajectory.
        # Since we just want a heuristic, we can pass a dummy trajectory of length 2: [state, goal_state]
        # and get the predicted cost.

        # Determine the grid size from the world
        gx, gy, gt = self.world.grid_size_x, self.world.grid_size_y, self.world.time_steps

        nx1, ny1, nt1 = state[0]/gx, state[1]/gy, state[2]/gt
        nx2, ny2, nt2 = goal_state[0]/gx, goal_state[1]/gy, min(state[2] + abs(state[0]-goal_state[0]) + abs(state[1]-goal_state[1]), gt-1)/gt

        features = [nx1, ny1, nt1, nx2, ny2, nt2]

        # Pad features if model expects a fixed trajectory length.
        # Assuming the model was trained on a fixed trajectory length, say max_depth=10,
        # meaning 11 states = 33 features. We pad with the goal state.
        input_dim = self.heuristic_model.network[0].in_features

        while len(features) < input_dim:
            features.extend([nx2, ny2, nt2])

        features = features[:input_dim] # Ensure exact size

        x_tensor = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            pred_cost = self.heuristic_model(x_tensor).item()

        return pred_cost

    def search(self, start_state: Tuple[int, int, int], goal_state: Tuple[int, int, int], max_depth: int) -> Optional[Tuple[List[Tuple[int, int, int]], float, int]]:
        # Priority queue stores tuples of (f_score, id, current_state, path)
        # id is used to break ties when f_scores are equal
        count = 0
        nodes_expanded = 0
        open_set = []
        heapq.heappush(open_set, (0, count, start_state, [start_state]))

        # g_score maps path tuple to cost.
        # Because costs depend on history (velocity), state alone is not enough.
        # But for A*, we can map state to best g_score and keep paths.
        g_score = {start_state: 0.0}

        while open_set:
            _, _, current_state, current_path = heapq.heappop(open_set)
            nodes_expanded += 1

            if (current_state[0], current_state[1]) == (goal_state[0], goal_state[1]):
                return current_path, self.calculator.calculate_cost(current_path), nodes_expanded

            if len(current_path) > max_depth:
                continue

            for neighbor_state, _ in self._get_neighbors(current_state):
                new_path = current_path + [neighbor_state]
                tentative_g_score = self.calculator.calculate_cost(new_path)

                # A heuristic approach: if we found a cheaper way to reach this state
                if neighbor_state not in g_score or tentative_g_score < g_score[neighbor_state]:
                    g_score[neighbor_state] = tentative_g_score
                    f_score = tentative_g_score + self._heuristic(neighbor_state, goal_state)
                    count += 1
                    heapq.heappush(open_set, (f_score, count, neighbor_state, new_path))

        return None

class BeamSearch(BaseSearch):
    def __init__(self, world: DiscreteWorld, calculator: ActionCalculator, beam_width: int = 3):
        super().__init__(world, calculator)
        self.beam_width = beam_width

    def search(self, start_state: Tuple[int, int, int], goal_state: Tuple[int, int, int], max_depth: int) -> Optional[Tuple[List[Tuple[int, int, int]], float, int]]:
        beam = [(start_state, [start_state], 0.0)] # (state, path, cost)
        nodes_expanded = 0

        for _ in range(max_depth):
            next_beam = []
            for state, path, _ in beam:
                nodes_expanded += 1
                if (state[0], state[1]) == (goal_state[0], goal_state[1]):
                    # Found goal, could return here or keep searching
                    return path, self.calculator.calculate_cost(path), nodes_expanded

                for neighbor_state, _ in self._get_neighbors(state):
                    new_path = path + [neighbor_state]
                    new_cost = self.calculator.calculate_cost(new_path)

                    # Heuristic sorting: use cost to reach so far plus manhattan distance
                    score = new_cost + abs(neighbor_state[0] - goal_state[0]) + abs(neighbor_state[1] - goal_state[1])
                    next_beam.append((neighbor_state, new_path, score))

            if not next_beam:
                break

            # Keep top k
            next_beam.sort(key=lambda x: x[2])
            beam = next_beam[:self.beam_width]

            # Check if any in beam reached goal
            for state, path, _ in beam:
                if (state[0], state[1]) == (goal_state[0], goal_state[1]):
                    return path, self.calculator.calculate_cost(path), nodes_expanded

        return None

class MCTSNode:
    def __init__(self, state: Tuple[int, int, int], parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action # Action taken to reach this state
        self.children = []
        self.visits = 0
        self.value = 0.0 # We will accumulate negative cost (reward) here
        self.untried_actions = None # To be populated on expansion

class MCTS(BaseSearch):
    def __init__(self, world: DiscreteWorld, calculator: ActionCalculator, iterations: int = 100):
        super().__init__(world, calculator)
        self.iterations = iterations
        self.c_param = 1.41 # Exploration parameter

    def search(self, start_state: Tuple[int, int, int], goal_state: Tuple[int, int, int], max_depth: int) -> Optional[Tuple[List[Tuple[int, int, int]], float, int]]:
        import random
        import math

        root = MCTSNode(start_state)
        best_overall_path = None
        best_overall_cost = float('inf')
        nodes_expanded = 0

        def get_untried_actions(state):
            return self._get_neighbors(state)

        root.untried_actions = get_untried_actions(root.state)

        def select(node):
            # Select child with highest UCB1
            best_score = -float('inf')
            best_child = None
            for child in node.children:
                if child.visits == 0:
                    score = float('inf')
                else:
                    exploit = child.value / child.visits
                    explore = self.c_param * math.sqrt(2 * math.log(node.visits) / child.visits)
                    score = exploit + explore
                if score > best_score:
                    best_score = score
                    best_child = child
            return best_child

        def expand(node):
            if not node.untried_actions:
                return None

            # Pop an untried action
            idx = random.randint(0, len(node.untried_actions)-1)
            next_state, action = node.untried_actions.pop(idx)

            child = MCTSNode(next_state, parent=node, action=action)
            child.untried_actions = get_untried_actions(child.state)
            node.children.append(child)
            return child

        def simulate(node):
            nonlocal best_overall_path, best_overall_cost, nodes_expanded

            # Reconstruct path to current node
            curr_node = node
            path = []
            while curr_node:
                path.append(curr_node.state)
                curr_node = curr_node.parent
            path.reverse()

            curr_state = node.state
            depth = len(path) - 1

            # Random rollout
            while depth < max_depth:
                nodes_expanded += 1
                if (curr_state[0], curr_state[1]) == (goal_state[0], goal_state[1]):
                    cost = self.calculator.calculate_cost(path)
                    if cost < best_overall_cost:
                        best_overall_cost = cost
                        best_overall_path = list(path)
                    return -cost # Reward is negative cost

                neighbors = self._get_neighbors(curr_state)
                if not neighbors:
                    break

                curr_state, _ = random.choice(neighbors)
                path.append(curr_state)
                depth += 1

            return -float('inf') # Failed to reach goal

        def backpropagate(node, reward):
            while node:
                node.visits += 1
                node.value += reward
                node = node.parent

        for _ in range(self.iterations):
            node = root

            # Selection
            while node and not node.untried_actions and node.children:
                node = select(node)

            # Expansion
            if node and node.untried_actions:
                node = expand(node)
                nodes_expanded += 1

            # Simulation
            if node:
                reward = simulate(node)
                # Backpropagation
                backpropagate(node, reward)

        if best_overall_path:
            return best_overall_path, best_overall_cost, nodes_expanded

        return None
