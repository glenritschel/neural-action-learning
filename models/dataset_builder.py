import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any

class TrajectoryDataset(Dataset):
    def __init__(self, features: torch.Tensor, costs: torch.Tensor):
        self.features = features
        self.costs = costs

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.costs[idx]

class DatasetBuilder:
    def __init__(self, grid_size_x: int, grid_size_y: int, time_steps: int):
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        self.time_steps = time_steps

    def build_features(self, state: tuple, goal_state: tuple, world_time_steps: int = None) -> List[float]:
        """
        Builds a fixed-size feature vector for a state-to-goal prediction.
        state is (x, y, vx, vy, t)
        goal_state is (gx, gy)

        Returns: [nx, ny, nvx, nvy, nt, ngx, ngy, steps_remaining_norm, dx_norm, dy_norm, manhattan_norm]
        """
        x, y, vx, vy, t = state
        gx, gy = goal_state

        # Determine time steps for normalization
        max_t = float(world_time_steps) if world_time_steps is not None else float(self.time_steps)
        if max_t == 0: max_t = 1.0

        nx = x / float(self.grid_size_x)
        ny = y / float(self.grid_size_y)

        # Max velocity in discrete grid is usually 1, so divide by 1.0 or just keep it
        # Assuming velocity is in [-1, 0, 1]
        nvx = float(vx)
        nvy = float(vy)

        nt = t / max_t

        ngx = gx / float(self.grid_size_x)
        ngy = gy / float(self.grid_size_y)

        steps_remaining = max_t - t
        steps_remaining_norm = steps_remaining / max_t

        dx = (gx - x) / float(self.grid_size_x)
        dy = (gy - y) / float(self.grid_size_y)

        manhattan_dist = abs(gx - x) + abs(gy - y)
        manhattan_norm = manhattan_dist / float(self.grid_size_x + self.grid_size_y)

        return [nx, ny, nvx, nvy, nt, ngx, ngy, steps_remaining_norm, dx, dy, manhattan_norm]

    def generate_cost_to_go_dataset(self, world, calculator, num_instances: int, max_depth: int, train_goals: List[tuple] = None) -> pd.DataFrame:
        """
        Uses exact A* search to solve instances and generate true cost-to-go labels
        for states on the optimal path and sampled off-path states.
        Holds out specific goals to avoid data leakage.
        """
        from search.search_algorithms import a_star_search
        import random

        # We need a dummy admissible heuristic for A* to act as an exact solver
        class AdmissibleBaseline:
            def predict_cost(self, state, goal):
                x, y, _, _, _ = state
                gx, gy = goal
                # minimum cost is move_weight (which is 1.0) * manhattan
                return float(abs(gx - x) + abs(gy - y)) * calculator.move_weight

        baseline_heuristic = AdmissibleBaseline()

        dataset_records = []

        if train_goals is None:
            # Fallback for old tests, but shouldn't be used in real benchmark
            train_goals = [(x, y) for x in range(world.grid_size_x) for y in range(world.grid_size_y)]

        for _ in range(num_instances):
            # Sample goal from the strictly held-out training set
            goal_state = tuple(random.choice(train_goals))

            # Random valid start state
            while True:
                sx = random.randint(0, world.grid_size_x - 1)
                sy = random.randint(0, world.grid_size_y - 1)
                if world.is_valid_state(sx, sy, 0) and (sx, sy) != goal_state:
                    break

            start_state = (sx, sy, 0, 0, 0)

            # Solve using A*
            res = a_star_search(world, calculator, baseline_heuristic, start_state, goal_state, max_depth)
            path = res.path
            final_cost = res.cost

            if final_cost == float('inf') or not path:
                continue # Unsolvable or didn't reach in max_depth

            # For each state on the optimal path, compute true cost-to-go
            for i, state in enumerate(path):
                # cost-to-go is the cost of the remainder of the path
                path_remainder = path[i:]
                cost_to_go = calculator.calculate_cost(path_remainder)

                features = self.build_features(state, goal_state, world.time_steps)
                dataset_records.append({"features": features, "cost": cost_to_go})

                # Sample 1 off-path state per on-path state for coverage
                x, y, vx, vy, t = state
                from environment.discrete_world import Action
                for action in Action:
                    dx, dy = action.value
                    nx, ny = x + dx, y + dy
                    next_t = t + 1

                    if next_t < world.time_steps and world.is_valid_state(nx, ny, next_t):
                        off_path_state = (nx, ny, dx, dy, next_t)
                        if off_path_state not in path:
                            # Try to solve from off-path state to get its cost-to-go
                            off_res = a_star_search(world, calculator, baseline_heuristic, off_path_state, goal_state, max_depth - i - 1)
                            off_cost = off_res.cost
                            if off_cost != float('inf'):
                                off_features = self.build_features(off_path_state, goal_state, world.time_steps)
                                dataset_records.append({"features": off_features, "cost": off_cost})
                            break # Just one off-path state per on-path state to save time

        return self.process_trajectories(dataset_records)

    def process_trajectories(self, dataset_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Converts the dataset records into a DataFrame.
        Each record should have a 'features' key with the list returned by build_features,
        and a 'cost' key with the target value.
        """
        features_list = []

        for item in dataset_records:
            features = item["features"]
            cost = item["cost"]

            record = {f"feature_{i}": val for i, val in enumerate(features)}
            record["cost"] = cost
            features_list.append(record)

        return pd.DataFrame(features_list)

    def to_torch_dataset(self, df: pd.DataFrame) -> TrajectoryDataset:
        """
        Returns a PyTorch Dataset object from the dataframe.
        """
        if df.empty:
            return TrajectoryDataset(torch.empty(0), torch.empty(0))

        feature_cols = [col for col in df.columns if col.startswith("feature_")]

        features_tensor = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        costs_tensor = torch.tensor(df["cost"].values, dtype=torch.float32).unsqueeze(1)

        return TrajectoryDataset(features_tensor, costs_tensor)

    def export_csv(self, df: pd.DataFrame, filepath: str):
        df.to_csv(filepath, index=False)

    def export_parquet(self, df: pd.DataFrame, filepath: str):
        df.to_parquet(filepath, index=False)
