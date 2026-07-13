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

    def build_features(self, trajectories_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Converts the raw trajectories into a DataFrame of flattened, normalized coordinates.
        """
        features_list = []

        for item in trajectories_data:
            trajectory = item["trajectory"]
            cost = item["cost"]

            flattened = []
            for x, y, t in trajectory:
                # Normalize
                nx = x / float(self.grid_size_x)
                ny = y / float(self.grid_size_y)
                nt = t / float(self.time_steps)
                flattened.extend([nx, ny, nt])

            record = {f"feature_{i}": val for i, val in enumerate(flattened)}
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
