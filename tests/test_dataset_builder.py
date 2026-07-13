import pytest
import os
import pandas as pd
import torch
from models.dataset_builder import DatasetBuilder, TrajectoryDataset

@pytest.fixture
def builder():
    return DatasetBuilder(grid_size_x=10, grid_size_y=10, time_steps=20)

@pytest.fixture
def mock_trajectories():
    return [
        {
            "trajectory": [(0, 0, 0), (1, 0, 1)],
            "cost": 1.0,
            "endpoint": (1, 0, 1),
            "metadata": {"length": 2, "start": (0, 0, 0)}
        },
        {
            "trajectory": [(5, 5, 5), (5, 6, 6)],
            "cost": 2.5,
            "endpoint": (5, 6, 6),
            "metadata": {"length": 2, "start": (5, 5, 5)}
        }
    ]

def test_build_features(builder, mock_trajectories):
    df = builder.build_features(mock_trajectories)

    assert len(df) == 2
    # 2 states per trajectory, 3 features (x,y,t) per state -> 6 features total + 1 cost column = 7
    assert len(df.columns) == 7
    assert "cost" in df.columns

    # Check normalization for first state of first trajectory
    # (0, 0, 0) -> (0.0, 0.0, 0.0)
    assert df.iloc[0]["feature_0"] == 0.0
    assert df.iloc[0]["feature_1"] == 0.0
    assert df.iloc[0]["feature_2"] == 0.0

    # Check normalization for second state of first trajectory
    # (1, 0, 1) -> (1/10, 0, 1/20) -> (0.1, 0.0, 0.05)
    assert pytest.approx(df.iloc[0]["feature_3"]) == 0.1
    assert df.iloc[0]["feature_4"] == 0.0
    assert pytest.approx(df.iloc[0]["feature_5"]) == 0.05

    assert df.iloc[0]["cost"] == 1.0

def test_to_torch_dataset(builder, mock_trajectories):
    df = builder.build_features(mock_trajectories)
    dataset = builder.to_torch_dataset(df)

    assert len(dataset) == 2

    features, cost = dataset[0]
    assert features.shape == (6,)
    assert cost.shape == (1,)
    assert cost.item() == 1.0

def test_exports(builder, mock_trajectories, tmp_path):
    df = builder.build_features(mock_trajectories)

    csv_path = tmp_path / "test.csv"
    builder.export_csv(df, str(csv_path))
    assert csv_path.exists()

    loaded_csv = pd.read_csv(str(csv_path))
    assert len(loaded_csv) == 2

    parquet_path = tmp_path / "test.parquet"
    builder.export_parquet(df, str(parquet_path))
    assert parquet_path.exists()

    loaded_parquet = pd.read_parquet(str(parquet_path))
    assert len(loaded_parquet) == 2
