import pytest
import os
import pandas as pd
import torch
from models.dataset_builder import DatasetBuilder, TrajectoryDataset

@pytest.fixture
def builder():
    return DatasetBuilder(grid_size_x=10, grid_size_y=10, time_steps=20)

@pytest.fixture
def mock_dataset_records():
    return [
        {
            "features": [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1],
            "cost": 1.0,
        },
        {
            "features": [0.5, 0.5, 1.0, 0.0, 0.25, 0.9, 0.9, 0.75, 0.4, 0.4, 0.4],
            "cost": 2.5,
        }
    ]

def test_build_features(builder):
    state = (0, 0, 1, -1, 0)
    goal = (10, 10)

    features = builder.build_features(state, goal)

    assert len(features) == 11
    # [nx, ny, nvx, nvy, nt, ngx, ngy, steps_remaining_norm, dx, dy, manhattan_norm]
    # nx, ny
    assert features[0] == 0.0
    assert features[1] == 0.0
    # nvx, nvy
    assert features[2] == 1.0
    assert features[3] == -1.0
    # nt
    assert features[4] == 0.0
    # ngx, ngy
    assert features[5] == 1.0
    assert features[6] == 1.0
    # steps_remaining_norm = (20 - 0) / 20 = 1.0
    assert features[7] == 1.0
    # dx, dy = (10 - 0) / 10 = 1.0
    assert features[8] == 1.0
    assert features[9] == 1.0
    # manhattan_norm = 20 / 20 = 1.0
    assert features[10] == 1.0

def test_process_trajectories(builder, mock_dataset_records):
    df = builder.process_trajectories(mock_dataset_records)

    assert len(df) == 2
    # 11 features + 1 cost column = 12
    assert len(df.columns) == 12
    assert "cost" in df.columns

    assert df.iloc[0]["feature_0"] == 0.0
    assert df.iloc[0]["feature_10"] == 0.1

    assert df.iloc[0]["cost"] == 1.0

def test_to_torch_dataset(builder, mock_dataset_records):
    df = builder.process_trajectories(mock_dataset_records)
    dataset = builder.to_torch_dataset(df)

    assert len(dataset) == 2

    features, cost = dataset[0]
    assert features.shape == (11,)
    assert cost.shape == (1,)
    assert cost.item() == 1.0

def test_exports(builder, mock_dataset_records, tmp_path):
    df = builder.process_trajectories(mock_dataset_records)

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
