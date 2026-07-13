import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.discrete_world import DiscreteWorld
from action.action_calculator import ActionCalculator
from models.dataset_builder import DatasetBuilder
from models.mlp import ActionMLP

def train():
    print("Training ActionMLP...")

    # Setup environment
    world = DiscreteWorld(grid_size_x=10, grid_size_y=10, time_steps=12)
    calculator = ActionCalculator(world, move_weight=1.0, turn_weight=0.25, obstacle_weight=100.0, acceleration_weight=1.0)
    builder = DatasetBuilder(world.grid_size_x, world.grid_size_y, world.time_steps)

    # Load goals
    split_path = os.path.join(os.path.dirname(__file__), "goal_split.json")
    if not os.path.exists(split_path):
        from experiments.goal_split import create_goal_split
        create_goal_split(world.grid_size_x, world.grid_size_y)

    with open(split_path, "r") as f:
        split_data = json.load(f)

    train_goals = [tuple(g) for g in split_data["train_goals"]]

    print(f"Generating dataset from {len(train_goals)} training goals...")
    df = builder.generate_cost_to_go_dataset(world, calculator, num_instances=200, max_depth=12, train_goals=train_goals)

    dataset = builder.to_torch_dataset(df)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model Setup
    sample_state = (0, 0, 0, 0, 0)
    sample_goal = (0, 0)
    input_dim = len(builder.build_features(sample_state, sample_goal))

    model = ActionMLP(input_dim=input_dim, output_dim=1)

    # Runtime assertion
    assert input_dim == len(builder.build_features(sample_state, sample_goal)), "Input dimension mismatch"

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting training loop...")
    epochs = 20
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for features, costs in train_loader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = model.compute_loss(predictions, costs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for features, costs in val_loader:
                predictions = model(features)
                loss = model.compute_loss(predictions, costs)
                val_loss += loss.item()
                all_preds.append(predictions)
                all_targets.append(costs)

        if all_preds:
            preds_tensor = torch.cat(all_preds)
            targets_tensor = torch.cat(all_targets)
            metrics = model.compute_metrics(preds_tensor, targets_tensor)
        else:
            metrics = {"mae": 0, "r2": 0}

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f} - Val Loss: {val_loss/len(val_loader):.4f} - Val MAE: {metrics['mae']:.4f} - Val R2: {metrics['r2']:.4f}")

    metrics_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"mae": float(metrics["mae"]), "r2": float(metrics["r2"])}, f)

    weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "action_mlp_weights.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"Model saved to {weights_path}")

if __name__ == "__main__":
    train()
