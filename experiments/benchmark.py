import sys
import os
import json
import random
import pandas as pd
import torch

# Add root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.discrete_world import DiscreteWorld
from action.action_calculator import ActionCalculator
from search.search_algorithms import brute_force_search, dynamic_programming_search, a_star_search, focal_search, beam_search
from models.mlp import ActionMLP
from search.heuristic import MLPHeuristic

def run_benchmark():
    print("Running Benchmark Harness...")
    random.seed(42)
    torch.manual_seed(42)

    # 1. Setup Environment
    world = DiscreteWorld(grid_size_x=10, grid_size_y=10, time_steps=12)
    calculator = ActionCalculator(world, move_weight=1.0, turn_weight=0.25, obstacle_weight=100.0, acceleration_weight=1.0)

    # 2. Add some random obstacles
    for _ in range(15):
        x, y = random.randint(0, 9), random.randint(0, 9)
        if (x, y) != (0, 0): # Keep start clear
            world.set_obstacle(x, y)

    # 3. Load trained MLP model
    # The feature size is 11 as defined in DatasetBuilder
    model = ActionMLP(input_dim=11, output_dim=1)
    weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "action_mlp_weights.pth")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}. Run train_model.py first.")
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    heuristic = MLPHeuristic(model, world)

    # Baseline dummy heuristic (admissible only)
    class DummyHeuristic:
        pass
    dummy_h = DummyHeuristic()

    # 4. Generate Held-Out Instances
    num_instances = 10
    max_depth = 10
    held_out_pairs = []

    for _ in range(num_instances):
        # Start state is always (0,0) for simplicity
        start = (0, 0, 0, 0, 0)

        while True:
            gx, gy = random.randint(3, 9), random.randint(3, 9)
            if not world.is_obstacle(gx, gy):
                break
        goal = (gx, gy)
        held_out_pairs.append((start, goal))

    results = []

    # 5. Run Methods
    for i, (start, goal) in enumerate(held_out_pairs):
        print(f"Instance {i+1}/{num_instances} - Goal: {goal}")

        # Method 1: Dynamic Programming (Ground Truth)
        dp_res = dynamic_programming_search(world, calculator, start, goal, max_depth)

        # Method 2: Standard A* (Admissible baseline)
        astar_std_res = a_star_search(world, calculator, dummy_h, start, goal, max_depth, weight=0.0)

        # Method 3: Neural-Guided A* (Bounded-suboptimal, w=1.5) using Focal Search
        astar_nn_res = focal_search(world, calculator, heuristic, start, goal, max_depth, weight=1.5)

        # Method 4: Beam Search (w/ Neural Heuristic)
        beam_res = beam_search(world, calculator, heuristic, start, goal, max_depth, beam_width=5)

        # Helper to compute suboptimality
        def compute_optimality(cost, gt_cost):
            if gt_cost == float('inf'):
                return 0.0 # Unsolvable
            if cost == float('inf'):
                return 0.0
            if gt_cost == 0:
                return 100.0 if cost == 0 else 0.0
            return (gt_cost / cost) * 100.0

        for method_name, res in [
            ("DP (Ground Truth)", dp_res),
            ("Standard A*", astar_std_res),
            ("Neural A* (w=1.5)", astar_nn_res),
            ("Beam Search", beam_res)
        ]:
            # Completeness: Did it find a path if one exists?
            path_found = len(res.path) > 0
            is_complete = path_found == (len(dp_res.path) > 0)

            # % optimal
            pct_optimal = compute_optimality(res.cost, dp_res.cost)

            results.append({
                "Instance": i,
                "Goal": goal,
                "Method": method_name,
                "Cost": res.cost,
                "Nodes Expanded": res.nodes_expanded,
                "Runtime (s)": res.runtime,
                "Memory (bytes)": res.memory_footprint,
                "% Optimal": pct_optimal,
                "Completeness": is_complete
            })

    # 6. Save and Report
    df = pd.DataFrame(results)

    # Calculate aggregate metrics
    agg_df = df.groupby("Method").agg({
        "Cost": ["mean"],
        "Nodes Expanded": ["mean"],
        "Runtime (s)": ["mean"],
        "% Optimal": ["mean"],
        "Completeness": ["mean"]
    }).reset_index()

    # Flatten columns
    agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns.values]

    print("\nAggregate Results:")
    print(agg_df.to_string(index=False))

    # Save CSV
    df.to_csv("experiments/benchmark_results.csv", index=False)

    # Read metrics
    metrics_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "metrics.json")
    held_out_mae = "N/A"
    held_out_r2 = "N/A"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            held_out_mae = metrics.get("mae", "N/A")
            held_out_r2 = metrics.get("r2", "N/A")

    # Write RESULTS.md
    with open("RESULTS.md", "w") as f:
        f.write("# Headline Metrics\n\n")
        f.write("Generated by `experiments/benchmark.py` on held-out test goals.\n\n")
        f.write(f"**Held-out MAE:** {held_out_mae}\n\n")
        f.write(f"**Held-out R2:** {held_out_r2}\n\n")
        f.write("## Aggregate Performance\n\n")
        f.write(agg_df.to_markdown(index=False))

if __name__ == "__main__":
    run_benchmark()
