import json
import random
import os

def create_goal_split(grid_size_x=10, grid_size_y=10, test_size=10, seed=42):
    random.seed(seed)

    # Generate all possible coordinates
    all_goals = [(x, y) for x in range(grid_size_x) for y in range(grid_size_y)]

    # Shuffle and split
    random.shuffle(all_goals)

    test_goals = all_goals[:test_size]
    train_goals = all_goals[test_size:]

    split_data = {
        "train_goals": train_goals,
        "test_goals": test_goals
    }

    output_path = os.path.join(os.path.dirname(__file__), "goal_split.json")
    with open(output_path, "w") as f:
        json.dump(split_data, f, indent=2)

    print(f"Goal split saved to {output_path}")
    return split_data

if __name__ == "__main__":
    create_goal_split()
