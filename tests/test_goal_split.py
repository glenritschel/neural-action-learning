import pytest
from environment.discrete_world import DiscreteWorld
from action.action_calculator import ActionCalculator
from models.dataset_builder import DatasetBuilder
from experiments.goal_split import create_goal_split

def test_goal_split_leakage():
    grid_x, grid_y = 10, 10
    split = create_goal_split(grid_x, grid_y, test_size=10, seed=42)

    train_goals = [tuple(g) for g in split["train_goals"]]
    test_goals = [tuple(g) for g in split["test_goals"]]

    # Assert disjoint
    for tg in test_goals:
        assert tg not in train_goals

    world = DiscreteWorld(grid_size_x=grid_x, grid_size_y=grid_y, time_steps=12)
    calculator = ActionCalculator(world)
    builder = DatasetBuilder(grid_x, grid_y, 12)

    # Build dataset using only train goals
    df = builder.generate_cost_to_go_dataset(world, calculator, num_instances=20, max_depth=10, train_goals=train_goals)

    # Check that in the generated dataset features, none of them contain a test goal
    if not df.empty:
        for idx, row in df.iterrows():
            gx_recovered = round(row["feature_5"] * grid_x)
            gy_recovered = round(row["feature_6"] * grid_y)

            recovered_goal = (gx_recovered, gy_recovered)
            assert recovered_goal not in test_goals, f"Test goal {recovered_goal} leaked into training data!"
