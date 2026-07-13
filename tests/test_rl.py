import pytest
from environment.discrete_world import DiscreteWorld, Action
from action.action_calculator import ActionCalculator
from rl.rl_env import DiscreteWorldEnv
from rl.rl_policy import QLearningAgent, train_q_learning

def test_rl_env():
    world = DiscreteWorld(grid_size_x=5, grid_size_y=5, time_steps=10)
    calculator = ActionCalculator(world)
    env = DiscreteWorldEnv(world, calculator, goal_state=(4, 4), max_steps=20)

    start_state = (0, 0, 0)
    state = env.reset(start_state)
    assert state == start_state

    next_state, reward, done, info = env.step(Action.UP)
    assert next_state == (0, 1, 1)
    assert not done
    assert reward <= 0 # step cost is non-positive

def test_rl_policy_training():
    world = DiscreteWorld(grid_size_x=3, grid_size_y=3, time_steps=10)
    calculator = ActionCalculator(world)
    env = DiscreteWorldEnv(world, calculator, goal_state=(2, 2), max_steps=10)

    agent = QLearningAgent(actions=list(Action), alpha=0.1, gamma=0.9, epsilon=1.0)

    rewards = train_q_learning(env, agent, start_state=(0, 0, 0), episodes=5)

    assert len(rewards) == 5
    # Just checking it executes without crashing
