import random
from typing import Dict, Tuple, List, Any
from collections import defaultdict
import numpy as np

from environment.discrete_world import Action
from rl.rl_env import DiscreteWorldEnv

class QLearningAgent:
    def __init__(self, actions: List[Action], alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table: state -> dict(action -> value)
        self.q_table = defaultdict(lambda: {a: 0.0 for a in self.actions})

    def get_action(self, state: Tuple[int, int, int]) -> Action:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)

    def update(self, state: Tuple[int, int, int], action: Action, reward: float, next_state: Tuple[int, int, int], done: bool):
        """Q-learning update rule."""
        best_next_q = 0.0 if done else max(self.q_table[next_state].values())

        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state][action]

        self.q_table[state][action] += self.alpha * td_error

def train_q_learning(env: DiscreteWorldEnv, agent: QLearningAgent, start_state: Tuple[int, int, int], episodes: int = 1000) -> List[float]:
    """
    Trains the QLearningAgent in the given environment.
    Returns a list of total rewards per episode.
    """
    episode_rewards = []

    for ep in range(episodes):
        state = env.reset(start_state)
        total_reward = 0.0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        # Optionally decay epsilon
        if agent.epsilon > 0.01:
            agent.epsilon *= 0.995

    return episode_rewards
