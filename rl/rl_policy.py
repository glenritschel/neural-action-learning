import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import List

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class REINFORCE:
    def __init__(self, policy: PolicyNetwork, learning_rate: float = 1e-3, gamma: float = 0.99):
        self.policy = policy
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma

    def update_policy(self, rewards: List[float], log_probs: List[torch.Tensor]) -> float:
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

        # Normalize rewards
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

        return loss.item()

def train_reinforce(env, agent: REINFORCE, num_episodes: int, start_state: tuple) -> list:
    """
    Runs a REINFORCE training loop by collecting rollouts from the environment.
    """
    loss_history = []

    for episode in range(num_episodes):
        state = env.reset(start_state)

        log_probs = []
        rewards = []

        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            # Forward pass to get action probabilities
            probs = agent.policy(state_tensor)

            # Sample an action
            m = Categorical(probs)
            action_idx = m.sample()

            log_prob = m.log_prob(action_idx)

            # Step environment
            next_state, reward, done, info = env.step(action_idx.item())

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state

        # Update policy at the end of the episode
        if len(rewards) > 0:
            loss = agent.update_policy(rewards, log_probs)
            loss_history.append(loss)

    return loss_history
