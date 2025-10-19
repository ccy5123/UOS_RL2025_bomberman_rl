"""
MAPPO Agent - Neural Network Models
Actor, Critic, and RND networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .config import *


class PolicyNetwork(nn.Module):
    """
    Actor Network (Policy)
    Input: Local observation (7, 17, 17)
    Output: Action distribution over 6 actions
    """
    def __init__(self):
        super(PolicyNetwork, self).__init__()

        # CNN for spatial features
        self.conv = nn.Sequential(
            nn.Conv2d(STATE_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        conv_out_size = 64 * BOARD_SIZE * BOARD_SIZE

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Linear(128, N_ACTIONS)

    def forward(self, obs):
        """
        Args:
            obs: (batch, 7, 17, 17)
        Returns:
            Categorical distribution over actions
        """
        x = self.conv(obs)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logits = self.policy_head(x)

        return Categorical(logits=logits)

    def evaluate_actions(self, obs, actions):
        """
        Evaluate log prob and entropy for given actions.
        Used in PPO update.
        """
        dist = self.forward(obs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy


class CentralizedValueNetwork(nn.Module):
    """
    Centralized Critic Network
    Input: Global state (GLOBAL_STATE_DIM,)
    Output: State value V(s_global)
    """
    def __init__(self):
        super(CentralizedValueNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(GLOBAL_STATE_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Output: single value
        )

    def forward(self, global_state):
        """
        Args:
            global_state: (batch, GLOBAL_STATE_DIM)
        Returns:
            values: (batch,)
        """
        return self.fc(global_state).squeeze(-1)


class RandomNetwork(nn.Module):
    """
    Random Network for RND (Random Network Distillation)
    Used for exploration bonus
    """
    def __init__(self, output_dim=128):
        super(RandomNetwork, self).__init__()

        # CNN similar to policy
        self.conv = nn.Sequential(
            nn.Conv2d(STATE_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        conv_out_size = 64 * BOARD_SIZE * BOARD_SIZE

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, obs):
        """
        Args:
            obs: (batch, 7, 17, 17)
        Returns:
            features: (batch, output_dim)
        """
        x = self.conv(obs)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        return features


class RND:
    """
    Random Network Distillation for exploration
    """
    def __init__(self, device='cpu'):
        self.device = device

        # Fixed random target network
        self.target_network = RandomNetwork().to(device)
        self.target_network.eval()
        for param in self.target_network.parameters():
            param.requires_grad = False

        # Trainable predictor network
        self.predictor_network = RandomNetwork().to(device)
        self.optimizer = torch.optim.Adam(self.predictor_network.parameters(), lr=1e-4)

    def compute_intrinsic_reward(self, obs):
        """
        Compute exploration bonus based on prediction error.

        Args:
            obs: (batch, 7, 17, 17)
        Returns:
            intrinsic_rewards: (batch,)
        """
        with torch.no_grad():
            target_features = self.target_network(obs)

        predictor_features = self.predictor_network(obs)

        # Intrinsic reward = MSE between predictor and target
        intrinsic_reward = F.mse_loss(predictor_features, target_features.detach(), reduction='none').mean(dim=1)

        return intrinsic_reward.detach()

    def update(self, obs_batch):
        """
        Train predictor to match target.

        Args:
            obs_batch: (batch, 7, 17, 17)
        """
        target_features = self.target_network(obs_batch).detach()
        predictor_features = self.predictor_network(obs_batch)

        loss = F.mse_loss(predictor_features, target_features)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path):
        """Save RND networks"""
        torch.save({
            'target': self.target_network.state_dict(),
            'predictor': self.predictor_network.state_dict(),
        }, path)

    def load(self, path):
        """Load RND networks"""
        checkpoint = torch.load(path)
        self.target_network.load_state_dict(checkpoint['target'])
        self.predictor_network.load_state_dict(checkpoint['predictor'])
