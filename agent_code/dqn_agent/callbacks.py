"""
DQN Agent - Callbacks
Deep Q-Network with rule-guided exploration
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup agent for playing.
    Called once at the beginning.
    """
    self.logger.info("Setting up DQN agent...")

    # Model path
    self.model_path = "dqn_model.pt"

    # Initialize Q-network
    if self.train or not os.path.exists(self.model_path):
        self.logger.info("Creating new DQN model")
        self.model = DQN()
        self.target_model = DQN()  # Target network for stable learning
        self.target_model.load_state_dict(self.model.state_dict())
    else:
        self.logger.info(f"Loading model from {self.model_path}")
        self.model = DQN()
        self.model.load_state_dict(torch.load(self.model_path))
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())

    self.model.eval()  # Set to evaluation mode for inference
    self.target_model.eval()

    # Exploration parameters
    if self.train:
        self.epsilon = 1.0  # Start with full exploration
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
    else:
        self.epsilon = 0.0  # No exploration during evaluation

    # Track statistics
    self.step_count = 0


def act(self, game_state: dict) -> str:
    """
    Choose action based on current game state.
    Uses epsilon-greedy with rule-based guidance.
    """
    self.step_count += 1

    # Extract features
    features = state_to_features(game_state)

    if features is None:
        return 'WAIT'

    # Epsilon-greedy action selection
    if self.train and random.random() < self.epsilon:
        # Exploration: use rule-based logic instead of pure random
        action = rule_based_action(game_state, self.logger)
        self.logger.debug(f"Exploring with rule-based: {action}")
    else:
        # Exploitation: use Q-network
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            q_values = self.model(features_tensor)
            action_idx = q_values.argmax().item()
            action = ACTIONS[action_idx]
            self.logger.debug(f"Q-values: {q_values.numpy()[0]}, chose {action}")

    return action


class DQN(nn.Module):
    """
    Deep Q-Network
    Input: 7-channel 17x17 state
    Output: 6 Q-values (one per action)
    """
    def __init__(self):
        super(DQN, self).__init__()

        # Convolutional layers for spatial features
        self.conv = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Calculate flattened size: 64 channels × 17 × 17
        conv_out_size = 64 * 17 * 17

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(ACTIONS)),
        )

    def forward(self, x):
        # x: (batch, 7, 17, 17)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        q_values = self.fc(x)
        return q_values


def state_to_features(game_state: dict) -> np.array:
    """
    Convert game state to 7-channel feature representation.

    Channels:
        0: Walls and crates (-1: wall, 0: free, 1: crate)
        1: Self position (1 at agent position)
        2: Other agents (1 at each opponent)
        3: Coins (1 at coin positions)
        4: Bombs (value = timer)
        5: Danger zones (predicted explosion)
        6: Current explosions
    """
    if game_state is None:
        return None

    # Board size
    rows, cols = 17, 17

    # Initialize channels
    channels = []

    # Channel 0: Walls and crates
    field = game_state['field']
    channels.append(field.copy())

    # Channel 1: Self position
    self_channel = np.zeros((rows, cols))
    _, _, _, (x, y) = game_state['self']
    self_channel[x, y] = 1
    channels.append(self_channel)

    # Channel 2: Other agents
    others_channel = np.zeros((rows, cols))
    for _, _, _, (x, y) in game_state['others']:
        others_channel[x, y] = 1
    channels.append(others_channel)

    # Channel 3: Coins
    coins_channel = np.zeros((rows, cols))
    for x, y in game_state['coins']:
        coins_channel[x, y] = 1
    channels.append(coins_channel)

    # Channel 4: Bombs (with timer)
    bombs_channel = np.zeros((rows, cols))
    for (x, y), timer in game_state['bombs']:
        bombs_channel[x, y] = timer
    channels.append(bombs_channel)

    # Channel 5: Danger zones (explosion prediction)
    danger_channel = compute_danger_map(game_state)
    channels.append(danger_channel)

    # Channel 6: Current explosions
    explosion_channel = game_state['explosion_map'].copy()
    channels.append(explosion_channel)

    # Stack channels: (7, 17, 17)
    features = np.stack(channels)

    return features.astype(np.float32)


def compute_danger_map(game_state: dict) -> np.array:
    """
    Compute danger map showing predicted explosion zones.
    """
    rows, cols = 17, 17
    danger_map = np.zeros((rows, cols))
    field = game_state['field']

    for (bx, by), timer in game_state['bombs']:
        # Bomb explosion range = 3
        blast_coords = [(bx, by)]

        # Expand in 4 directions
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            for i in range(1, 4):  # Range = 3
                x, y = bx + i * dx, by + i * dy
                if 0 <= x < cols and 0 <= y < rows:
                    if field[x, y] == -1:  # Wall blocks
                        break
                    blast_coords.append((x, y))
                    if field[x, y] == 1:  # Crate blocks but still affected
                        break

        # Mark danger with timer value (higher = more urgent)
        for x, y in blast_coords:
            danger_map[x, y] = max(danger_map[x, y], timer)

    return danger_map


def rule_based_action(game_state: dict, logger) -> str:
    """
    Simple rule-based policy for exploration.
    Simplified version of rule_based_agent.
    """
    arena = game_state['field']
    _, _, bombs_left, (x, y) = game_state['self']
    coins = game_state['coins']
    bombs = game_state['bombs']
    others = [xy for (_, _, _, xy) in game_state['others']]

    # Check valid moves
    valid_actions = []
    directions = {
        'UP': (x, y - 1),
        'DOWN': (x, y + 1),
        'LEFT': (x - 1, y),
        'RIGHT': (x + 1, y),
    }

    for action, (nx, ny) in directions.items():
        if (arena[nx, ny] == 0 and
            (nx, ny) not in others and
            not any((nx, ny) == bomb_pos for bomb_pos, _ in bombs)):
            valid_actions.append(action)

    valid_actions.append('WAIT')

    if bombs_left:
        valid_actions.append('BOMB')

    # Simple strategy: move towards coins, avoid bombs
    danger_map = compute_danger_map(game_state)

    if danger_map[x, y] > 0:
        # In danger! Try to escape
        for action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            if action in valid_actions:
                nx, ny = directions[action]
                if danger_map[nx, ny] == 0:
                    logger.debug(f"Escaping danger via {action}")
                    return action

    # Move towards nearest coin
    if coins:
        nearest_coin = min(coins, key=lambda c: abs(c[0] - x) + abs(c[1] - y))
        cx, cy = nearest_coin

        if cx > x and 'RIGHT' in valid_actions:
            return 'RIGHT'
        elif cx < x and 'LEFT' in valid_actions:
            return 'LEFT'
        elif cy > y and 'DOWN' in valid_actions:
            return 'DOWN'
        elif cy < y and 'UP' in valid_actions:
            return 'UP'

    # Default: random valid action
    return random.choice(valid_actions) if valid_actions else 'WAIT'
