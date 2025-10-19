"""
MAPPO Agent - Callbacks
Main entry points for the agent
"""

import os
import torch
import numpy as np
from .config import *
from .models import PolicyNetwork, CentralizedValueNetwork, RND
from .features import extract_local_observation, extract_global_state


def setup(self):
    """
    Setup agent for playing.
    Called once when loading the agent.
    """
    self.logger.info("Setting up MAPPO agent...")

    # Device
    self.device = torch.device('cpu')  # Use CPU for compatibility

    # Initialize networks
    self.actor = PolicyNetwork().to(self.device)
    self.critic = CentralizedValueNetwork().to(self.device)

    # Load models if available
    if not self.train and os.path.exists(ACTOR_PATH):
        self.logger.info(f"Loading actor from {ACTOR_PATH}")
        self.actor.load_state_dict(torch.load(ACTOR_PATH, map_location=self.device))

    if not self.train and os.path.exists(CRITIC_PATH):
        self.logger.info(f"Loading critic from {CRITIC_PATH}")
        self.critic.load_state_dict(torch.load(CRITIC_PATH, map_location=self.device))

    # Set to evaluation mode
    self.actor.eval()
    self.critic.eval()

    # RND for exploration
    if USE_RND and self.train:
        self.rnd = RND(device=self.device)
        self.intrinsic_coef = INTRINSIC_REWARD_COEF
    else:
        self.rnd = None
        self.intrinsic_coef = 0.0

    # Statistics
    self.step_count = 0
    self.episode_count = 0


def act(self, game_state: dict) -> str:
    """
    Choose action based on current game state.

    Args:
        game_state: Dictionary containing game information

    Returns:
        action: One of ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
    """
    self.step_count += 1

    # Extract local observation
    local_obs = extract_local_observation(game_state)

    if local_obs is None:
        return 'WAIT'

    # Convert to tensor
    obs_tensor = torch.FloatTensor(local_obs).unsqueeze(0).to(self.device)

    # Sample action from policy
    with torch.no_grad():
        dist = self.actor(obs_tensor)

        if self.train:
            # Sample stochastically during training
            action_idx = dist.sample().item()
        else:
            # Use greedy action during evaluation
            action_idx = dist.probs.argmax().item()

    action = ACTIONS[action_idx]

    self.logger.debug(f"Step {self.step_count}: Action = {action}, "
                     f"Probs = {dist.probs.detach().cpu().numpy()[0]}")

    return action


def state_to_features(game_state: dict):
    """
    Legacy function for compatibility.
    Now redirects to extract_local_observation.
    """
    return extract_local_observation(game_state)
