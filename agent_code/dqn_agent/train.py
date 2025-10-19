"""
DQN Agent - Training
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import numpy as np
import events as e
from .callbacks import state_to_features, ACTIONS, compute_danger_map

# Transition tuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 0.001
REPLAY_BUFFER_SIZE = 10000
TARGET_UPDATE_FREQ = 100  # Update target network every N episodes
MIN_REPLAY_SIZE = 1000  # Start training after collecting N transitions


def setup_training(self):
    """
    Initialize training-specific components.
    """
    self.logger.info("Setting up DQN training...")

    # Replay buffer
    self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    # Optimizer
    self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    # Training metrics
    self.episode_count = 0
    self.total_steps = 0
    self.losses = []

    # For tracking previous state
    self.last_state = None
    self.last_action = None
    self.last_features = None


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):
    """
    Called after each step. Store transition and train.
    """
    self.logger.debug(f'Events: {", ".join(map(repr, events))}')

    # Convert states to features
    old_features = state_to_features(old_game_state)
    new_features = state_to_features(new_game_state)

    if old_features is None or new_features is None:
        return

    # Compute reward
    reward = reward_from_events(self, events, old_game_state, self_action, new_game_state)

    # Store transition
    action_idx = ACTIONS.index(self_action)
    transition = Transition(old_features, action_idx, new_features, reward)
    self.replay_buffer.append(transition)

    self.total_steps += 1

    # Train if enough samples
    if len(self.replay_buffer) >= MIN_REPLAY_SIZE:
        train_step(self)

    # Decay epsilon
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """
    Called at the end of each round.
    """
    self.logger.debug(f'End of round events: {", ".join(map(repr, events))}')

    # Store final transition (with None as next_state)
    if last_game_state is not None:
        last_features = state_to_features(last_game_state)
        if last_features is not None:
            reward = reward_from_events(self, events, last_game_state, last_action, None)
            action_idx = ACTIONS.index(last_action)
            transition = Transition(last_features, action_idx, None, reward)
            self.replay_buffer.append(transition)

    self.episode_count += 1

    # Update target network periodically
    if self.episode_count % TARGET_UPDATE_FREQ == 0:
        self.logger.info(f"Updating target network (episode {self.episode_count})")
        self.target_model.load_state_dict(self.model.state_dict())

    # Save model periodically
    if self.episode_count % 100 == 0:
        torch.save(self.model.state_dict(), "dqn_model.pt")
        self.logger.info(f"Model saved at episode {self.episode_count}")

    # Log statistics
    avg_loss = np.mean(self.losses) if self.losses else 0
    self.logger.info(f"Episode {self.episode_count} | Epsilon: {self.epsilon:.3f} | "
                     f"Avg Loss: {avg_loss:.4f} | Buffer: {len(self.replay_buffer)}")
    self.losses = []  # Reset for next episode


def train_step(self):
    """
    Sample batch and perform one gradient descent step.
    """
    if len(self.replay_buffer) < BATCH_SIZE:
        return

    # Sample random batch
    batch = random.sample(self.replay_buffer, BATCH_SIZE)

    # Separate components
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []

    for transition in batch:
        states.append(transition.state)
        actions.append(transition.action)
        rewards.append(transition.reward)

        if transition.next_state is None:
            # Terminal state
            next_states.append(np.zeros_like(transition.state))
            dones.append(1)
        else:
            next_states.append(transition.next_state)
            dones.append(0)

    # Convert to tensors
    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(np.array(next_states))
    dones = torch.FloatTensor(dones)

    # Compute current Q values
    current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute target Q values (using target network)
    with torch.no_grad():
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    # Compute loss
    loss = F.mse_loss(current_q_values, target_q_values)

    # Optimize
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
    self.optimizer.step()

    self.losses.append(loss.item())


def reward_from_events(self, events: list, old_state: dict, action: str, new_state: dict) -> float:
    """
    Compute reward from events and state transitions.
    Combines environment rewards with shaped rewards.
    """
    # Base rewards from environment
    game_rewards = {
        e.COIN_COLLECTED: 1.0,
        e.KILLED_OPPONENT: 5.0,
        e.KILLED_SELF: -5.0,
        e.GOT_KILLED: -5.0,
        e.SURVIVED_ROUND: 2.0,
        e.CRATE_DESTROYED: 0.1,
        e.COIN_FOUND: 0.2,
        e.INVALID_ACTION: -0.1,
    }

    reward = 0
    for event in events:
        if event in game_rewards:
            reward += game_rewards[event]

    # Shaped rewards
    if new_state is not None:
        reward += compute_shaped_reward(old_state, action, new_state)

    return reward


def compute_shaped_reward(old_state: dict, action: str, new_state: dict) -> float:
    """
    Additional reward shaping based on state transitions.
    """
    shaped_reward = 0

    # Get positions
    _, _, _, (old_x, old_y) = old_state['self']
    _, _, _, (new_x, new_y) = new_state['self']

    # 1. Escape from danger
    old_danger = compute_danger_map(old_state)
    new_danger = compute_danger_map(new_state)

    if old_danger[old_x, old_y] > 0 and new_danger[new_x, new_y] == 0:
        shaped_reward += 0.5  # Successfully escaped danger!

    if old_danger[old_x, old_y] == 0 and new_danger[new_x, new_y] > 0:
        shaped_reward -= 0.3  # Walked into danger

    # 2. Movement towards coins
    old_coins = old_state['coins']
    new_coins = new_state['coins']

    if old_coins and new_coins:  # If coins exist in both states
        old_min_dist = min(abs(c[0] - old_x) + abs(c[1] - old_y) for c in old_coins)
        new_min_dist = min(abs(c[0] - new_x) + abs(c[1] - new_y) for c in new_coins)

        if new_min_dist < old_min_dist:
            shaped_reward += 0.05  # Moved closer to coin
        elif new_min_dist > old_min_dist:
            shaped_reward -= 0.02  # Moved away from coin

    # 3. Penalize staying in same position (anti-camping)
    if (old_x, old_y) == (new_x, new_y) and action != 'BOMB':
        shaped_reward -= 0.01

    return shaped_reward


# Import random for sampling
import random
