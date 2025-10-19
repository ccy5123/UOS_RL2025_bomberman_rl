"""
MAPPO Agent - Training
PPO update with centralized critic
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import numpy as np
import events as e
from .config import *
from .features import extract_local_observation, extract_global_state, compute_danger_map

# Transition for episode data
Transition = namedtuple('Transition',
                        ('local_obs', 'global_state', 'action', 'log_prob', 'value', 'reward', 'done'))


def setup_training(self):
    """
    Initialize training-specific components.
    """
    self.logger.info("Setting up MAPPO training...")

    # Set networks to training mode
    self.actor.train()
    self.critic.train()

    # Optimizers
    self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE_ACTOR)
    self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE_CRITIC)

    # Episode buffer (collect multiple episodes before update)
    self.episode_buffer = []
    self.current_episode_data = []

    # Training statistics
    self.total_episodes = 0
    self.actor_losses = []
    self.critic_losses = []


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list):
    """
    Called after each step during training.
    Store transition for later PPO update.
    """
    self.logger.debug(f'Events: {", ".join(map(repr, events))}')

    # Extract features
    local_obs = extract_local_observation(old_game_state)
    global_state = extract_global_state(old_game_state)

    if local_obs is None or global_state is None:
        return

    # Get action index
    action_idx = ACTIONS.index(self_action)

    # Compute reward
    reward = reward_from_events(self, events, old_game_state, self_action, new_game_state)

    # Compute value and log_prob for this state-action
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(local_obs).unsqueeze(0).to(self.device)
        global_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        action_tensor = torch.LongTensor([action_idx]).to(self.device)

        dist = self.actor(obs_tensor)
        log_prob = dist.log_prob(action_tensor).item()
        value = self.critic(global_tensor).item()

    # Compute intrinsic reward (RND)
    if self.rnd is not None:
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(local_obs).unsqueeze(0).to(self.device)
            intrinsic = self.rnd.compute_intrinsic_reward(obs_tensor).item()
            reward += self.intrinsic_coef * intrinsic

    # Store transition
    transition = Transition(
        local_obs=local_obs,
        global_state=global_state,
        action=action_idx,
        log_prob=log_prob,
        value=value,
        reward=reward,
        done=False
    )

    self.current_episode_data.append(transition)


def end_of_round(self, last_game_state: dict, last_action: str, events: list):
    """
    Called at the end of each round.
    Process episode, add to buffer, and update if enough episodes collected.
    """
    self.logger.debug(f'End of round events: {", ".join(map(repr, events))}')

    # Store final transition
    if last_game_state is not None and self.current_episode_data:
        local_obs = extract_local_observation(last_game_state)
        global_state = extract_global_state(last_game_state)

        if local_obs is not None and global_state is not None:
            action_idx = ACTIONS.index(last_action)
            reward = reward_from_events(self, events, last_game_state, last_action, None)

            with torch.no_grad():
                obs_tensor = torch.FloatTensor(local_obs).unsqueeze(0).to(self.device)
                global_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
                action_tensor = torch.LongTensor([action_idx]).to(self.device)

                dist = self.actor(obs_tensor)
                log_prob = dist.log_prob(action_tensor).item()
                value = self.critic(global_tensor).item()

            transition = Transition(
                local_obs=local_obs,
                global_state=global_state,
                action=action_idx,
                log_prob=log_prob,
                value=value,
                reward=reward,
                done=True
            )
            self.current_episode_data.append(transition)

    # Add episode to buffer
    if self.current_episode_data:
        self.episode_buffer.append(self.current_episode_data)
        self.current_episode_data = []

    self.total_episodes += 1

    # Update policy when enough episodes collected
    if len(self.episode_buffer) >= EPISODES_PER_UPDATE:
        self.logger.info(f"Updating policy (collected {len(self.episode_buffer)} episodes)")
        ppo_update(self)
        self.episode_buffer = []  # Clear buffer (on-policy)

    # Save periodically
    if self.total_episodes % SAVE_INTERVAL == 0:
        save_checkpoint(self)

    # Log statistics
    if self.total_episodes % LOG_INTERVAL == 0:
        avg_actor_loss = np.mean(self.actor_losses) if self.actor_losses else 0
        avg_critic_loss = np.mean(self.critic_losses) if self.critic_losses else 0

        self.logger.info(f"Episode {self.total_episodes} | "
                        f"Actor Loss: {avg_actor_loss:.4f} | "
                        f"Critic Loss: {avg_critic_loss:.4f}")

        self.actor_losses = []
        self.critic_losses = []

    # Anneal intrinsic reward coefficient
    if self.rnd is not None:
        self.intrinsic_coef *= INTRINSIC_ANNEAL_RATE


def ppo_update(self):
    """
    Perform PPO update on collected episodes.
    """
    # Flatten all transitions from all episodes
    all_transitions = [t for episode in self.episode_buffer for t in episode]

    if len(all_transitions) < BATCH_SIZE:
        self.logger.warning(f"Not enough transitions ({len(all_transitions)}) for update")
        return

    # Compute GAE advantages and returns
    compute_gae(self.episode_buffer)

    # Convert to tensors
    local_obs = torch.FloatTensor(np.array([t.local_obs for t in all_transitions])).to(self.device)
    global_states = torch.FloatTensor(np.array([t.global_state for t in all_transitions])).to(self.device)
    actions = torch.LongTensor([t.action for t in all_transitions]).to(self.device)
    old_log_probs = torch.FloatTensor([t.log_prob for t in all_transitions]).to(self.device)
    returns = torch.FloatTensor([t.reward for t in all_transitions]).to(self.device)  # Will use computed returns
    advantages = returns - torch.FloatTensor([t.value for t in all_transitions]).to(self.device)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO epochs
    for epoch in range(PPO_EPOCHS):
        # Sample mini-batches
        indices = np.random.permutation(len(all_transitions))

        for start in range(0, len(indices), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(indices))
            batch_indices = indices[start:end]

            # Get batch
            batch_obs = local_obs[batch_indices]
            batch_global = global_states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_old_log_probs = old_log_probs[batch_indices]
            batch_returns = returns[batch_indices]
            batch_advantages = advantages[batch_indices]

            # Evaluate current policy
            new_log_probs, entropy = self.actor.evaluate_actions(batch_obs, batch_actions)
            values = self.critic(batch_global)

            # PPO actor loss
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy.mean()

            # Critic loss
            critic_loss = F.mse_loss(values, batch_returns)

            # Update actor
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
            self.optimizer_actor.step()

            # Update critic
            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
            self.optimizer_critic.step()

            self.actor_losses.append(actor_loss.item())
            self.critic_losses.append(critic_loss.item())

    # Update RND if enabled
    if self.rnd is not None:
        rnd_loss = self.rnd.update(local_obs)
        self.logger.debug(f"RND loss: {rnd_loss:.4f}")


def compute_gae(episode_buffer):
    """
    Compute Generalized Advantage Estimation for all episodes.
    Modifies transitions in-place to add 'advantage' and 'return'.
    """
    for episode in episode_buffer:
        advantages = []
        returns = []
        gae = 0
        next_value = 0

        for t in reversed(episode):
            if t.done:
                delta = t.reward - t.value
                gae = delta
            else:
                delta = t.reward + GAMMA * next_value - t.value
                gae = delta + GAMMA * GAE_LAMBDA * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + t.value)
            next_value = t.value

        # Update transitions with computed values
        for i, t in enumerate(episode):
            # Create new transition with updated values
            episode[i] = t._replace(reward=returns[i])  # Store return in reward field


def reward_from_events(self, events: list, old_state: dict, action: str, new_state: dict) -> float:
    """
    Compute reward from events and state transitions.
    """
    # Base rewards
    reward = 0
    for event in events:
        if event in e.__dict__:
            event_name = [k for k, v in e.__dict__.items() if v == event][0]
            if event_name in REWARDS:
                reward += REWARDS[event_name]

    # Shaped rewards
    if new_state is not None:
        reward += compute_shaped_reward(old_state, action, new_state)

    return reward


def compute_shaped_reward(old_state: dict, action: str, new_state: dict) -> float:
    """
    Additional reward shaping.
    """
    shaped_reward = 0

    _, _, _, (old_x, old_y) = old_state['self']
    _, _, _, (new_x, new_y) = new_state['self']

    # Escape from danger
    old_danger = compute_danger_map(old_state)
    new_danger = compute_danger_map(new_state)

    if old_danger[old_x, old_y] > 0 and new_danger[new_x, new_y] == 0:
        shaped_reward += REWARDS['ESCAPED_DANGER']

    if old_danger[old_x, old_y] == 0 and new_danger[new_x, new_y] > 0:
        shaped_reward += REWARDS['ENTERED_DANGER']

    # Movement towards coins
    old_coins = old_state['coins']
    new_coins = new_state['coins']

    if old_coins and new_coins:
        old_min_dist = min(abs(c[0] - old_x) + abs(c[1] - old_y) for c in old_coins)
        new_min_dist = min(abs(c[0] - new_x) + abs(c[1] - new_y) for c in new_coins)

        if new_min_dist < old_min_dist:
            shaped_reward += REWARDS['MOVED_CLOSER_TO_COIN']
        elif new_min_dist > old_min_dist:
            shaped_reward += REWARDS['MOVED_AWAY_FROM_COIN']

    # Anti-camping
    if (old_x, old_y) == (new_x, new_y) and action != 'BOMB':
        shaped_reward += REWARDS['REPEATED_POSITION']

    return shaped_reward


def save_checkpoint(self):
    """
    Save model checkpoints.
    """
    torch.save(self.actor.state_dict(), ACTOR_PATH)
    torch.save(self.critic.state_dict(), CRITIC_PATH)

    if self.rnd is not None:
        self.rnd.save(RND_TARGET_PATH)

    self.logger.info(f"Checkpoint saved at episode {self.total_episodes}")
