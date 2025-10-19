# MAPPO Implementation Design for Bomberman RL

## 📋 Executive Summary

**Goal**: Implement Multi-Agent Proximal Policy Optimization (MAPPO) with Centralized Training and Decentralized Execution (CTDE) for Bomberman competition.

**Key Insight**: This environment provides full observability (`others`, `bombs`, `coins`) → Perfect for centralized critic while maintaining decentralized actors.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   CENTRALIZED TRAINING                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Centralized Value Network (Critic)          │  │
│  │   Input: Global State (all agents, bombs, coins)     │  │
│  │   Output: State value V(s_global)                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                            ↓                                 │
│              Shared Parameters + PPO Update                  │
│                            ↓                                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Policy Network (Actor)                   │  │
│  │   Input: Local Observation (ego-centric view)        │  │
│  │   Output: Action distribution π(a|o_local)           │  │
│  │   Parameters: Shared across all agents               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             ↓ Deployment
┌─────────────────────────────────────────────────────────────┐
│                 DECENTRALIZED EXECUTION                      │
│                                                              │
│  Agent 1: π(a₁|o₁)    Agent 2: π(a₂|o₂)                    │
│  Agent 3: π(a₃|o₃)    Agent 4: π(a₄|o₄)                    │
│                                                              │
│  (Each uses same policy network independently)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 State Representation

### Local Observation (for Actor)
**7-channel CNN input (17×17 each)**

```python
local_obs = [
    # Channel 0: Terrain
    walls_and_crates,     # -1: wall, 0: free, 1: crate

    # Channel 1: Self position
    self_position,        # 1 at agent position, 0 elsewhere

    # Channel 2: Other agents
    others_position,      # 1 at each opponent position

    # Channel 3: Coins
    coins_map,            # 1 at coin positions

    # Channel 4: Bombs (timer encoded)
    bombs_timer_map,      # value = timer (4→1), 0 if no bomb

    # Channel 5: Danger zones
    danger_map,           # predicted explosion coverage

    # Channel 6: Current explosions
    explosion_map,        # current explosion damage zones
]
# Shape: (7, 17, 17)
```

### Global State (for Critic)
**Concatenated features**

```python
global_state = [
    # All agent positions (4 agents × 2 coords)
    agent_positions,      # [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

    # All agent features (4 agents × features)
    agent_features,       # [score, bombs_left, alive] × 4

    # Bomb information
    bomb_positions,       # All bomb (x,y,timer) tuples

    # Coin positions
    coin_positions,       # All coin (x,y) tuples

    # Map hash (to capture crate state)
    map_encoding,         # Flattened 17×17 or hash
]
# Shape: Variable, but fixed max size with padding
```

---

## 🧠 Network Architectures

### Actor Network (Policy)
```python
class PolicyNetwork(nn.Module):
    def __init__(self):
        # CNN for spatial features
        self.conv = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Fully connected for decision
        self.fc = nn.Sequential(
            nn.Linear(64 * 17 * 17, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6),  # 6 actions
        )

    def forward(self, obs):
        # obs: (batch, 7, 17, 17)
        x = self.conv(obs)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        return Categorical(logits=logits)
```

### Critic Network (Centralized Value)
```python
class CentralizedValueNetwork(nn.Module):
    def __init__(self, global_state_dim):
        self.fc = nn.Sequential(
            nn.Linear(global_state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # State value
        )

    def forward(self, global_state):
        # global_state: (batch, global_state_dim)
        return self.fc(global_state)
```

---

## 🎁 Reward Shaping

### Base Rewards (from environment)
```python
base_rewards = {
    'COIN_COLLECTED': +1.0,
    'KILLED_OPPONENT': +5.0,
    'GOT_KILLED': -5.0,
    'KILLED_SELF': -5.0,
    'SURVIVED_ROUND': +2.0,
}
```

### Shaped Rewards (custom)
```python
shaped_rewards = {
    # Survival incentives
    'ESCAPED_DANGER': +0.5,        # Left danger zone successfully
    'ENTERED_DANGER': -0.3,        # Walked into danger

    # Exploration
    'CRATE_DESTROYED': +0.1,       # Destroyed a crate
    'COIN_FOUND': +0.2,            # Revealed a coin

    # Movement quality
    'MOVED_CLOSER_TO_COIN': +0.05, # Reduced distance to nearest coin
    'MOVED_AWAY_FROM_COIN': -0.02, # Increased distance

    # Bomb strategy
    'BOMB_GOOD_PLACEMENT': +0.3,   # Bomb near crates/opponents
    'BOMB_BAD_PLACEMENT': -0.1,    # Bomb in empty area

    # Anti-camping
    'REPEATED_POSITION': -0.01,    # Stayed in same position
    'INVALID_ACTION': -0.05,       # Tried invalid move

    # Potential-based shaping (PBRS)
    'POTENTIAL_DELTA': γ*φ(s') - φ(s),  # See below
}
```

### Potential Function (for PBRS)
```python
def compute_potential(state):
    """
    Potential-based reward shaping (preserves optimal policy)
    φ(s) = weighted sum of:
      - Distance to nearest coin (negative)
      - Distance to nearest crate (negative)
      - Distance to danger zones (positive)
      - Current score
    """
    coin_potential = -min_distance_to_coins(state)
    crate_potential = -min_distance_to_crates(state)
    safety_potential = min_distance_to_danger(state)
    score_potential = state['self']['score']

    φ = (0.1 * coin_potential +
         0.05 * crate_potential +
         0.2 * safety_potential +
         1.0 * score_potential)

    return φ

# In reward computation:
shaped_reward += γ * φ(s_next) - φ(s_current)
```

---

## 🔄 Training Algorithm

### PPO Update (on-policy)
```python
def ppo_update(trajectories, actor, critic, optimizer_actor, optimizer_critic):
    """
    Standard PPO update with centralized critic
    """
    for epoch in range(PPO_EPOCHS):
        for batch in sample_batches(trajectories):
            # Unpack batch
            local_obs, global_state, actions, old_log_probs, returns, advantages = batch

            # Actor update
            dist = actor(local_obs)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-ε, 1+ε) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic update
            values = critic(global_state)
            critic_loss = F.mse_loss(values, returns)

            # Optimize
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()
```

### GAE (Generalized Advantage Estimation)
```python
def compute_gae(rewards, values, next_values, dones, γ=0.99, λ=0.95):
    """
    Compute GAE advantages
    """
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        if dones[t]:
            delta = rewards[t] - values[t]
            gae = delta
        else:
            delta = rewards[t] + γ * next_values[t] - values[t]
            gae = delta + γ * λ * gae

        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns
```

---

## 🎮 Self-Play Strategy

### Curriculum Progression
```python
curriculum = [
    # Stage 1: Learn basics (2000 episodes)
    {
        'scenario': 'coin-heaven',
        'opponents': ['self'] * 3,  # All same policy
        'episodes': 2000,
    },

    # Stage 2: Learn bomb mechanics (3000 episodes)
    {
        'scenario': 'loot-crate',
        'opponents': ['self'] * 3,
        'episodes': 3000,
    },

    # Stage 3: Combat vs rule-based (2000 episodes)
    {
        'scenario': 'classic',
        'opponents': ['rule_based_agent'] * 3,
        'episodes': 2000,
    },

    # Stage 4: Self-play league (5000 episodes)
    {
        'scenario': 'classic',
        'opponents': 'league',  # Mix of current, best, past checkpoints
        'episodes': 5000,
    },
]
```

### League System
```python
class SelfPlayLeague:
    def __init__(self):
        self.current_policy = None
        self.best_policy = None
        self.past_policies = []  # Checkpoints every 500 episodes

    def sample_opponents(self):
        """
        Sample 3 opponents for 4-player game
        Distribution:
          - 50% current policy (latest)
          - 30% best policy (highest win rate)
          - 20% random past policy
        """
        opponents = []

        # 1 or 2 current
        if random.random() < 0.5:
            opponents.append(('current', self.current_policy))

        # 1 best
        if random.random() < 0.3 or len(opponents) == 0:
            opponents.append(('best', self.best_policy))

        # Fill rest with past
        while len(opponents) < 3:
            policy = random.choice(self.past_policies)
            opponents.append(('past', policy))

        return opponents
```

---

## 🔬 Exploration: RND (Random Network Distillation)

### Implementation
```python
class RND:
    """
    Exploration bonus via prediction error
    """
    def __init__(self, obs_shape):
        # Random target network (fixed)
        self.target = RandomCNN(obs_shape, output_dim=128)
        self.target.eval()

        # Predictor network (trained)
        self.predictor = RandomCNN(obs_shape, output_dim=128)

    def compute_bonus(self, obs):
        """
        Intrinsic reward = prediction error
        """
        with torch.no_grad():
            target_feat = self.target(obs)

        pred_feat = self.predictor(obs)

        bonus = F.mse_loss(pred_feat, target_feat, reduction='none').mean(dim=1)
        return bonus

    def update(self, obs_batch):
        """
        Train predictor to match target
        """
        target_feat = self.target(obs_batch).detach()
        pred_feat = self.predictor(obs_batch)
        loss = F.mse_loss(pred_feat, target_feat)
        # Optimize...
```

### Reward Combination
```python
total_reward = extrinsic_reward + β * intrinsic_reward
# β typically 0.1-0.5, anneal over time
```

---

## 📊 Training Pipeline

### Episode Loop
```python
def train_episode(env, actor, critic, rnd, replay_buffer):
    """
    Single episode training
    """
    state = env.reset()
    episode_data = []

    while not done:
        # Get local obs and global state
        local_obs = extract_local_obs(state)
        global_state = extract_global_state(state)

        # Sample action from policy
        with torch.no_grad():
            dist = actor(local_obs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = critic(global_state)

        # Execute action
        next_state, reward, done, info = env.step(action)

        # Compute intrinsic reward
        intrinsic = rnd.compute_bonus(local_obs)

        # Compute shaped reward
        shaped_reward = compute_shaped_reward(state, action, next_state, info)

        # Total reward
        total_reward = reward + shaped_reward + β * intrinsic

        # Store transition
        episode_data.append({
            'local_obs': local_obs,
            'global_state': global_state,
            'action': action,
            'log_prob': log_prob,
            'value': value,
            'reward': total_reward,
            'done': done,
        })

        state = next_state

    # Compute GAE
    compute_gae_for_episode(episode_data)

    # Add to replay buffer
    replay_buffer.extend(episode_data)
```

### Main Training Loop
```python
def main_training_loop():
    # Initialize
    actor = PolicyNetwork()
    critic = CentralizedValueNetwork()
    rnd = RND(obs_shape=(7, 17, 17))
    replay_buffer = []

    # Training
    for episode in range(TOTAL_EPISODES):
        # Collect trajectories (N episodes per update)
        for _ in range(N_EPISODES_PER_UPDATE):
            train_episode(env, actor, critic, rnd, replay_buffer)

        # PPO update
        if len(replay_buffer) >= BATCH_SIZE:
            ppo_update(replay_buffer, actor, critic, ...)

            # Update RND
            rnd.update(sample_obs_from_buffer(replay_buffer))

            # Clear buffer (on-policy)
            replay_buffer.clear()

        # Evaluate periodically
        if episode % 100 == 0:
            eval_score = evaluate(actor, n_games=10)
            print(f"Episode {episode}: Eval score = {eval_score}")

            # Save checkpoint
            if eval_score > best_score:
                save_checkpoint(actor, critic, episode)
                best_score = eval_score
```

---

## 🔧 Hyperparameters

```python
# PPO
PPO_EPOCHS = 4
CLIP_EPSILON = 0.2
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01

# Training
TOTAL_EPISODES = 10000
N_EPISODES_PER_UPDATE = 4  # Collect 4 episodes, then update
BATCH_SIZE = 256
MAX_GRAD_NORM = 0.5

# Exploration (RND)
INTRINSIC_REWARD_COEF = 0.1  # β
INTRINSIC_ANNEAL_RATE = 0.9999  # Decay per episode

# Curriculum
CURRICULUM_THRESHOLDS = {
    'coin-heaven': 5.0,   # Avg score to advance
    'loot-crate': 8.0,
    'classic': 10.0,
}
```

---

## 📁 File Structure

```
agent_code/mappo_agent/
├── callbacks.py          # Main entry point (setup, act)
├── train.py              # Training callbacks (game_events_occurred, end_of_round)
├── models.py             # Network definitions (Actor, Critic, RND)
├── ppo.py                # PPO update logic
├── features.py           # State extraction (local_obs, global_state)
├── rewards.py            # Reward shaping functions
├── self_play.py          # League system
├── config.py             # Hyperparameters
├── checkpoints/          # Saved models
│   ├── actor_best.pt
│   ├── critic_best.pt
│   ├── actor_ep1000.pt
│   └── ...
└── logs/                 # Training logs
    └── training.log
```

---

## 🎯 Success Metrics

### Phase 1 (Weeks 1-2)
- ✅ Agent learns to collect coins in `coin-heaven` (avg score > 5)
- ✅ Agent avoids self-destruction (suicide rate < 10%)
- ✅ Policy loss converges

### Phase 2 (Weeks 3-4)
- ✅ Agent destroys crates efficiently in `loot-crate` (avg score > 8)
- ✅ Agent escapes bombs reliably (survival rate > 70%)
- ✅ Beats random_agent 90%+ win rate

### Phase 3 (Weeks 5-6)
- ✅ Beats rule_based_agent 60%+ win rate in 1v1
- ✅ Avg score > 10 in `classic` 4-player
- ✅ Demonstrates strategic bomb placement

### Phase 4 (Weeks 7-8)
- ✅ Beats rule_based_agent 70%+ win rate
- ✅ Avg score > 15 in `classic`
- ✅ Wins internal tournament

---

## 🚀 Next Steps

1. **Implement baseline DQN** (for comparison)
2. **Build feature extraction** (`features.py`)
3. **Implement Actor network** (`models.py`)
4. **Implement Critic network** (`models.py`)
5. **Implement PPO update** (`ppo.py`)
6. **Test on `coin-heaven`** (simplest scenario)
7. **Add reward shaping** (`rewards.py`)
8. **Add RND exploration** (`models.py`)
9. **Implement self-play league** (`self_play.py`)
10. **Curriculum training** on all scenarios

---

**Document Version**: 1.0
**Last Updated**: 2025-10-19
**Status**: Design Phase → Ready for Implementation
