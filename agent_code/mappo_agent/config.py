"""
MAPPO Agent - Configuration
Hyperparameters and settings
"""

# PPO Hyperparameters
PPO_EPOCHS = 4
CLIP_EPSILON = 0.2
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
VALUE_LOSS_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5

# Training Parameters
EPISODES_PER_UPDATE = 4  # Collect N episodes before updating
BATCH_SIZE = 256
MAX_EPISODE_STEPS = 400

# Exploration (RND)
USE_RND = True
INTRINSIC_REWARD_COEF = 0.1
INTRINSIC_ANNEAL_RATE = 0.9999

# State Representation
STATE_CHANNELS = 7
BOARD_SIZE = 17
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
N_ACTIONS = len(ACTIONS)

# Global State Dimension (for centralized critic)
# 4 agents × (2 pos + 3 features) + max 20 bombs × 3 + max 50 coins × 2 + map hash
MAX_AGENTS = 4
MAX_BOMBS = 20
MAX_COINS = 50
GLOBAL_STATE_DIM = (MAX_AGENTS * 5 +  # Agent features
                     MAX_BOMBS * 3 +    # Bomb features
                     MAX_COINS * 2 +    # Coin features
                     100)                # Map encoding (hash/small)

# Reward Shaping
REWARDS = {
    # Base rewards
    'COIN_COLLECTED': 1.0,
    'KILLED_OPPONENT': 5.0,
    'KILLED_SELF': -5.0,
    'GOT_KILLED': -5.0,
    'SURVIVED_ROUND': 2.0,
    'CRATE_DESTROYED': 0.1,
    'COIN_FOUND': 0.2,
    'INVALID_ACTION': -0.1,

    # Shaped rewards
    'ESCAPED_DANGER': 0.5,
    'ENTERED_DANGER': -0.3,
    'MOVED_CLOSER_TO_COIN': 0.05,
    'MOVED_AWAY_FROM_COIN': -0.02,
    'REPEATED_POSITION': -0.01,
    'BOMB_GOOD_PLACEMENT': 0.3,
    'BOMB_BAD_PLACEMENT': -0.1,
}

# Model Paths
ACTOR_PATH = "checkpoints/actor.pt"
CRITIC_PATH = "checkpoints/critic.pt"
RND_TARGET_PATH = "checkpoints/rnd_target.pt"
RND_PREDICTOR_PATH = "checkpoints/rnd_predictor.pt"

# Logging
LOG_INTERVAL = 10  # Log every N episodes
SAVE_INTERVAL = 100  # Save checkpoints every N episodes
EVAL_INTERVAL = 50  # Evaluate every N episodes
