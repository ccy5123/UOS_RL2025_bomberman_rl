"""
MAPPO Agent - Feature Extraction
Convert game state to neural network inputs
"""

import numpy as np
from .config import *


def extract_local_observation(game_state: dict) -> np.array:
    """
    Extract local observation for actor (ego-centric view).

    Returns:
        np.array of shape (7, 17, 17)
    """
    if game_state is None:
        return None

    channels = []

    # Channel 0: Walls and crates
    field = game_state['field']
    channels.append(field.copy())

    # Channel 1: Self position
    self_channel = np.zeros((BOARD_SIZE, BOARD_SIZE))
    _, _, _, (x, y) = game_state['self']
    self_channel[x, y] = 1
    channels.append(self_channel)

    # Channel 2: Other agents
    others_channel = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for _, _, _, (x, y) in game_state['others']:
        others_channel[x, y] = 1
    channels.append(others_channel)

    # Channel 3: Coins
    coins_channel = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for x, y in game_state['coins']:
        coins_channel[x, y] = 1
    channels.append(coins_channel)

    # Channel 4: Bombs (timer encoded)
    bombs_channel = np.zeros((BOARD_SIZE, BOARD_SIZE))
    for (x, y), timer in game_state['bombs']:
        bombs_channel[x, y] = timer / 4.0  # Normalize timer (max 4)
    channels.append(bombs_channel)

    # Channel 5: Danger zones
    danger_channel = compute_danger_map(game_state)
    channels.append(danger_channel)

    # Channel 6: Current explosions
    explosion_channel = game_state['explosion_map'].copy()
    # Normalize explosion map
    if explosion_channel.max() > 0:
        explosion_channel = explosion_channel / explosion_channel.max()
    channels.append(explosion_channel)

    # Stack: (7, 17, 17)
    local_obs = np.stack(channels).astype(np.float32)

    return local_obs


def extract_global_state(game_state: dict) -> np.array:
    """
    Extract global state for centralized critic.

    Returns:
        np.array of shape (GLOBAL_STATE_DIM,)
    """
    if game_state is None:
        return None

    features = []

    # 1. All agent information (self + others)
    all_agents = [game_state['self']] + list(game_state['others'])

    agent_features = np.zeros(MAX_AGENTS * 5)  # 5 features per agent
    for i, (name, score, bombs_left, (x, y)) in enumerate(all_agents[:MAX_AGENTS]):
        offset = i * 5
        agent_features[offset] = x / BOARD_SIZE      # Normalized x
        agent_features[offset + 1] = y / BOARD_SIZE  # Normalized y
        agent_features[offset + 2] = score / 20.0    # Normalized score
        agent_features[offset + 3] = bombs_left      # Boolean: has bomb
        agent_features[offset + 4] = 1.0             # Alive flag
    features.append(agent_features)

    # 2. Bomb information
    bomb_features = np.zeros(MAX_BOMBS * 3)
    for i, ((x, y), timer) in enumerate(game_state['bombs'][:MAX_BOMBS]):
        offset = i * 3
        bomb_features[offset] = x / BOARD_SIZE
        bomb_features[offset + 1] = y / BOARD_SIZE
        bomb_features[offset + 2] = timer / 4.0
    features.append(bomb_features)

    # 3. Coin information
    coin_features = np.zeros(MAX_COINS * 2)
    for i, (x, y) in enumerate(game_state['coins'][:MAX_COINS]):
        offset = i * 2
        coin_features[offset] = x / BOARD_SIZE
        coin_features[offset + 1] = y / BOARD_SIZE
    features.append(coin_features)

    # 4. Map encoding (simple: count of crates in each quadrant)
    field = game_state['field']
    map_encoding = []

    # Divide into 4 quadrants
    mid_x, mid_y = BOARD_SIZE // 2, BOARD_SIZE // 2
    quadrants = [
        field[:mid_x, :mid_y],
        field[mid_x:, :mid_y],
        field[:mid_x, mid_y:],
        field[mid_x:, mid_y:],
    ]

    for quadrant in quadrants:
        crates = (quadrant == 1).sum()
        map_encoding.append(crates / 50.0)  # Normalize

    # Add some global map statistics
    total_crates = (field == 1).sum()
    map_encoding.append(total_crates / 100.0)
    map_encoding.append(len(game_state['coins']) / MAX_COINS)

    # Pad to 100 dimensions
    map_encoding = np.array(map_encoding)
    map_encoding = np.pad(map_encoding, (0, 100 - len(map_encoding)))
    features.append(map_encoding)

    # Concatenate all features
    global_state = np.concatenate(features).astype(np.float32)

    assert len(global_state) == GLOBAL_STATE_DIM, f"Expected {GLOBAL_STATE_DIM}, got {len(global_state)}"

    return global_state


def compute_danger_map(game_state: dict) -> np.array:
    """
    Compute danger map showing explosion zones.
    Value = urgency (higher timer = more urgent to escape)
    """
    danger_map = np.zeros((BOARD_SIZE, BOARD_SIZE))
    field = game_state['field']

    for (bx, by), timer in game_state['bombs']:
        # Explosion range = 3 (from settings.py: BOMB_POWER = 3)
        blast_coords = [(bx, by)]

        # Expand in 4 directions
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            for i in range(1, 4):  # Range 3
                x, y = bx + i * dx, by + i * dy
                if 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE:
                    if field[x, y] == -1:  # Wall blocks
                        break
                    blast_coords.append((x, y))
                    if field[x, y] == 1:  # Crate blocks but is affected
                        break

        # Mark danger (timer value: 4 = most urgent, 1 = about to explode)
        for x, y in blast_coords:
            danger_map[x, y] = max(danger_map[x, y], timer / 4.0)

    return danger_map
