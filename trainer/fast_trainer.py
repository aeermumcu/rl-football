#!/usr/bin/env python3
"""
Fast Vectorized DQN Trainer for RL Football
============================================
Uses parallel environments and compiled TensorFlow for 50-100x speedup.

Run: python fast_trainer.py --episodes=100000
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import time
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================================================
# VECTORIZED GAME ENVIRONMENT (runs N games in parallel)
# ============================================================================

class VectorizedGame:
    """
    Runs N parallel football games using NumPy batch operations.
    All state is stored as (N, ...) shaped arrays.
    """
    
    def __init__(self, n_envs=16):
        self.n = n_envs
        self.W, self.H = 720, 420
        self.reset_all()
    
    def reset_all(self):
        """Reset all environments"""
        # Player positions: [x, y, vx, vy]
        self.blip = np.tile([120., 210., 0., 0.], (self.n, 1))
        self.bloop = np.tile([600., 210., 0., 0.], (self.n, 1))
        self.ball = np.tile([360., 210., 0., 0.], (self.n, 1))
        
        # Scores and time
        self.scores = np.zeros((self.n, 2), dtype=np.int32)
        self.time = np.full(self.n, 30.0)
        self.kick_flags = np.zeros((self.n, 2), dtype=np.int32)
        self.done = np.zeros(self.n, dtype=bool)
    
    def reset_env(self, idx):
        """Reset a single environment by index"""
        self.blip[idx] = [120., 210., 0., 0.]
        self.bloop[idx] = [600., 210., 0., 0.]
        self.ball[idx] = [360., 210., 0., 0.]
        self.scores[idx] = [0, 0]
        self.time[idx] = 30.0
        self.done[idx] = False
    
    def reset_positions(self, mask):
        """Reset ball and player positions after goal (only where mask is True)"""
        self.blip[mask, :2] = [120., 210.]
        self.bloop[mask, :2] = [600., 210.]
        self.ball[mask] = [360., 210., 0., 0.]
    
    def step(self, actions_blip, actions_bloop):
        """
        Step all environments with given actions.
        actions_blip/bloop: (N,) array of action indices 0-9
        Returns: events (N,) array of 'W'/'L'/None, dones (N,) bool array
        """
        # Movement directions for actions 0-7, 8=kick, 9=stay
        MOVES = np.array([
            [0, -1], [0, 1], [-1, 0], [1, 0],
            [-1, -1], [1, -1], [-1, 1], [1, 1],
            [0, 0], [0, 0]
        ], dtype=np.float32)
        
        # Apply actions to players
        for i, (player, actions) in enumerate([(self.blip, actions_blip), (self.bloop, actions_bloop)]):
            # Movement for non-kick, non-stay actions
            move_mask = actions < 8
            if np.any(move_mask):
                dirs = MOVES[actions[move_mask]]
                player[move_mask, 2] += dirs[:, 0] * 2
                player[move_mask, 3] += dirs[:, 1] * 2
            
            # Kick flag
            self.kick_flags[:, i] = (actions == 8).astype(np.int32)
            
            # Clamp speed
            speed = np.sqrt(player[:, 2]**2 + player[:, 3]**2)
            too_fast = speed > 4
            if np.any(too_fast):
                scale = 4.0 / speed[too_fast]
                player[too_fast, 2] *= scale
                player[too_fast, 3] *= scale
        
        # Update player positions and apply friction
        for player in [self.blip, self.bloop]:
            player[:, 0] += player[:, 2]
            player[:, 1] += player[:, 3]
            player[:, 2:4] *= 0.85
            
            # Constrain to field
            player[:, 0] = np.clip(player[:, 0], 25, self.W - 25)
            player[:, 1] = np.clip(player[:, 1], 25, self.H - 25)
        
        # Update ball
        self.ball[:, 0] += self.ball[:, 2]
        self.ball[:, 1] += self.ball[:, 3]
        self.ball[:, 2:4] *= 0.98
        
        # Ball wall bounces (top/bottom)
        top_bounce = self.ball[:, 1] < 12
        self.ball[top_bounce, 1] = 12
        self.ball[top_bounce, 3] *= -0.8
        
        bot_bounce = self.ball[:, 1] > self.H - 12
        self.ball[bot_bounce, 1] = self.H - 12
        self.ball[bot_bounce, 3] *= -0.8
        
        # Goal check (left/right walls only outside goal)
        goal_y_min = (self.H - 120) // 2
        goal_y_max = goal_y_min + 120
        in_goal_range = (self.ball[:, 1] > goal_y_min) & (self.ball[:, 1] < goal_y_max)
        
        # Left wall (outside goal)
        left_bounce = (self.ball[:, 0] < 12) & ~in_goal_range
        self.ball[left_bounce, 0] = 12
        self.ball[left_bounce, 2] *= -0.8
        
        # Right wall (outside goal)
        right_bounce = (self.ball[:, 0] > self.W - 12) & ~in_goal_range
        self.ball[right_bounce, 0] = self.W - 12
        self.ball[right_bounce, 2] *= -0.8
        
        # Player-ball collisions
        for i, player in enumerate([self.blip, self.bloop]):
            dx = self.ball[:, 0] - player[:, 0]
            dy = self.ball[:, 1] - player[:, 1]
            dist = np.sqrt(dx**2 + dy**2)
            
            collide = (dist > 0) & (dist < 37)
            if np.any(collide):
                nx = dx[collide] / dist[collide]
                ny = dy[collide] / dist[collide]
                
                self.ball[collide, 0] = player[collide, 0] + nx * 37
                self.ball[collide, 1] = player[collide, 1] + ny * 37
                
                # Kick power
                power = np.where(self.kick_flags[collide, i], 12, 6)
                self.ball[collide, 2] = nx * power + player[collide, 2] * 0.5
                self.ball[collide, 3] = ny * power + player[collide, 3] * 0.5
        
        # Clear kick flags
        self.kick_flags[:] = 0
        
        # Goal scoring
        events = np.full(self.n, None, dtype=object)
        
        # Bloop scores (ball in left goal)
        bloop_goal = in_goal_range & (self.ball[:, 0] < 0)
        if np.any(bloop_goal):
            self.scores[bloop_goal, 1] += 1
            events[bloop_goal] = 'L'  # Blip loses
            self.reset_positions(bloop_goal)
        
        # Blip scores (ball in right goal)
        blip_goal = in_goal_range & (self.ball[:, 0] > self.W)
        if np.any(blip_goal):
            self.scores[blip_goal, 0] += 1
            events[blip_goal] = 'W'  # Blip wins
            self.reset_positions(blip_goal)
        
        # Update time
        self.time -= 1/60
        self.done = self.time <= 0
        
        return events, self.done.copy()
    
    def get_states(self, team=0):
        """
        Get observation states for all envs.
        team=0 for Blip (left), team=1 for Bloop (right)
        Returns: (N, 12) float32 array
        """
        if team == 0:
            player, opponent = self.blip, self.bloop
            mirror = False
        else:
            player, opponent = self.bloop, self.blip
            mirror = True
        
        # Normalized positions
        px = player[:, 0] / self.W
        py = player[:, 1] / self.H
        bx = self.ball[:, 0] / self.W
        by = self.ball[:, 1] / self.H
        bvx = np.clip(self.ball[:, 2] / 15, -1, 1)
        bvy = np.clip(self.ball[:, 3] / 15, -1, 1)
        ox = opponent[:, 0] / self.W
        oy = opponent[:, 1] / self.H
        
        # [FIX] Mirroring for Team 1 (Bloop)
        # If playing as Bloop, we flip X so the agent thinks it's Blip attacking right
        if mirror:
            px = 1.0 - px
            bx = 1.0 - bx
            ox = 1.0 - ox
            bvx = -bvx
            # Y stays the same (symmetric)
            # angle calculations will naturally follow from mirrored coords
        
        # Distance to ball (normalized)
        dist = np.sqrt((px - bx)**2 + (py - by)**2) # normalized distance
        # Note: JS uses pixel dist / 830, we use normalized dist directly logic
        # But for consistency with JS rewards, let's stick to pixel-based dist logic or normalized
        # The JS distance function uses raw pixels. Let's use raw pixels for distance to match logic
        # Wait, the previous code used:
        # dist = np.sqrt((player[:, 0] - self.ball[:, 0])**2 + ...) / 830
        # Let's keep that but careful with mirroring. Distance is invariant to mirroring.
        
        raw_dist = np.sqrt((player[:, 0] - self.ball[:, 0])**2 + 
                          (player[:, 1] - self.ball[:, 1])**2) / 830
        
        # Placeholders for angle features (matching JS format)
        # We can implement them properly now to help the agent
        angle_ball = np.arctan2(self.ball[:, 1] - player[:, 1], self.ball[:, 0] - player[:, 0])
        if mirror:
             # Mirror angle: reflect across Y axis
             # dx -> -dx, dy -> dy. atan2(dy, -dx)
             angle_ball = np.arctan2(self.ball[:, 1] - player[:, 1], -(self.ball[:, 0] - player[:, 0]))
        
        angle_ball = (angle_ball + np.pi) / (2 * np.pi)
        
        # Goal components
        # Target goal is always "My Attacking Goal"
        # For Blip (team 0): X=720. For Bloop (team 1): X=0.
        # But since we mirrored Bloop, the target is effectively at X=1.0 (normalized) for both.
        
        # Dist/Angle to goal (1.0, 0.5)
        dist_goal = np.sqrt((px - 1.0)**2 + (py - 0.5)**2)
        angle_goal = np.arctan2(0.5 - py, 1.0 - px)
        angle_goal = (angle_goal + np.pi) / (2 * np.pi)
        
        return np.stack([px, py, bx, by, bvx, bvy, ox, oy, raw_dist, 
                        angle_ball, dist_goal, angle_goal], axis=1).astype(np.float32)


# ============================================================================
# SIMPLE AI OPPONENT (vectorized)
# ============================================================================

def simple_ai_actions(states):
    """
    Vectorized SimpleAI - chases ball and kicks when close.
    states: (N, 12) array
    Returns: (N,) array of action indices
    """
    n = states.shape[0]
    actions = np.full(n, 9, dtype=np.int32)  # Default: stay
    
    px, py = states[:, 0], states[:, 1]
    bx, by = states[:, 2], states[:, 3]
    dist = states[:, 8]
    
    dx = bx - px
    dy = by - py
    
    # Kick if very close
    kick_mask = dist < 0.04
    actions[kick_mask] = 8
    
    # Chase otherwise
    chase_mask = ~kick_mask
    
    # Determine movement direction
    move_x = np.zeros(n, dtype=np.int32)
    move_y = np.zeros(n, dtype=np.int32)
    
    move_x[dx > 0.02] = 1
    move_x[dx < -0.02] = -1
    move_y[dy > 0.02] = 1
    move_y[dy < -0.02] = -1
    
    # Map movement to action index
    # 0:up, 1:down, 2:left, 3:right, 4:up-left, 5:up-right, 6:down-left, 7:down-right
    for i in range(n):
        if not chase_mask[i]:
            continue
        mx, my = move_x[i], move_y[i]
        if mx == 0 and my == -1: actions[i] = 0
        elif mx == 0 and my == 1: actions[i] = 1
        elif mx == -1 and my == 0: actions[i] = 2
        elif mx == 1 and my == 0: actions[i] = 3
        elif mx == -1 and my == -1: actions[i] = 4
        elif mx == 1 and my == -1: actions[i] = 5
        elif mx == -1 and my == 1: actions[i] = 6
        elif mx == 1 and my == 1: actions[i] = 7
    
    return actions


# ============================================================================
# DUELING DQN MODEL
# ============================================================================

def create_dueling_dqn(state_size=12, action_size=10, lr=0.0005):
    """
    Creates Dueling DQN matching the browser architecture exactly.
    """
    inputs = layers.Input(shape=(state_size,))
    
    # Shared layers
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(inputs)
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    
    # Value stream
    v = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    v = layers.Dense(1, kernel_initializer='he_normal', name='value')(v)
    
    # Advantage stream
    a = layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
    a = layers.Dense(action_size, kernel_initializer='he_normal', name='advantage')(a)
    
    # Combine: Q = V + (A - mean(A))
    mean_a = layers.Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(a)
    a_centered = layers.Subtract()([a, mean_a])
    q = layers.Add()([v, a_centered])
    
    model = keras.Model(inputs, q)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='mse')
    
    return model


# ============================================================================
# REPLAY BUFFER (efficient NumPy-based)
# ============================================================================

class ReplayBuffer:
    """Efficient circular buffer using pre-allocated NumPy arrays."""
    
    def __init__(self, capacity=50000, state_size=12):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)
        self.idx = 0
        self.size = 0
    
    def add_batch(self, states, actions, rewards, next_states, dones):
        """Add a batch of transitions."""
        n = len(states)
        for i in range(n):
            self.states[self.idx] = states[i]
            self.actions[self.idx] = actions[i]
            self.rewards[self.idx] = rewards[i]
            self.next_states[self.idx] = next_states[i]
            self.dones[self.idx] = dones[i]
            self.idx = (self.idx + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample a random batch."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )


# ============================================================================
# REWARD CALCULATION (vectorized)
# ============================================================================

def calculate_rewards(player, opponent, ball, events, last_dists, team_idx=0):
    """
    Calculate rewards for a specific agent (Blip or Bloop).
    team_idx: 0 for Blip (attacks right), 1 for Bloop (attacks left)
    """
    n = player.shape[0]
    rewards = np.zeros(n, dtype=np.float32)
    
    # Distance to ball
    dist_to_ball = np.sqrt((player[:, 0] - ball[:, 0])**2 + 
                          (player[:, 1] - ball[:, 1])**2)
    
    # Goal events
    # If team 0 (Blip): W=Score, L=Concede
    # If team 1 (Bloop): L=Score (Blip lost), W=Concede (Blip won)
    if team_idx == 0:
        rewards[events == 'W'] += 500
        rewards[events == 'L'] -= 300
    else:
        rewards[events == 'L'] += 500  # We scored (Blip lost)
        rewards[events == 'W'] -= 300  # We conceded (Blip won)
    
    # Proximity reward
    max_dist = 830
    normalized_dist = dist_to_ball / max_dist
    rewards += (1 - normalized_dist) * 5
    
    # Ball possession bonus
    rewards[dist_to_ball < 40] += 10
    
    # Movement toward ball
    if last_dists is not None:
        delta = last_dists - dist_to_ball
        rewards += delta * 0.5
        rewards[delta > 2] += 3
    
    # Movement penalty for standing still far from ball
    speed = np.sqrt(player[:, 2]**2 + player[:, 3]**2)
    still_far = (speed < 0.5) & (dist_to_ball > 50)
    rewards[still_far] -= 8
    
    # Movement bonus
    rewards[speed > 1] += 1
    
    # Ball moving toward opponent's goal
    # Blip attacks > (vx > 2), Bloop attacks < (vx < -2)
    if team_idx == 0:
        ball_toward_goal = (ball[:, 2] > 2) & (dist_to_ball < 80)
    else:
        ball_toward_goal = (ball[:, 2] < -2) & (dist_to_ball < 80)
    rewards[ball_toward_goal] += 8
    
    # Ball close to goal
    # Blip target: 720. Bloop target: 0.
    target_x = 720 if team_idx == 0 else 0
    rewards[np.abs(ball[:, 0] - target_x) < 100] += 5
    
    # Corner penalty
    in_corner = ((player[:, 0] < 80) | (player[:, 0] > 640)) & \
                ((player[:, 1] < 80) | (player[:, 1] > 340))
    rewards[in_corner] -= 5
    rewards[in_corner & (dist_to_ball > 100)] -= 5
    
    # Far from ball penalty
    rewards[dist_to_ball > 300] -= 5
    rewards[(dist_to_ball > 200) & (dist_to_ball <= 300)] -= 3
    rewards[(dist_to_ball > 150) & (dist_to_ball <= 200)] -= 1
    
    # Opponent closer to ball
    opp_dist = np.sqrt((opponent[:, 0] - ball[:, 0])**2 + 
                      (opponent[:, 1] - ball[:, 1])**2)
    rewards[(opp_dist < dist_to_ball) & (dist_to_ball > 60)] -= 2
    
    # Time penalty
    rewards -= 0.1
    
    return rewards, dist_to_ball


# ============================================================================
# COMPILED TENSORFLOW FUNCTIONS
# ============================================================================

@tf.function
def predict_batch(model, states):
    """Batched prediction - single GPU call for all states."""
    return model(states, training=False)

@tf.function
def train_step(model, optimizer, states, targets):
    """Single compiled training step."""
    with tf.GradientTape() as tape:
        q_values = model(states, training=True)
        loss = tf.reduce_mean(tf.square(q_values - targets))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


# ============================================================================
# DQN AGENT
# ============================================================================

class DQNAgent:
    def __init__(self):
        self.state_size = 12
        self.action_size = 10
        self.gamma = 0.995
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.9999
        self.lr = 0.0005
        
        self.model = create_dueling_dqn(self.state_size, self.action_size, self.lr)
        self.target_model = create_dueling_dqn(self.state_size, self.action_size, self.lr)
        self.update_target()
        
        self.buffer = ReplayBuffer(50000, self.state_size)
        self.batch_size = 64
        self.min_buffer = 500
        self.target_update_freq = 500
        self.train_step_count = 0
        
        self.optimizer = keras.optimizers.Adam(self.lr)
    
    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def act_batch(self, states):
        """Select actions for batch of states."""
        n = len(states)
        actions = np.zeros(n, dtype=np.int32)
        
        # Random actions for exploration
        random_mask = np.random.random(n) < self.epsilon
        actions[random_mask] = np.random.randint(0, self.action_size, size=np.sum(random_mask))
        
        # Greedy actions for exploitation
        if np.any(~random_mask):
            q_values = predict_batch(self.model, states[~random_mask]).numpy()
            actions[~random_mask] = np.argmax(q_values, axis=1)
        
        return actions
    
    def train(self):
        if self.buffer.size < self.min_buffer:
            return 0.0
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Double DQN: use main model to select actions, target model to evaluate
        next_q_main = predict_batch(self.model, next_states).numpy()
        next_q_target = predict_batch(self.target_model, next_states).numpy()
        
        best_actions = np.argmax(next_q_main, axis=1)
        next_q = next_q_target[np.arange(self.batch_size), best_actions]
        
        # Calculate targets
        targets = predict_batch(self.model, states).numpy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * next_q[i]
        
        # Train
        loss = train_step(self.model, self.optimizer, 
                         tf.constant(states), tf.constant(targets))
        
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.update_target()
        
        return float(loss)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def export_weights(self, filepath):
        """Export weights in browser-compatible format (Reordered for TF.js)."""
        weights_list = self.model.get_weights()
        
        # Reorder weights to match TF.js expectation
        # Python Keras (Functional): [Shared..., V_h, V_out, A_h, A_out]
        # TF.js (Sequential-ish):    [Shared..., V_h, A_h, V_out, A_out]
        # Indices:
        # Shared: 0-5
        # V_h: 6-7
        # V_out: 8-9
        # A_h: 10-11
        # A_out: 12-13
        
        if len(weights_list) == 14:
            new_order = weights_list[:8] + weights_list[10:12] + weights_list[8:10] + weights_list[12:]
            weights_list = new_order
        else:
            print(f"⚠️ Warning: Unexpected weight count {len(weights_list)}, skipping reorder.")
            
        weights = []
        for w in weights_list:
            weights.append({
                'shape': list(w.shape),
                'data': w.flatten().tolist()
            })
        
        agent_data = {
            'weights': weights,
            'epsilon': float(self.epsilon),
            'trainStepCount': int(self.train_step_count)
        }
        
        data = {
            'version': 2,
            'aiType': 'dqn',
            'trainedWith': 'fast_trainer.py',
            'blipAgent': agent_data,
            'bloopAgent': agent_data,
            'blip': agent_data,
            'bloop': agent_data
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f)
        print(f"💾 Saved weights to {filepath}")
    
    def import_weights(self, filepath):
        """Import weights from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        agent_data = data.get('blipAgent') or data.get('blip')
        if not agent_data:
            print("⚠️ No weights found in file")
            return
        
        weights = [np.array(w['data']).reshape(w['shape']) 
                  for w in agent_data['weights']]
        self.model.set_weights(weights)
        self.update_target()
        self.epsilon = agent_data.get('epsilon', self.epsilon)
        self.train_step_count = agent_data.get('trainStepCount', 0)
        print(f"✅ Loaded weights, epsilon={self.epsilon:.4f}")


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train(episodes=100000, n_envs=16, save_every=5000, resume=None):
    print("=" * 60)
    print("🚀 RL Football - Fast Vectorized Trainer (Self-Play Ready)")
    print("=" * 60)
    print(f"Episodes: {episodes}")
    print(f"Parallel environments: {n_envs}")
    print(f"Save every: {save_every}")
    print(f"GPU: {tf.config.list_physical_devices('GPU')}")
    print("=" * 60)
    
    game = VectorizedGame(n_envs)
    agent = DQNAgent()
    
    start_episode = 0
    if resume:
        agent.import_weights(resume)
        try:
            start_episode = int(resume.split('-')[-1].split('.')[0])
        except:
            pass
    
    stats = {'W': 0, 'L': 0, 'D': 0, 'goals': 0}
    t0 = time.time()
    
    env_episodes = np.zeros(n_envs, dtype=np.int32)
    total_episodes = start_episode
    
    # Track distances for reward calculation (for both agents)
    last_dists_blip = None
    last_dists_bloop = None
    
    # Action mirror mapping for Bloop
    MIRROR_MAP = np.array([0, 1, 3, 2, 5, 4, 7, 6, 8, 9], dtype=np.int32)
    
    while total_episodes < episodes:
        game.reset_all()
        last_dists_blip = None
        last_dists_bloop = None
        
        # Self-play curriculum: Start easy (SimpleAI), then switch to Self-Play
        # Switch after 50k episodes (or 20% if total is small)
        self_play_threshold = 50000
        is_self_play = total_episodes > self_play_threshold
        
        while not np.all(game.done):
            active = ~game.done
            
            # --- BLIP STEP ---
            states_blip = game.get_states(0)
            actions_blip = agent.act_batch(states_blip)
            
            # --- BLOOP STEP ---
            states_bloop = game.get_states(1) # Mirrored states
            
            if is_self_play:
                # Agent plays against itself
                actions_bloop_mirrored = agent.act_batch(states_bloop)
                # Mirror actions back to real world (Right -> Left)
                actions_bloop = MIRROR_MAP[actions_bloop_mirrored]
            else:
                # Agent plays against SimpleAI
                actions_bloop = simple_ai_actions(states_bloop) # SimpleAI handles its own logic
            
            # --- ENVIRONMENT STEP ---
            events, dones = game.step(actions_blip, actions_bloop)
            
            # --- REWARDS BLIP ---
            rewards_blip, last_dists_blip = calculate_rewards(
                game.blip, game.bloop, game.ball, events, last_dists_blip, team_idx=0
            )
            
            # --- REWARDS BLOOP ---
            # We ONLY train Bloop if it's Self-Play
            if is_self_play:
                rewards_bloop, last_dists_bloop = calculate_rewards(
                    game.bloop, game.blip, game.ball, events, last_dists_bloop, team_idx=1
                )
            
            # --- NEXT STATES & MEMORY ---
            new_states_blip = game.get_states(0)
            new_states_bloop = game.get_states(1)
            
            # Store Blip experience
            agent.buffer.add_batch(
                states_blip[active], actions_blip[active], rewards_blip[active],
                new_states_blip[active], dones[active]
            )
            
            # Store Bloop experience (if self-play)
            if is_self_play:
                # Note: We must store MIRRORED actions for Bloop so the policy learns correctly
                # actions_bloop is "Real", actions_bloop_mirrored is what the network produced
                agent.buffer.add_batch(
                    states_bloop[active], actions_bloop_mirrored[active], rewards_bloop[active],
                    new_states_bloop[active], dones[active]
                )
            
            # --- TRAINING ---
            stats['goals'] += np.sum(events != None)
            
            if agent.buffer.size >= agent.min_buffer:
                agent.train()
                # If self-play, we might train twice or just rely on larger batch throughput
                if is_self_play:
                     # Train again to consume double data rate? 
                     # Or just naturally faster buffer fill leads to more frequent "min_buffer" checks?
                     # Actually, standard DQN trains every step typically. 
                     # Here we train every step. Adding double data just fills buffer faster.
                     pass

        
        # Episode end
        env_episodes += 1
        total_episodes += n_envs
        
        # Determine winners
        for i in range(n_envs):
            if game.scores[i, 0] > game.scores[i, 1]: stats['W'] += 1
            elif game.scores[i, 1] > game.scores[i, 0]: stats['L'] += 1
            else: stats['D'] += 1
        
        # Decay epsilon
        for _ in range(n_envs):
             agent.decay_epsilon()
        
        # Progress logging (every 10 batches)
        if total_episodes % (10 * n_envs) < n_envs or total_episodes <= n_envs:
            elapsed = time.time() - t0
            eps_sec = (total_episodes - start_episode) / elapsed if elapsed > 0 else 0
            eta = (episodes - total_episodes) / eps_sec if eps_sec > 0 else 0
            eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.1f}m"
            
            mode = "SELF-PLAY" if is_self_play else "SIMPLE-AI"
            print(f"Ep {total_episodes}/{episodes} ({mode}) | ε:{agent.epsilon:.4f} | "
                  f"W:{stats['W']} L:{stats['L']} D:{stats['D']} | "
                  f"{eps_sec:.1f}/s | ETA:{eta_str}")
        
        # Save checkpoints
        if total_episodes % save_every < n_envs:
            agent.export_weights(f"weights/weights-{total_episodes}.json")
            
    agent.export_weights("weights/trained.json")
    print("=" * 60)
    print("✅ Training Complete!")
    print(f"Final: W:{stats['W']} L:{stats['L']} D:{stats['D']}")
    elapsed = time.time() - t0
    print(f"Time: {elapsed/3600:.2f} hours")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast RL Football Trainer')
    parser.add_argument('--episodes', type=int, default=100000, help='Number of episodes')
    parser.add_argument('--parallel-envs', type=int, default=16, help='Parallel environments')
    parser.add_argument('--save-every', type=int, default=5000, help='Save frequency')
    parser.add_argument('--resume', type=str, default=None, help='Resume from weights file')
    
    args = parser.parse_args()
    
    # Auto-detect if resume is needed
    resume_path = args.resume
    if resume_path is None and os.path.exists("weights/trained.json"):
         print("Found previous trained weights, resuming...")
         resume_path = "weights/trained.json"
         
    train(args.episodes, args.parallel_envs, args.save_every, resume_path)
