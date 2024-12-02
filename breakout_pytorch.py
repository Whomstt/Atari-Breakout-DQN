# References
# https://keras.io/examples/rl/deep_q_network_breakout/
# https://ale.farama.org/environments/breakout/
# https://gymnasium.farama.org/
# https://github.com/KJ-Waller/DQN-PyTorch-Breakout/blob/master/Breakout/DQN_model.py

import os
import gymnasium as gym
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStackObservation,
    RecordVideo,
)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import ale_py

# Configuration parameters
seed = 42
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = epsilon_max - epsilon_min
batch_size = 32
max_steps_per_episode = 10000
max_episodes = 10
video_folder = "recorded_episodes"

os.makedirs(video_folder, exist_ok=True)

gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder=video_folder,
    episode_trigger=lambda x: x % 10 == 0,
)
env = AtariPreprocessing(env, frame_skip=1)
env = FrameStackObservation(env, 4)

env.reset(seed=seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_actions = 4
input_dim = (4, 84, 84)
output_dim = num_actions


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        channels, _, _ = input_dim

        # Three convolutional layers
        self.l1 = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Fully connected layers
        conv_output_size = self.conv_output_dim()
        lin1_output_size = 512

        # Two linear layers
        self.l2 = nn.Sequential(
            nn.Linear(conv_output_size, lin1_output_size),
            nn.ReLU(),
            nn.Linear(lin1_output_size, output_dim),
        )

    # Returns the output dimension of the convolutional layers
    def conv_output_dim(self):
        x = torch.zeros(1, *self.input_dim)
        x = self.l1(x)
        return int(np.prod(x.shape))

    # Forward pass
    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.shape[0], -1)
        actions = self.l2(x)

        return actions


model = DQN(input_dim, output_dim).to(device)
model_target = DQN(input_dim, output_dim).to(device)
model_target.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=0.00025)

# Experience replay buffers
action_history, state_history, state_next_history = [], [], []
rewards_history, done_history = [], []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0

# Hyperparameters
epsilon_random_frames = 50000
epsilon_greedy_frames = 1000000.0
max_memory_length = 100000
update_after_actions = 4
update_target_network = 10000

while True:
    observation, _ = env.reset()
    state = np.array(observation)
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        frame_count += 1

        # Epsilon-greedy exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            action = np.random.choice(num_actions)
        else:
            with torch.no_grad():
                state_tensor = (
                    torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                )
                action_probs = model(state_tensor)
                action = action_probs.argmax().cpu().item()

        # Decay exploration
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        # Environment step
        state_next, reward, done, _, _ = env.step(action)
        state_next = np.array(state_next)

        episode_reward += reward

        # Save experiences
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # Update network
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            # Sample batch
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Prepare batch tensors
            state_sample = torch.tensor(
                np.array([state_history[i] for i in indices]), dtype=torch.float32
            ).to(device)
            state_next_sample = torch.tensor(
                np.array([state_next_history[i] for i in indices]), dtype=torch.float32
            ).to(device)
            rewards_sample = torch.tensor(
                [rewards_history[i] for i in indices], dtype=torch.float32
            ).to(device)
            action_sample = torch.tensor(
                [action_history[i] for i in indices], dtype=torch.long
            ).to(device)
            done_sample = torch.tensor(
                [float(done_history[i]) for i in indices], dtype=torch.float32
            ).to(device)

            # Predict future rewards
            with torch.no_grad():
                future_rewards = model_target(state_next_sample)
                updated_q_values = rewards_sample + gamma * future_rewards.max(1)[0] * (
                    1 - done_sample
                )

            # Compute Q-values
            q_values = model(state_sample)
            q_action = q_values.gather(1, action_sample.unsqueeze(1)).squeeze(1)

            # Compute loss
            loss = F.smooth_l1_loss(q_action, updated_q_values)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update target network
        if frame_count % update_target_network == 0:
            model_target.load_state_dict(model.state_dict())
            print(
                f"running reward: {running_reward:.2f} at episode {episode_count}, frame count {frame_count}"
            )

        # Trim memory
        if len(rewards_history) > max_memory_length:
            for history in [
                rewards_history,
                state_history,
                state_next_history,
                action_history,
                done_history,
            ]:
                del history[:1]

        if done:
            break

    # Update running reward
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    # Termination conditions
    if running_reward > 40:
        print(f"Solved at episode {episode_count}!")
        break

    if max_episodes > 0 and episode_count >= max_episodes:
        print(f"Stopped at episode {episode_count}!")
        break

env.close()
