import os
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ale_py
import matplotlib.pyplot as plt
from collections import deque
import random

# Configuration parameters remain the same
seed = 42
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_max = 1.0
epsilon_interval = epsilon_max - epsilon_min
batch_size = 32
max_steps_per_episode = 10000
max_episodes = 10000

# Environment setup remains the same
os.makedirs("recorded_episodes", exist_ok=True)
gym.register_envs(ale_py)
env = gym.make(
    "ALE/Breakout-v5",
    render_mode="rgb_array",
    obs_type="grayscale",
    frameskip=1,
    repeat_action_probability=0.25,
)
env = RecordVideo(
    env,
    video_folder="recorded_episodes",
    episode_trigger=lambda x: x % 1000 == 0,
)
env = AtariPreprocessing(env, frame_skip=4, scale_obs=True)
env = FrameStackObservation(env, stack_size=4)

num_actions = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, maxlen):
        self.states = deque(maxlen=maxlen)
        self.actions = deque(maxlen=maxlen)
        self.rewards = deque(maxlen=maxlen)
        self.next_states = deque(maxlen=maxlen)
        self.dones = deque(maxlen=maxlen)

    def add(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def sample(self, batch_size):
        indices = random.sample(range(len(self.states)), batch_size)
        return (
            torch.FloatTensor(np.array([self.states[i] for i in indices])),
            torch.LongTensor([self.actions[i] for i in indices]),
            torch.FloatTensor([self.rewards[i] for i in indices]),
            torch.FloatTensor(np.array([self.next_states[i] for i in indices])),
            torch.FloatTensor([float(self.dones[i]) for i in indices]),
        )

    def __len__(self):
        return len(self.states)


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.network(x)


# Create models and optimizer
model = QNetwork().to(device)
model_target = QNetwork().to(device)
model_target.load_state_dict(model.state_dict())  # Initialize target network
optimizer = optim.Adam(model.parameters(), lr=0.00025)
loss_function = nn.SmoothL1Loss()  # Huber loss

# Experience replay buffers remain the same
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0

# Training parameters remain the same
epsilon_random_frames = 50000
epsilon_greedy_frames = 1000000.0
max_memory_length = 100000
update_after_actions = 4
update_target_network = 10000

# Start the training loop
for episode_count in range(max_episodes):
    observation, _ = env.reset()
    state = np.array(observation)
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        frame_count += 1

        # Epsilon-greedy action selection
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            action = np.random.choice(num_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_probs = model(state_tensor)
            action = torch.argmax(action_probs).item()

        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        state_next, reward, done, _, _ = env.step(action)
        state_next = np.array(state_next)
        episode_reward += reward

        # Save to replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # Sample a batch from the experience replay buffer
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            state_sample = torch.FloatTensor(
                np.array([state_history[i] for i in indices])
            ).to(device)
            state_next_sample = torch.FloatTensor(
                np.array([state_next_history[i] for i in indices])
            ).to(device)
            rewards_sample = torch.FloatTensor(
                [rewards_history[i] for i in indices]
            ).to(device)
            action_sample = torch.LongTensor([action_history[i] for i in indices]).to(
                device
            )
            done_sample = torch.FloatTensor(
                [float(done_history[i]) for i in indices]
            ).to(device)

            # Double DQN: Use the model for action selection and the target model for Q-value calculation
            with torch.no_grad():
                next_action = model(state_next_sample).max(1)[
                    1
                ]  # Action selected by policy model
                future_rewards = (
                    model_target(state_next_sample)
                    .gather(1, next_action.unsqueeze(1))
                    .squeeze()
                )  # Q-values from target model
                target_q_values = rewards_sample + gamma * future_rewards * (
                    1 - done_sample
                )

            current_q_values = (
                model(state_sample).gather(1, action_sample.unsqueeze(1)).squeeze()
            )
            loss = loss_function(current_q_values, target_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update the target network periodically
        if frame_count % update_target_network == 0:
            model_target.load_state_dict(model.state_dict())

        # Remove old experiences from the replay buffer if it exceeds max memory length
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    episode_reward_history.append(episode_reward)
    running_reward = np.mean(episode_reward_history)

    # Print relevant info after each episode
    print(f"Episode {episode_count + 1}/{max_episodes}")
    print(f"Episode Reward: {episode_reward:.2f}")
    print(f"Running Reward: {running_reward:.2f}")
    print(f"Epsilon: {epsilon:.2f}\n")

    # Check if we solved the environment (running reward > 40)
    if running_reward > 40:
        print("Solved at episode {}!".format(episode_count + 1))
        break

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot Episode Rewards
plt.subplot(1, 2, 1)
plt.plot(episode_reward_history)
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Episode Reward")

# Plot Running Reward (Total Running Average over all episodes)
plt.subplot(1, 2, 2)
plt.plot(
    np.cumsum(episode_reward_history) / (np.arange(len(episode_reward_history)) + 1)
)
plt.title("Total Running Reward (All Episodes)")
plt.xlabel("Episode")
plt.ylabel("Running Reward")

plt.tight_layout()
plt.show()

print("Training finished.")
