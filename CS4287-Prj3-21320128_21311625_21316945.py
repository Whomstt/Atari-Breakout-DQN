import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo, AtariPreprocessing, FrameStackObservation
from typing import Tuple, List
import logging
import ale_py

# Adjustable parameters
NUM_EPISODES = 1000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 250000
TARGET_UPDATE = 5
BUFFER_CAPACITY = 100000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 1e-4
INPUT_SHAPE = (4, 84, 84)
VIDEO_FOLDER = "recorded_episodes"
VIDEO_EPISODE_TRIGGER = 100
MAX_STEPS_PER_EPISODE = 10000
PRIORITY_ALPHA = 0.6
PRIORITY_BETA_START = 0.4
PRIORITY_BETA_FRAMES = 50000

# Environment setup
os.makedirs(VIDEO_FOLDER, exist_ok=True)
gym.register_envs(ale_py)
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
env = RecordVideo(
    env,
    video_folder=VIDEO_FOLDER,
    episode_trigger=lambda x: x % VIDEO_EPISODE_TRIGGER == 0,
)
env = AtariPreprocessing(env, frame_skip=1)  # Env already using default frame_skip 4
env = FrameStackObservation(env, 4)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class DuelingDQN(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int], num_actions: int):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.feature_size(input_shape), 512)
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, num_actions)

    def feature_size(self, input_shape: Tuple[int, int, int]) -> int:
        return (
            self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_shape))))
            .view(1, -1)
            .size(1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean()


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]
        probs = prios**self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        batch = list(zip(*samples))
        states = np.stack(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.stack(batch[3])
        dones = np.array(batch[4])
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices: List[int], batch_priorities: np.ndarray):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self) -> int:
        return len(self.buffer)


class DDQNAgent:
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_actions: int,
        buffer_capacity: int = BUFFER_CAPACITY,
        batch_size: int = BATCH_SIZE,
        gamma: float = GAMMA,
        lr: float = LR,
        alpha: float = PRIORITY_ALPHA,
        beta_start: float = PRIORITY_BETA_START,
        beta_frames: int = PRIORITY_BETA_FRAMES,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingDQN(input_shape, num_actions).to(self.device)
        self.target_net = DuelingDQN(input_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.99
        )
        self.memory = PrioritizedReplayBuffer(buffer_capacity, alpha)
        self.batch_size = batch_size
        self.gamma = gamma
        self.num_actions = num_actions
        self.steps_done = 0
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.best_avg_reward = -float("inf")

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.tensor(
                    state, device=self.device, dtype=torch.float32
                ).unsqueeze(0)
                return self.policy_net(state).max(1)[1].item()
        else:
            return random.randrange(self.num_actions)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        beta = (
            self.beta_start
            + self.steps_done * (1.0 - self.beta_start) / self.beta_frames
        )
        states, actions, rewards, next_states, dones, indices, weights = (
            self.memory.sample(self.batch_size, beta)
        )
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)
        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        state_action_values = (
            self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )
        next_state_actions = self.policy_net(next_states).max(1)[1]
        next_state_values = (
            self.target_net(next_states)
            .gather(1, next_state_actions.unsqueeze(1))
            .squeeze(1)
        )
        expected_state_action_values = rewards + (
            self.gamma * next_state_values * (1 - dones)
        )
        loss = (state_action_values - expected_state_action_values.detach()).pow(
            2
        ) * weights
        prios = loss + 1e-5
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        self.scheduler.step()
        self.memory.update_priorities(indices, prios.data.cpu().numpy())

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path: str):
        torch.save(
            {
                "model_state_dict": self.policy_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def train_agent(agent: DDQNAgent, env, num_episodes: int):
    total_rewards = []
    running_avg_rewards = []
    running_avg_window = 100

    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        for t in range(MAX_STEPS_PER_EPISODE):
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(
                -1.0 * agent.steps_done / EPSILON_DECAY
            )
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.optimize_model()
            agent.steps_done += 1
            if done:
                break
        if episode % TARGET_UPDATE == 0:
            agent.update_target_net()
        total_rewards.append(total_reward)
        running_avg_rewards.append(np.mean(total_rewards[-running_avg_window:]))
        logging.info(
            f"Episode {episode}, Total Reward: {total_reward}, Running Avg Reward: {running_avg_rewards[-1]:.3f}"
        )
        if running_avg_rewards[-1] > agent.best_avg_reward:
            agent.best_avg_reward = running_avg_rewards[-1]
            agent.save_model(f"best_ddqn_model.pth")
        if episode % 100 == 0:
            agent.save_model(f"ddqn_model_{episode}.pth")

    return total_rewards, running_avg_rewards


num_actions = env.action_space.n
agent = DDQNAgent(INPUT_SHAPE, num_actions)

total_rewards, running_avg_rewards = train_agent(agent, env, NUM_EPISODES)

env.close()

# Plotting the total rewards and running average rewards
plt.plot(total_rewards, label="Total Rewards")
plt.plot(running_avg_rewards, label="Running Avg Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Total and Running Average Rewards per Episode")
plt.legend()
plt.show()
