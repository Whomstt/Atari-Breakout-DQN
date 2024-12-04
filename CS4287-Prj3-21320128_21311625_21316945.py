import gymnasium as gym
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py


class DQN(nn.Module):
    def __init__(self, nb_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, nb_actions),
        )

    def forward(self, x):
        return self.network(x / 255.0)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def Deep_Q_Learning(
    env,
    replay_memory_size=1_000_000,
    nb_epochs=100_000,
    update_frequency=4,
    batch_size=32,
    discount_factor=0.99,
    replay_start_size=10_000,
    initial_exploration=1,
    final_exploration=0.01,
    exploration_steps=1_000_000,
    device="cuda",
):
    # Initialize replay memory D to capacity N
    rb = ReplayBuffer(replay_memory_size)

    # Initialize action-value function Q with random weights
    q_network = DQN(env.action_space.n).to(device)
    target_network = DQN(env.action_space.n).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = torch.optim.Adam(q_network.parameters(), lr=1.25e-4)

    epoch = 0
    smoothed_rewards = []
    rewards = []

    progress_bar = tqdm(total=nb_epochs)
    while epoch <= nb_epochs:
        terminated = truncated = False
        total_rewards = 0

        # Initialise sequence s1 = {x1} and preprocessed sequenced φ1 = φ(s1)
        obs, _ = env.reset()

        while not (terminated or truncated):
            epsilon = max(
                (final_exploration - initial_exploration) / exploration_steps * epoch
                + initial_exploration,
                final_exploration,
            )
            if random.random() < epsilon:  # With probability ε select a random action a
                action = env.action_space.sample()
            else:  # Otherwise select a = max_a Q∗(φ(st), a; θ)
                q_values = q_network(torch.Tensor(obs).unsqueeze(0).to(device))
                action = torch.argmax(q_values, dim=1).item()

            # Execute action a in emulator and observe reward rt and image xt+1
            next_obs, reward, terminated, truncated, info = env.step(action)

            total_rewards += reward

            # Set st+1 = st, at, xt+1 and preprocess φt+1 = φ(st+1)
            rb.push(obs, action, reward, next_obs, terminated or truncated)

            obs = next_obs

            if epoch > replay_start_size and epoch % update_frequency == 0:
                # Sample random minibatch of transitions (φj , aj , rj , φj +1 ) from D
                batch = rb.sample(batch_size)

                states, actions, rewards_batch, next_states, dones = zip(*batch)

                states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
                next_states = torch.tensor(
                    np.array(next_states), dtype=torch.float32
                ).to(device)
                actions = torch.tensor(actions, dtype=torch.long).to(device)
                rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).to(
                    device
                )
                dones = torch.tensor(dones, dtype=torch.float32).to(device)

                with torch.no_grad():
                    max_q_value, _ = target_network(next_states).max(dim=1)
                    y = rewards_batch + discount_factor * max_q_value * (1 - dones)

                current_q_value = (
                    q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
                )
                loss = F.huber_loss(y, current_q_value)

                # Perform a gradient descent step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Soft update of target network
                for target_param, param in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_param.data.copy_(
                        0.995 * target_param.data + 0.005 * param.data
                    )

            epoch += 1
            if (epoch % 10_000 == 0) and epoch > 0:
                smoothed_rewards.append(np.mean(rewards))
                rewards = []
            progress_bar.update(1)
        rewards.append(total_rewards)
    plt.plot(smoothed_rewards)
    plt.title("Average Reward on Breakout")
    plt.xlabel("Training Epochs")
    plt.ylabel("Average Reward per Episode")
    plt.show()


if __name__ == "__main__":
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5")
    env = AtariPreprocessing(env, frame_skip=1)
    env = FrameStackObservation(env, 4)

    Deep_Q_Learning(env, device="cuda")
    env.close()
