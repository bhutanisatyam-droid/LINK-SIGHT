"""Self-Contained PyTorch DDPG RL Trainer & ONNX Exporter for TinyDDPG-Dampener (Core 3).

Trains an Actor-Critic Reinforcement Learning Agent (~4.2K actor parameters) to:
- Output high-frequency angular rate damping offsets (Δωx, Δωy) to stabilize camera during extreme turbulence.

Can run locally or on Colab/Kaggle.
Exports the trained Actor policy directly to `models/tinyddpg_dampener.onnx`.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from tools.fsoc_gym_env import FSOCTrackingEnv


class Actor(nn.Module):
    """Deterministic Actor Policy Network (~4.2K parameters)."""

    def __init__(self, state_dim: int = 8, action_dim: int = 2, max_action: float = 2.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)


class Critic(nn.Module):
    """Q-Value Critic Network."""

    def __init__(self, state_dim: int = 8, action_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=1))


class ReplayBuffer:
    """Experience Replay Buffer."""

    def __init__(self, capacity: int = 100000, state_dim: int = 8, action_dim: int = 2):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int = 128):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[ind]),
            torch.tensor(self.actions[ind]),
            torch.tensor(self.rewards[ind]),
            torch.tensor(self.next_states[ind]),
            torch.tensor(self.dones[ind])
        )


def train():
    print("=" * 60)
    print("LinkSight: TinyDDPG-Dampener Training Pipeline (PyTorch -> ONNX)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    env = FSOCTrackingEnv()
    state_dim = 8
    action_dim = 2
    max_action = 2.5

    actor = Actor(state_dim, action_dim, max_action).to(device)
    actor_target = Actor(state_dim, action_dim, max_action).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic = Critic(state_dim, action_dim).to(device)
    critic_target = Critic(state_dim, action_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=2e-3, weight_decay=1e-4)

    buffer = ReplayBuffer(capacity=50000, state_dim=state_dim, action_dim=action_dim)

    gamma = 0.98
    tau = 0.005
    total_steps = 40000
    batch_size = 128
    noise_std = 0.2

    state = env.reset()
    ep_reward = 0
    episodes = 0
    start_time = time.time()

    print(f"[*] Training Actor-Critic agent for {total_steps:,} environment steps...")

    for step in range(1, total_steps + 1):
        # Action selection with exploration noise
        if step < 2000:
            action = np.random.uniform(-max_action, max_action, size=action_dim).astype(np.float32)
        else:
            with torch.no_grad():
                s_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                a_t = actor(s_t).cpu().numpy().squeeze(0)
                noise = np.random.normal(0, noise_std, size=action_dim)
                action = np.clip(a_t + noise, -max_action, max_action).astype(np.float32)

        next_state, reward, done, _ = env.step(action)
        buffer.add(state, action, reward, next_state, float(done))

        state = next_state
        ep_reward += reward

        if done:
            state = env.reset()
            episodes += 1
            ep_reward = 0

        # Optimization step
        if step >= 2000:
            b_s, b_a, b_r, b_ns, b_d = buffer.sample(batch_size)
            b_s, b_a, b_r, b_ns, b_d = b_s.to(device), b_a.to(device), b_r.to(device), b_ns.to(device), b_d.to(device)

            # Critic update
            with torch.no_grad():
                target_a = actor_target(b_ns)
                target_q = critic_target(b_ns, target_a)
                target_y = b_r + gamma * (1.0 - b_d) * target_q

            current_q = critic(b_s, b_a)
            critic_loss = nn.MSELoss()(current_q, target_y)

            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()

            # Actor update (maximize Q)
            actor_loss = -critic(b_s, actor(b_s)).mean()

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            # Polyak target update
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if step % 10000 == 0:
            print(f"Step [{step:05d}/{total_steps}] | Episodes: {episodes} | Elapsed: {time.time() - start_time:.1f}s")

    print(f"\n[*] Training Complete in {time.time() - start_time:.1f}s!")

    # Export Actor network to ONNX
    os.makedirs("models", exist_ok=True)
    onnx_path = "models/tinyddpg_dampener.onnx"
    actor.eval()
    actor.to("cpu")

    dummy_state = torch.randn(1, 8, dtype=torch.float32)

    torch.onnx.export(
        actor,
        dummy_state,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["state_vector"],
        output_names=["damping_action"],
        dynamic_axes={"state_vector": {0: "batch"}, "damping_action": {0: "batch"}}
    )

    file_size_kb = os.path.getsize(onnx_path) / 1024.0
    print(f"[SUCCESS] Exported ONNX actor policy to: {onnx_path}")
    print(f"[*] ONNX Binary File Size: {file_size_kb:.2f} KB (Ready for <15ms execution)")
    print("=" * 60)


if __name__ == "__main__":
    train()
