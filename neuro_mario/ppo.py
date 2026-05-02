
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

def make_env():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env

def preprocess(state):
    gray = np.mean(state, axis=2)
    resized = gray[::2, ::2]
    normalized = resized / 255.0
    return normalized

def get_frames(frames):
    return np.array(frames)

class ActorCritic(nn.Module):
    def __init__(self, num_frames, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(8448, 512)
        self.policy = nn.Linear(512, num_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.policy(x), self.value(x).squeeze(-1)

    def act(self, state):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value

def train_ppo():
    env = make_env()
    num_actions = 7
    num_frames = 4

    model = ActorCritic(num_frames, num_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)

    rollout_steps = 128
    epochs = 4
    batch_size = 32
    clip_eps = 0.2
    gamma = 0.99
    lam = 0.95

    state = env.reset()
    frame = preprocess(state)
    frames = deque([frame] * num_frames, maxlen=num_frames)

    episode = 0
    total_steps = 0

    print("Starting PPO training!")

    while True:
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []

        for _ in range(rollout_steps):
            obs = torch.FloatTensor(get_frames(frames)).unsqueeze(0).to(device)

            with torch.no_grad():
                action, logprob, value = model.act(obs)

            next_state, reward, done, info = env.step(action.item())[:4]
            reward = np.sign(reward) * (np.sqrt(abs(reward) + 1) - 1) + 0.001 * reward

            obs_buf.append(get_frames(frames).copy())
            act_buf.append(action.item())
            logp_buf.append(logprob.item())
            val_buf.append(value.item())
            rew_buf.append(reward)
            done_buf.append(float(done))

            next_frame = preprocess(next_state)
            frames.append(next_frame)
            total_steps += 1

            if done:
                episode += 1
                state = env.reset()
                frame = preprocess(state)
                frames = deque([frame] * num_frames, maxlen=num_frames)

        obs = torch.FloatTensor(get_frames(frames)).unsqueeze(0).to(device)
        with torch.no_grad():
            _, last_value = model.forward(obs)

        advantages = []
        gae = 0
        values_extended = val_buf + [last_value.item()]

        for t in reversed(range(rollout_steps)):
            not_done = 1.0 - done_buf[t]
            delta = rew_buf[t] + gamma * values_extended[t+1] * not_done - values_extended[t]
            gae = delta + gamma * lam * not_done * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, val_buf)]

        obs_t = torch.FloatTensor(np.array(obs_buf)).to(device)
        act_t = torch.LongTensor(act_buf).to(device)
        logp_t = torch.FloatTensor(logp_buf).to(device)
        adv_t = torch.FloatTensor(advantages).to(device)
        ret_t = torch.FloatTensor(returns).to(device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(epochs):
            idx = torch.randperm(rollout_steps)
            for start in range(0, rollout_steps, batch_size):
                i = idx[start:start+batch_size]
                logits, value = model(obs_t[i])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act_t[i])
                ratio = torch.exp(logp - logp_t[i])
                clipped = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv_t[i]
                policy_loss = -torch.min(ratio * adv_t[i], clipped).mean()
                value_loss = (ret_t[i] - value).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        avg_return = np.mean(returns)
        print(f"Episode {episode} | Steps {total_steps} | Avg Return {avg_return:.2f}")

        if episode % 50 == 0:
            torch.save(model.state_dict(), 'ppo_mario.pt')
            print("Model saved!")

if __name__ == "__main__":
    train_ppo()