import time
import numpy as np
import torch
import torch.nn as nn
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

CUSTOM_ACTIONS = [
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
]


class MarioNet(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 13 * 14, 16),
            nn.ReLU(),
            nn.Linear(16, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


def preprocess(state):
    gray = np.mean(state, axis=2).astype(np.float32)
    resized = gray[::2, ::2]
    return resized / 255.0


def get_action(net, frames, epsilon=0.03):
    if np.random.random() < epsilon:
        return np.random.randint(net.fc[-1].out_features)

    obs = torch.tensor(np.array(frames), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = net(obs)
    return logits.argmax(dim=1).item()


def main():
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, CUSTOM_ACTIONS)

    net = MarioNet(len(CUSTOM_ACTIONS))
    net.load_state_dict(torch.load("genetic_mario_best.pt", map_location="cpu"))
    net.eval()

    reset_out = env.reset()
    state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    frames = [preprocess(state)] * 4

    done = False
    steps = 0

    while not done:
        env.render()

        action = get_action(net, frames, epsilon=0.03)
        step_out = env.step(action)

        if len(step_out) == 5:
            next_state, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            next_state, reward, done, info = step_out[:4]

        frame = preprocess(next_state)
        frames.pop(0)
        frames.append(frame)

        steps += 1
        print(
            f"step={steps} action={CUSTOM_ACTIONS[action]} "
            f"x={info.get('x_pos', 0)} flag={info.get('flag_get', False)}"
        )

        time.sleep(0.02)

    env.close()


if __name__ == "__main__":
    main()
