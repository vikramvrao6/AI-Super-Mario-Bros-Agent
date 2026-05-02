import time
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

CUSTOM_ACTIONS = [
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
]

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
env = JoypadSpace(env, CUSTOM_ACTIONS)

actions = np.load("best_run_actions.npy")

reset_out = env.reset()
state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

for i, action in enumerate(actions):
    env.render()
    step_out = env.step(int(action))

    if len(step_out) == 5:
        _, _, terminated, truncated, info = step_out
        done = terminated or truncated
    else:
        _, _, done, info = step_out[:4]

    print(f"step={i} action={CUSTOM_ACTIONS[int(action)]} x={info.get('x_pos', 0)} flag={info.get('flag_get', False)}")
    time.sleep(0.02)

    if done:
        break

env.close()

