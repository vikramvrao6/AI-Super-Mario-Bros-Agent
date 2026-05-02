
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np

class MarioEnvironment:
    def __init__(self, num_frames=4):
        self.env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        self.num_frames = num_frames
        self.frames = []

    def preprocess(self, state):
        gray = np.mean(state, axis=2)
        resized = gray[::2, ::2]
        normalized = resized / 255.0
        return normalized

    def reset(self):
        state = self.env.reset()
        frame = self.preprocess(state)
        self.frames = [frame] * self.num_frames
        return np.array(self.frames)

    def step(self, action):
        state, reward, done, info = self.env.step(action)[:4]
        frame = self.preprocess(state)
        self.frames.pop(0)
        self.frames.append(frame)
        reward = reward / 15.0
        return np.array(self.frames), reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def sample_action(self):
        return self.env.action_space.sample()