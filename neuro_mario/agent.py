
import random
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from model import MarioModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MarioAgent:
    def __init__(self, action_size, num_frames=4):
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.steps = 0
        self.model = MarioModel(num_frames, action_size).to(device)
        self.target_model = MarioModel(num_frames, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = self.model(state_tensor)
        return q_values.argmax().item()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay