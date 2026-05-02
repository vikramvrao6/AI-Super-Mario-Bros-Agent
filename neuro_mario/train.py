
import torch
import numpy as np
import random

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def train(agent, batch_size=32):
    if len(agent.memory) < batch_size:
        return

    batch = random.sample(agent.memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(dones).to(device)

    current_q = agent.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = agent.target_model(next_states).max(1)[0]

    target_q = rewards + (1 - dones) * agent.gamma * next_q

    loss = agent.loss_fn(current_q, target_q)
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    agent.steps += 1
    if agent.steps % agent.target_update == 0:
        agent.update_target()

    return loss.item()