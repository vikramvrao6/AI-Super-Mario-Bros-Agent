
from environment import MarioEnvironment
from agent import MarioAgent
from utils import load_model
import time

env = MarioEnvironment()
agent = MarioAgent(input_size = 3840, action_size = 7)
load_model(agent)

agent.epsilon = 0.0

state = env.reset()

print("Watching trained agent play!")

while True:
    env.render()
    action = agent.act(state)
    state, reward, done, info = env.step(action)
    time.sleep(0.05)
    if done:
        print("Episode done! Restarting...")
        state = env.reset()