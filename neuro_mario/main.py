
from environment import MarioEnvironment
from agent import MarioAgent
from train import train
from utils import save_model, plot_rewards

env = MarioEnvironment()
agent = MarioAgent(action_size = 7)

state = env.reset()

episodes = 0
steps_per_episode = 0
max_steps = 1000
total_reward = 0
reward_history = []

for step in range (10000000):
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    agent.remember(state, action, reward, next_state, done)

    loss = train(agent)
    state = next_state
    steps_per_episode += 1
    total_reward += reward

    if done or steps_per_episode >= max_steps:
        episodes += 1
        agent.update_epsilon()
        reward_history.append(total_reward)
        if loss:
            print(f"Episode {episodes} done! Epsilon: {agent.epsilon:.4f} Loss: {loss:.4f} Reward: {total_reward:.2f}")
        else:
            print(f"Episode {episodes} done! Epsilon: {agent.epsilon:.4f} Loss: N/A Reward: {total_reward:.2f}")

        if episodes % 10 == 0:
            save_model(agent)
            plot_rewards(reward_history)

        state = env.reset()
        total_reward = 0
        steps_per_episode = 0

env.close()
