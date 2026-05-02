
import torch
import matplotlib.pyplot as plt

def save_model(agent, filepath='mario_model.pth'):
    torch.save(agent.model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def load_model(agent, filepath='mario_model.pth'):
    agent.model.load_state_dict(torch.load(filepath))
    print(f"Model loaded from {filepath}")

def plot_rewards(reward_history, save=True):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history)
    plt.title('Mario Agent Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('training_progress.png')
    plt.close()
    print("Plot saved as training_progress.png")