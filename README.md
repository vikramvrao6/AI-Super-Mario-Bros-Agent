# Machine Learning: Reinforcement & Evolutionary AI Agents (Super Mario Bros NES)

Created three different approaches to beating World 1-1 of Super Mario Bros with no human demonstrations or hardcoded rules. Built this to understand reinforcement learning from scratch. DQN, PPO, and a genetic algorithm, each implemented without high-level RL libraries.

![Mario Agent Demo](demo.gif)

---

## What actually worked

The genetic algorithm beat 1-1 in 9 generations (~30 minutes). No gradients, just evolving a population of 100 neural networks through selection and mutation until one of them figured it out.

DQN and PPO are still training. Gradient-based methods are slower to converge on this environment than I expected — the sparse reward signal makes early exploration brutal.

| Approach | Outcome |
|---|---|
| Genetic Algorithm | Completed World 1-1 in 9 generations |
| PPO | Training |
| DQN | Baseline / Training |

---

## The three approaches

### Genetic Algorithm (`genetic.py`)

No backpropagation. Each generation, 100 networks play the level and get scored on how far they get. The best ones reproduce — their weights are combined via uniform crossover and mutated slightly. Repeat.

The interesting part is adaptive mutation: if the population stagnates for 5+ generations without improving, mutation rate and strength both increase automatically to escape local optima. That's what got it past the first big gap in the level.

Fitness function: `x_position + 0.1 * total_reward + 0.02 * time_remaining + 5000 * flag_get`

- Population: 100 networks
- Selection: Tournament (k=6)
- Elite carry-forward: Top 20 each generation
- Random injection: 10% fresh networks per generation to maintain diversity

### PPO (`ppo.py`)

Actor-Critic with GAE and clipped surrogate objectives. The clipping keeps policy updates from being too aggressive — without it the policy collapses pretty fast on this environment.

Reward is shaped with sign-preserving square root normalization to handle the wide range of raw reward values without crushing the signal.

- Rollout: 128 steps per update
- GAE: λ=0.95, γ=0.99
- Clip ε: 0.2
- Epochs per rollout: 4
- Grad clipping: 0.5

### DQN (`main.py`, `agent.py`, `model.py`, `train.py`)

Standard DQN — experience replay buffer, target network, epsilon-greedy exploration. The target network syncs every 10 steps which helps stabilize training but convergence is slow.

- Replay buffer: 10k transitions
- Batch size: 32
- γ: 0.9
- ε decay: 1.0 → 0.01 at rate 0.995
- Optimizer: Adam (lr=1e-4)

---

## Network architecture

All three use the same CNN backbone — 4 stacked grayscale frames in, action logits out.

```
(4, 42, 42) → Conv(32, 8x8, s4) → Conv(64, 4x4, s2) → Conv(64, 3x3, s1) → Linear(8448, 512) → Linear(512, n_actions)
```

Frames are converted to grayscale, downsampled 2x, normalized to [0,1], and stacked in groups of 4. Stacking gives the network enough temporal context to infer velocity and jump state from static frames.

Action space is restricted to 4 actions: `right`, `right+A`, `right+B`, `right+A+B`. The full action space has too many redundant options that slow down learning.

---

## Files

```
main.py               DQN training loop
agent.py              DQN agent — memory, act, epsilon updates
model.py              CNN architecture
environment.py        Gym wrapper + preprocessing
train.py              DQN training step (sample, compute loss, backprop)
utils.py              Save/load model, plot rewards
genetic.py            Full genetic algorithm
ppo.py                PPO with Actor-Critic
demo.py               Watch trained DQN agent play
playbestgenetic.py    Watch best genetic agent play
```

---

## Setup

```bash
pip install torch gym-super-mario-bros nes-py numpy matplotlib
```

**Train:**
```bash
python genetic.py       # genetic algorithm
python ppo.py           # PPO
python main.py          # DQN
```

**Watch:**
```bash
python playbestgenetic.py   # best genetic agent
python demo.py              # trained DQN agent
```

Trained on Google Colab A100. If you're running locally, MPS works on Apple Silicon and CUDA on NVIDIA — both are auto-detected.

---

## Notes

The genetic algorithm outperforming gradient-based methods this early was surprising. DQN and PPO both struggle with exploration when rewards are sparse and delayed, which is exactly the case here. The genetic algorithm sidesteps this entirely by evaluating complete rollouts and optimizing directly for x-position.

