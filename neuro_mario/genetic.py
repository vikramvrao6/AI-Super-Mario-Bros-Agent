import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
import torch
import torch.nn as nn

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


def evaluate(net, max_steps=5000, stall_limit=220, record_actions=False):
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")
    env = JoypadSpace(env, CUSTOM_ACTIONS)

    reset_out = env.reset()
    state = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    frames = [preprocess(state)] * 4
    total_reward = 0.0
    max_x = 0
    last_x = 0
    stall_steps = 0
    done = False
    steps = 0
    info = {}
    actions_taken = []

    while not done and steps < max_steps:
        action = get_action(net, frames, epsilon=0.03)

        if record_actions:
            actions_taken.append(action)

        step_out = env.step(action)

        if len(step_out) == 5:
            next_state, reward, terminated, truncated, info = step_out
            done = terminated or truncated
        else:
            next_state, reward, done, info = step_out[:4]

        frame = preprocess(next_state)
        frames.pop(0)
        frames.append(frame)

        x = info.get("x_pos", 0)
        total_reward += reward
        max_x = max(max_x, x)

        if x > last_x:
            last_x = x
            stall_steps = 0
        else:
            stall_steps += 1

        if stall_steps >= stall_limit:
            break

        steps += 1

    env.close()

    flag = info.get("flag_get", False)
    time_left = info.get("time", 0)

    fitness = float(max_x)
    fitness += total_reward * 0.1
    fitness += time_left * 0.02
    if flag:
        fitness += 5000.0

    return fitness, max_x, total_reward, flag, actions_taken


def get_weights(net):
    return np.concatenate([
        p.detach().cpu().numpy().ravel() for p in net.parameters()
    ])


def set_weights(net, weights):
    idx = 0
    for p in net.parameters():
        size = p.numel()
        chunk = weights[idx:idx + size].reshape(p.shape)
        tensor_chunk = torch.tensor(chunk, dtype=p.dtype, device=p.device)
        p.data.copy_(tensor_chunk)
        idx += size


def mutate(weights, mutation_rate=0.08, mutation_strength=0.03):
    new_weights = weights.copy()
    mask = np.random.random(len(weights)) < mutation_rate
    new_weights[mask] += np.random.randn(mask.sum()) * mutation_strength
    return new_weights


def crossover(w1, w2):
    mask = np.random.rand(len(w1)) < 0.5
    return np.where(mask, w1, w2)


def tournament_select(results, k=6):
    idxs = np.random.choice(len(results), size=min(k, len(results)), replace=False)
    subset = [results[i] for i in idxs]
    subset.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return subset[0][4]


def run_genetic():
    population_size = 100
    generations = 200
    elite_size = 20
    random_injection = 0.10
    num_actions = len(CUSTOM_ACTIONS)

    print(f"Using action set: {CUSTOM_ACTIONS}")
    print(f"Initializing population of {population_size} networks...")

    population = [MarioNet(num_actions) for _ in range(population_size)]

    best_ever_x = 0
    best_ever_fitness = -float("inf")
    best_ever_weights = None
    best_ever_actions = None

    stagnation = 0
    mutation_rate = 0.08
    mutation_strength = 0.03

    for gen in range(generations):
        print(
            f"\nGeneration {gen + 1}/{generations} | "
            f"mut_rate={mutation_rate:.3f} mut_str={mutation_strength:.3f}"
        )

        results = []

        for i, net in enumerate(population):
            fitness, max_x, reward, flag, actions_taken = evaluate(
                net,
                max_steps=5000,
                stall_limit=220,
                record_actions=True
            )

            results.append((fitness, max_x, reward, flag, i, actions_taken))

            print(
                f"  Agent {i + 1:3d}: fitness={fitness:8.1f} "
                f"x={max_x:4d} reward={reward:7.1f} flag={flag}"
            )

            if flag:
                print(f"LEVEL BEATEN by agent {i + 1} in generation {gen + 1}!")
                torch.save(net.state_dict(), "genetic_mario_best.pt")
                np.save("winning_actions.npy", np.array(actions_taken, dtype=np.int32))
                print("Saved winning action trace to winning_actions.npy")

        results.sort(key=lambda x: (x[1], x[0]), reverse=True)

        best_fitness, best_x, _, best_flag, best_idx, best_actions = results[0]

        improved = False

        if best_x > best_ever_x:
            best_ever_x = best_x
            best_ever_fitness = best_fitness
            best_ever_weights = get_weights(population[best_idx])
            best_ever_actions = best_actions
            improved = True
        elif best_x == best_ever_x and best_fitness > best_ever_fitness:
            best_ever_fitness = best_fitness
            best_ever_weights = get_weights(population[best_idx])
            best_ever_actions = best_actions
            improved = True

        if improved:
            stagnation = 0
            mutation_rate = 0.08
            mutation_strength = 0.03

            print(
                f"New best! fitness={best_ever_fitness:.1f} "
                f"x={best_ever_x} flag={best_flag}"
            )

            best_net = MarioNet(num_actions)
            set_weights(best_net, best_ever_weights)
            torch.save(best_net.state_dict(), "genetic_mario_best.pt")

            if best_ever_actions is not None:
                np.save("best_run_actions.npy", np.array(best_ever_actions, dtype=np.int32))
                print("Saved best run action trace to best_run_actions.npy")
        else:
            stagnation += 1
            if stagnation > 5:
                mutation_rate = min(0.14, mutation_rate * 1.1)
                mutation_strength = min(0.05, mutation_strength * 1.1)
                print("Stagnating! Increasing mutation for diversity.")

        print(
            f"Generation {gen + 1} best: fitness={best_fitness:.1f} x={best_x} | "
            f"All-time best x={best_ever_x} | Stagnation={stagnation}"
        )

        new_population = []

        if best_ever_weights is not None:
            champion = MarioNet(num_actions)
            set_weights(champion, best_ever_weights)
            new_population.append(champion)

        elites = [population[results[i][4]] for i in range(min(elite_size, len(results)))]
        for elite in elites:
            if len(new_population) < population_size:
                new_population.append(elite)

        while len(new_population) < population_size:
            if np.random.random() < random_injection:
                new_population.append(MarioNet(num_actions))
                continue

            if np.random.random() < 0.8:
                p1 = tournament_select(results)
                p2 = tournament_select(results)
                while p2 == p1:
                    p2 = tournament_select(results)

                child_w = crossover(
                    get_weights(population[p1]),
                    get_weights(population[p2])
                )
            else:
                p = tournament_select(results)
                child_w = get_weights(population[p]).copy()

            child_w = mutate(
                child_w,
                mutation_rate=mutation_rate,
                mutation_strength=mutation_strength
            )

            child_net = MarioNet(num_actions)
            set_weights(child_net, child_w)
            new_population.append(child_net)

        population = new_population

        if best_ever_weights is not None:
            best_net = MarioNet(num_actions)
            set_weights(best_net, best_ever_weights)
            torch.save(best_net.state_dict(), "genetic_mario_best.pt")
            print("Best model saved!")

            if best_ever_actions is not None:
                np.save("best_run_actions.npy", np.array(best_ever_actions, dtype=np.int32))

        if best_ever_x >= 3000:
            print("Reached target x >= 3000. Stopping early.")
            break


if __name__ == "__main__":
    run_genetic()
