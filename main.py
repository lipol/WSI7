from environment import Environment
import numpy as np
import random
import time

fileName = 'lake.dat'
env = Environment(fileName)
print(f'Loaded {fileName} environment.')
q_table = np.zeros((env.states, env.actions))


# Algorithm parameters
num_episodes = 10000
max_steps = 100

learning_rate = 0.5
discount_rate = 0.98

exploration_rate = 1.0
max_exploration_rate = 1.0
min_exploration_rate = 0.0
exploration_decay_rate = 0.001

rewards = []

for episode in range(1, num_episodes + 1):
    state = env.reset()
    done = False
    episode_reward = 0

    for step in range(max_steps):
        # Exploration to Exploitation trade-off
        exploration_rate_treshold = random.uniform(0, 1)
        if exploration_rate_treshold > exploration_rate and not np.all(q_table[state, :] == 0):
            action = np.argmax(q_table[state, :])  # Exploitation -> action with maximum q-value
        else:
            action = env.sampleAction()  # Exploration -> random action

        new_state, reward, done = env.step(action)  # Perform an action

        # Update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) \
            + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        episode_reward += reward

        if done == True:
            break

    # Exploration rate decay -> exploration rate decrase at a rate propotional to its current value
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

    rewards.append(episode_reward)
    if episode % 1000 == 0 and episode != 0:
        print(f'Avg reward after {episode} episodes: {round(np.sum(rewards)/episode,2)}')
    #print('Downloading File FooFile.txt [%d%%]\r' % i, end="")

# print(q_table)
# time.sleep(3)

# Vizualization
env.initCurses()
num_episodes = 2
for episode in range(1, num_episodes + 1):
    state = env.reset()
    done = False

    for step in range(max_steps):
        env.render()
        time.sleep(0.2)
        action = np.argmax(q_table[state, :])
        new_state, reward, done = env.step(action)

        if done:
            env.render()
            time.sleep(1)
            break

        state = new_state

time.sleep(1)
env.deInitCurses()
