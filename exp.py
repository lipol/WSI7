import numpy as np
import gym
import random
import time
#from IPython.display import clear_output
import os

env = gym.make("FrozenLake8x8-v0")
#env = gym.make("FrozenLake8x8-v0", is_slippery=False)

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))
# print(q_table)

num_episodes = 10000
max_steps = 100

learning_rate = 0.5  # alpha or lr
discount_rate = 0.98  # gamma or y

# Start with 1 cause first move need to be random
exploration_rate = 1.0  # epsilon
max_exploration_rate = 1.0
min_exploration_rate = 0.0
exploration_decay_rate = 0.001

rewards = []

for episode in range(1, num_episodes + 1):
    state = env.reset()

    done = False
    rewards_current_episode = 0

    for step in range(max_steps):
        # Exploration to Exploitation trade-off
        exploration_rate_treshold = random.uniform(0, 1)
        if exploration_rate_treshold > exploration_rate and not np.all(q_table[state, :] == 0):
            action = np.argmax(q_table[state, :])  # Exploitation -> action with maximum q-value
        else:
            action = env.action_space.sample()  # Exploration -> random action

        new_state, reward, done, info = env.step(action)  # Perform an action

        # Update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) \
            + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        rewards_current_episode += reward

        if done == True:
            break

    # Exploration rate decay -> exploration rate decrase at a rate propotional to its current value
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

    rewards.append(rewards_current_episode)
    if episode % 1000 == 0 and episode != 0:
        print(f'Avg reward after {episode} episodes: {round(np.sum(rewards)/episode,2)}')

print('')

# Calc and print the average reward per certain number of episodes
# Let's call them epochs
epoch_size = 1000
epoch_rewards = np.split(np.array(rewards), num_episodes/epoch_size)
for i, r in enumerate(epoch_rewards):
    print(f'Avg reward in {i+1} epoch: {round(np.sum(r/epoch_size),2)}')

print('\nQ-Table')
print(q_table)

# Visualize agant moves after learning process
for episode in range(1, 4):
    state = env.reset()
    done = False
    print(f'Episode {episode}\n')
    time.sleep(1)

    for step in range(max_steps):
        os.system('cls')
        env.render()
        time.sleep(0.2)

        action = np.argmax(q_table[state, :])
        new_state, reward, done, info = env.step(action)
        # os.system('cls')
        # env.render()

        if done:
            if reward == 1:
                os.system('cls')
                env.render()
                print('You reached the goal!')
                time.sleep(1)
            else:
                os.system('cls')
                env.render()
                print('You failed!')
                time.sleep(1)
            break

        state = new_state

env.close()
