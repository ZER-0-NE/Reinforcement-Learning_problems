'''
FrozenLake-v0
The agent controls the movement of a character in a grid world. 
Some tiles of the grid are walkable, and others lead to the agent falling into the water.
Additionally, the movement direction of the agent is uncertain and only partially depends on 
the chosen direction.The agent is rewarded for finding a walkable path to a goal tile.

Winter is here. You and your friends were tossing around a frisbee at the park when you made a 
wild throw that left the frisbee out in the middle of the lake. The water is mostly frozen,
but there are a few holes where the ice has melted. If you step into one of those holes,
you'll fall into the freezing water. At this time, there's an international frisbee shortage,
so it's absolutely imperative that you navigate across the lake and retrieve the disc.
However, the ice is slippery, so you won't always move in the direction you intend.

The surface is described using a grid like the following:

SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
The episode ends when you reach the goal or fall in a hole. You receive a reward of 1 if
you reach the goal, and zero otherwise.

'''

import gym
import numpy as np
import random
from collections import namedtuple
import collections
import matplotlib.pyplot as plt

# Parameters:
# episodes - a number of games we want the agent to play.
# gamma - aka decay or discount rate, to calculate the future discounted reward.
# epsilon - aka exploration rate, this is the rate in which an agent randomly decides its action rather than prediction.
# epsilon_decay - we want to decrease the number of explorations as it gets good at playing games.
# epsilon_min - we want the agent to explore at least this amount.

episodes = 100
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.9993
learning_rate = 0.8
max_games = 15001
'''
selecting the action using eps-greedy policy
'''
def eps_greedy_action(table, obs, n_actions):
	value, action = best_action_value(table, obs)
	if random.random() < epsilon:
		return random.randint(0, n_actions-1)
	else:
		return action

'''
selecting the action using a greedy policy
'''
def greedy_action(table, obs, n_actions):
	value, action = best_action_value(table, obs)
	return action

'''
Exploring the table and taking the action that best minimizes Q(s,a)
'''
def best_action_value(table, state):
	best_action = 0
	max_value = 0
	for action in range(n_actions):
		if table[(state, action)] > max_value:
			best_action = action
			max_value = table[(state, action)]
	return max_value, best_action

def Q_learning(table, obs0, obs1, reward, action):
	'''
	Update Q(obs0, action) according to Q(obs1, *) and the reward just obtained
	'''
	best_value, _ = best_action_value(table, obs1) # best value reachable from state obs1

	Q_target = reward + gamma * best_value # Q_target value

	Q_error = Q_target - table[(obs0, action)] #calculate the Q-error between target and previous value

	table[(obs0, action)] += learning_rate * Q_error

'''
Testing the policy
'''
def test_game(env, table, n_actions):
	reward_games = []
	for _ in range(episodes):
		obs = env.reset()
		rewards = 0
		while True:
			next_obs, reward, done, _ = env.step(greedy_action(table, obs, n_actions))
			obs = next_obs
			rewards += reward

			if done:
				reward_games.append(rewards)
				break
	return np.mean(reward_games)

'''
Creating the gym environment
'''
env = gym.make("FrozenLake-v0")
obs = env.reset()

obs_length = env.observation_space.n
n_actions = env.action_space.n

reward_count = 0
games_count = 0

table = collections.defaultdict(float)

test_rewards_list = []

while games_count < max_games:
	action = eps_greedy_action(table, obs, n_actions)
	next_obs, reward, done, _ = env.step(action)

	Q_learning(table, obs, next_obs, reward, action)

	reward_count += reward
	obs = next_obs
	env.render()
	if done:
		epsilon *= epsilon_decay

		if games_count %1000 == 0:
			test_reward = test_game(env, table, n_actions)
			print("\t Episode: ", games_count, "Test Reward: ", test_reward, np.round(epsilon, 2))

			test_rewards_list.append(test_reward)
		obs = env.reset()
		reward_count = 0
		games_count += 1

#Plotting the graphs
plt.figure(figsize = (20,10))
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.plot(test_rewards_list)
plt.show()














