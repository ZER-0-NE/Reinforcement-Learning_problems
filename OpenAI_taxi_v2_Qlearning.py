import tensorflow as tf
import numpy as np
import random
import gym

env = gym.make("Taxi-v2")
env.render()

#creating and initialising q-table
action_size = env.action_space.n #6
state_size = env.observation_space.n #500
qtable = np.zeros((state_size, action_size))
#print(qtable)

total_episodes = 1000
total_test_episodes = 100
max_steps = 99

learning_rate = 0.7
gamma = 0.618

#exploration parameters
epsilon = 1.0 
max_epsilon = 1.0 #exploration probability at start
min_epsilon = 0.01 # minimum exploration probability
decay_rate = 0.01 #exponential decay rate for exploration
# as training progress we need less exploration and more and more exploitation

for episode in range(total_episodes):
	state = env.reset()
	done = False
	step = 0

	for step in range(max_steps):
		exp_tradeoff = random.uniform(0,1) #random action intially

		# this leads to exploitation => takin biggest Q value of this state
		if exp_tradeoff > max_epsilon:
			action = np.argmax(qtable[state,:])
		# doing a random choice => exploration
		else:
			action = env.action_space.sample()

		#take the action and observe the outcome state and reward
		new_state, reward, done, info = env.step(action)

		# Update using bellman equation
		# Q(s, a) := Q(s,a) + lr [R(s,a) + gamma * max Q(s', a') - Q(s, a)]
		qtable[state, action] = qtable[state, action] + learning_rate * (
			reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

		state = new_state

		if done: # if finished episode
			break

	episode += 1

	#reduce epsilon because we need less and less exploration
	epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)

env.reset()
rewards = []

for episode in range(total_test_episodes):
	state = env.reset()
	step = 0
	done = False
	total_rewards = 0
	print("EPISODE", episode)

	for step in range(max_steps):
		env.render()
		# take the action that has the maximum expected future reward given that state
		action = np.argmax(qtable[state, :])
		new_state, reward, done, info = env.step(action)

		total_rewards += reward

		if done:
			rewards.append(total_rewards)
			print("Score", total_rewards)
			break
		state = new_state
env.close()
print("Score over time: " + str(sum(rewards)/total_test_episodes))







		 






