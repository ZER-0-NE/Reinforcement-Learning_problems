import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Flatten, Dense
from collections import deque
import random

env = gym.make('MountainCar-v0')

model = Sequential()
model.add(Dense(20, input_shape = (2,) + env.observation_space.shape, 
						init = 'uniform', activation = 'relu'))
model.add(Flatten())
model.add(Dense(18, init = 'uniform', activation = 'relu'))
model.add(Dense(10, init = 'uniform', activation = 'relu'))
model.add(Dense(env.action_space.n, init = 'uniform', activation = 'linear'))

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])

D = deque()

observetime = 500
epsilon = 0.7 #random move probability
gamma = 0.9 #discounted future reward
mb_size = 50

# Observing
observation = env.reset()
obs = np.expand_dims(observation, axis = 0)
state = np.stack((obs, obs), axis = 1)
done = False

for i in range(observetime):
	

