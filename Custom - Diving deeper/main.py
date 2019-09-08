import gym
import numpy as np

env = gym.make("MountainCar-v0")

#find how many actions/moves are possible
'''
For the various environments, we can query them for how many actions/moves are possible. In this case, there are 
"3" actions we can pass. This means, when we step the environment, we can pass a 0, 1, or 2 as our "action" for 
each step. Each time we do this, the environment will return to us the new state, a reward, whether or not the 
environment is done/complete, and then any extra info that some envs might have.

It doesnt matter to our model, but, for our understanding, a 0 means push left, 1 is stay still, and 2 means 
push right. 
'''
# print(env.action_space.n)

env.reset()

done = False

while not done:
	action = 2 # push right always since we want to reach the yellow flag
	new_state, reward, done, _ = env.step(action)
	# print(reward, new_state)
	'''
	-1.0 [-0.33386505 -0.00114066]
	-1.0 [-0.3353531  -0.00148806]
	-1.0 [-0.33717915 -0.00182604]
	-1.0 [-0.33933158 -0.00215244]

	Initially, the reward is -1 for every action. And the values are position and velocity. These 2 values are the 
	observation_space.
	'''

	# print(env.observation_space.high) #[0.6  0.07]
	# print(env.observation_space.low) #[-1.2  -0.07]
	
	env.render()
	'''
	As you can see, despite asking this car to go right constantly, we can see that it just doesn't quite have 
	the power to make it. Instead, we need to actually build momentum here to reach that flag. To do that, we'd 
	want to move back and forth to build up momentum.
	'''