# This script makes use of OpenAI gym to train on the cartpole game.
# Description of Game:

# A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. 
# The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, 
# and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that 
# the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or 
# the cart moves more than 2.4 units from the center.
# source : https://gym.openai.com/envs/CartPole-v1/


# Parameters:

# episodes - a number of games we want the agent to play.
# gamma - aka decay or discount rate, to calculate the future discounted reward.
# epsilon - aka exploration rate, this is the rate in which an agent randomly decides its action rather than prediction.
# epsilon_decay - we want to decrease the number of explorations as it gets good at playing games.
# epsilon_min - we want the agent to explore at least this amount.
# learning_rate - Determines how much neural net learns in each iteration.


#!pip install gym
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

### Inspired from the post of keon.io/deep-q-learning/

episodes = 1000

'''
By defining memory, we make sure that the state,action.reward and next_state
are remembered, as the neural network in DQN tends to forget them after each 
iteration.
'''

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        '''
        Neural Network for DQN
        '''
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        '''
        Keep appending the memory
        '''
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        '''
        The agent will select at first it's action at random because 
        it is better for the agent to try all kinds of things before 
        it starts to see the patterns. 
        '''
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # argmax picks the highest value among
        # the two values in act_values eg [0.67, 0.04]

    def replay(self, batch_size):
        '''
        Trains the neural net with experience in the memory.
        We need to maximise the rewards in the long run, so we define gamma/discount
        rate through which the agent will learn to maximise the discounted future 
        award in the long run.
        '''
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("Episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, episodes, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        