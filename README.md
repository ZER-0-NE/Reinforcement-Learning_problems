# Reinforcement Learning Problems: 
This repository shares my learning journey in the field of Reinforcement Learning.

**Reinforcement Learning** is a type of machine learning which allows an AI agent to learn from it's surrounding (or environment) by interacting with it. It is kind of a trail and error learning. Each successful action leads to a positive reward, or a negative reward in another case. The agent goes through a series of actions and gains an overall reward for the whole duration of interacting with the environment.

The **OpenAI gym** provides many standard environments for people to test their Reinforcement Learning algorithms.

**All reinforcement learning algorithms tend to maximize the reward over its whole time of learning. But how an agent chooses an action to maximize the reward varies differently and there are several approaches.**

We need to successfully balance the  **exploration** and
**exploitation** tradeoff. 

In order for an agent to learn how to deal optimally with all possible states in an environment, it must be exposed to as many of those states as possible. This can be thought of as exploring all the possible states in any environment. Once an agent has explored enough, it can exploit its experience to maximize the final reward.

Hence, an agent, in this case, explores and then exploits.
But **this is not the case always.**


### 1. CartPole
#### Description:
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. 
The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, 
and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that 
the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or 
the cart moves more than 2.4 units from the center.

In simple terms, we will code out an algorithm to teach our machine how to balance a pole at the top of a cart.

**Source : https://gym.openai.com/envs/CartPole-v1/**

It's very easy to interact with any environment in the OpenAI gym and is as simple as defining:

```
env = gym.make('Cartpole-v1')
```

You can then see the environment using ``` env.render() ```.

Make sure to have gym installed in your system. (``` pip install gym ```)


#### Hyperparameters:

- episodes - the number of games we want the agent to play.
- gamma - aka decay or discount rate, to calculate the future discounted reward.
- epsilon - aka exploration rate, this is the rate in which an agent randomly decides its action rather than prediction.
- epsilon_decay - we want to decrease the number of explorations as it gets good at playing games.
- epsilon_min - we want the agent to explore at least this amount.
- learning_rate - Determines how much a neural net learns in each iteration.


Below is how the training process looks like:

![cartpole](assets/cartpole1.gif)

