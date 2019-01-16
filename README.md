# Reinforcement Learning Problems: 
This repository shares my learning journey in the field of Reinforcement Learning.

**Reinforcement Learning** is a type of machine learning which allows an AI agent to learn from it's surrounding (or environment) by interacting with it. It is kind of a trail and error learning. Each successful action leads to a positive reward, or a negative reward in another case. The agent goes through a series of actions and gains an overall reward for the whole duration of interacting with the environment.

The **OpenAI gym** provides many standard environments for people to test their Reinforcement Learning algorithms.

**All reinforcement learning algorithms tend to maximize the reward over its whole time of learning. But how an agent chooses an action to maximize the reward varies differently and there are several approaches.**

We need to successfully balance the  **exploration** and
**exploitation** tradeoff. Ideally, an approach should encourage exploration until the point it has learned enough about it to make informed decisions in the environment by taking optimal actions.

In order for an agent to learn how to deal optimally with all possible states in an environment, it must be exposed to as many of those states as possible. This can be thought of as exploring all the possible states in any environment. Once an agent has explored enough, it can exploit its experience to maximize the final reward.

Hence, an agent, in this case, explores and then exploits.
But **this is not the case always.**

Some of the common methods of exploration are listed out below along with their implementation. (Thanks to [Arthur](https://medium.com/@awjuliani))


#### 1. Greedy Approach:
It is a naive method which simply chooses that optimal action which leads to greatest reward from the agent.

Taking the action which the agent estimates at the current moment to be the best is an example of exploitation: the agent is exploiting it's current knowledge about the reward structure of the environment to act.

**Implementation**:
```
# Q_out is the acivation from the final layer of Q-network.
Q_value = sess.run(Q_out, feed_dict={inputs:[state]})
action = np.argmax(Q_value)
```

![Greedy](/assets/greedy_exp.png)

In the above figure, each value corresponds to the Q-value for a given action at a random state in an environment. The height of the light blue bar corresponds to the probability of choosing a given action. The dark blue bar corresponds to a chosen action.


#### 2. Random Approach:
The opposite to the greedy approach is to always take a random action.

**Implementation**:
```
# assuming OpenAI gym environment
action = env.action_space.sample()

action = np.random.randint(0, total_actions)
```

![Greedy](/assets/random_exp.png)

In the above figure, each value corresponds to the Q-value for a given action at a random state in an environment. The height of the light blue bar corresponds to the probability of choosing a given action. The dark blue bar corresponds to a chosen action.

#### 3. ϵ-greedy Approach:
A simple combination of the greedy and random approaches yields one of the most used exploration strategies: ϵ-greedy. In this approach the agent chooses what it believes to be the optimal action most of the time, but occasionally acts randomly. This way the agent takes actions which it may not estimate to be ideal, but may provide new information to the agent. The ϵ in ϵ-greedy is an adjustable parameter which determines the probability of taking a random, rather than principled, action. Due to its simplicity and surprising power, this approach has become the defacto technique for most recent reinforcement learning algorithms, including DQN and its variants.


**Implementation**:
```
e = 0.1
if np.random.randint(1)< e:
    action = env.action_space.sample()
else:
    Q_dist = sess.run(Q_out,feed_dict={inputs:[state]})
    action = np.argmax(Q_dist)
```

![Greedy](/assets/e-greedy.png)

In the above figure, each value corresponds to the Q-value for a given action at a random state in an environment. The height of the light blue bar corresponds to the probability of choosing a given action. The dark blue bar corresponds to a chosen action.


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



### 2. Frozen Lake:
#### Description:

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
you reach the goal and zero otherwise.

**Hence, we need an algorithm which learns long term expected rewards.**

The **Q-Learning algorithm**, in its simplest implementation, is a table of values for every state(row) and action(column) in an environment. Within each cell of the table, we learn how good it is for an agent to take a given action within a given state. In FrozenLake environment, we have 16 possible states(one for each block), and 4 possible actions(the four directions of movement), giving us a 16x4 table of values.

We start by initializing the table uniformly with all zeros, and then as we observe the rewards we obtain for various actions, we update the table accordingly.
We make updates to our Q-table using the **Bellman equation**, which states that **the expected long-term reward for a given action is equal to the immediate reward from the current action combined with the expected reward from the best future action taken at the following state**.

![FrozenLake](/assets/frozenlake.png)

#### Parameters:
- episodes - the number of games we want the agent to play.
- gamma - aka decay or discount rate, to calculate the future discounted reward.
- epsilon - aka exploration rate, this is the rate in which an agent randomly decides its action rather than prediction.
- epsilon_decay - we want to decrease the number of explorations as it gets good at playing games.
- epsilon_min - we want the agent to explore at least this amount.

