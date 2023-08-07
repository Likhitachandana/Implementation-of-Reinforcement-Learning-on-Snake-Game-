"""
SNAKE GAME
"""

#Implementation of Reinforcement learning on a Snake Game 

"""
INSTALLATION PROCESS 

"""

![pytorch](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png)
install pytorch from here: https://pytorch.org/
"""
Requirements mentioned in the text file are needed to be installed for the project 
"""
#command for installing the libraries

pip install -r requirements.txt


""" 
Code for to run the Game
"""

python main.py


## Configurations
All static settings are in settings.py
```python
import random

SIZE = 25 #Snake_size 


CLOSE_RANGE = (0, 2)
FAR_RANGE = (CLOSE_RANGE[1], 9)

set_size  = lambda x: SIZE * x

DEFAULT_WINDOW_SIZES = (42, 34)


WINDOW_N_X = 12
WINDOW_N_Y = 12


SCREEN_WIDTH = set_size(WINDOW_N_X) or set_size(DEFAULT_WINDOW_SIZES[0])
SCREEN_HEIGHT = set_size(WINDOW_N_Y) or set_size(DEFAULT_WINDOW_SIZES[1])

DEFAULT_KILL_FRAME = 100
DEFAULT_SPEED = 50 
DEFAULT_N_FOOD = 1
DECREASE_FOOD_CHANCE = 0.8


HIDDEN_SIZE = 256
OUTPUT_SIZE = 3

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

GAMMA = 0.9

EPSILON = 80

EPS_RANGE = (0, 200)
is_random_move = lambda eps, eps_range: random.randint(eps_range[0], eps_range[1]) < eps


DEFAULT_END_GAME_POINT = 300
```

#Add the windows 

Go to main.py and add more windows and processors
```python

class Windows(Enum):

    
    W14 = (20, 20, 1, 1)
```

#TO RUN THE WINDOW

The following will run window 14
```python

if __name__ == "__main__":
    Game(14)
```

#ADDING PARAMETERS TO THE PROCESSESS
"""
In par_lev.json:
Add parameters for instance "world 1" where world is (20, 20, 1, 3) the following: 
first processor will have all these parameters, second the epsilon changes to 80 and graph name different and third parameters will use the 
default, look at the report for more info on default values and settings.py.
json
"""
{
"W1":[
    {
      "empty_cell":0,
      "very_far_range":0,
      "close_range":0,
      "far_range":0,
      "col_wall":-10,
      "loop":-10,
      "scored":10,
      "gamma":0.5,
      "eps":200,
      "eps_range":[0, 200],
      "hidden_size":256,
      "n_food":1,
      "decrease_food_chance":-1,
      "kill_frame":100,
      "num_games":-1,
      "is_dir":true,
      "moving_closer":10,
      "moving_away":-10,
      "lr":0.0005,
      "graph":"epsilon__200__0"
    },
    {
      "eps":80,
      "graph":"epsilon__200__1"
    },
    {
    }    
  ]
 }
```

#Reinforcement Learning implementation 

Reinforcement learning (RL) is an area of machine learning where an agent aims to make the optimal decision in an uncertain environment in order to get the maximum cumulative reward. Since RL requires an agent to operate and learn in its environment, it's often easier to implement agents inside  simulations on computers than in real world situations. 

The main prupose of our project is to create We use the reinforcement learning mechanism on an existing video game (snake game). 
Here, The goal is to create a Reinforcement Learning agent to make an optimal decision in the closed environment to get the highest reward.
In this environment, it has incomplete information, and the state space of agents are quite large, We need to deal with these hurdles.



#HURDLES 
In this environment, it has incomplete information, and the state space of agents are quite large, We need to deal with these hurdles



## Environment and State Space

Representation for the snake game is in the form of n*n matrix. Each cell has a width and height of l pixels. A simplest approach is to use the pixel image to feel the agent. From the dimensions mentioned above, the state space would be of size |S| ∈ (n x n x l x l). While this method can work only for the smaller images but not for the larger images.As the size keep on increasing, we couldn't able to feed our agent properly. The size of the state space would reduce to |S| ∈ (n x n). This way of representing state is still not ideal as the state size increases exponentially as n grows. We will explore state reduction techniques in the next section.

## Action Space


Due to the simplistic nature of Snake there are only four possible actions that can be taken: up, down, left, and right. To speed up training and reduce backwards collisions, we simplified the actions down to three: straight, clockwise turn, and counter-clockwise turn. Representing the actions in this way is beneficial because when the agent 'explores' and randomly picks an action, it will not turn 180 degrees into itself. 

## Positive/Negative Rewards


The main reward of the game is when the snake eats food and increases its score. Therefore the reward is directly linked to the final score of the game, similar to how a human would judge its reward. As we will discuss later, we experimented with other positive rewards but ran into a lot of unexpected behaviour. With other positive rewards, the agent may loop infinitely or learn to avoid food altogether to minimize its length. We included additional negative rewards to give the snake more information about its state: collision detection , loop , empty cells, and close/mid/far/very_far from food.


# Methods and Models
---

A common RL algorithm that is used is Q-Learning which has been expanded to include neural networks with Deep Q-Learning methods. We decided that we could experiment with this new method that is gaining popularity and is used in previous research done with Atari games .

To begin our tests we first used PyGame to create our own Snake game with the basic rules of movement. The snake's actions  are simply to move forward, left, or right based on the direction its facing. The game ends if the snake hits itself or the wall. As it consumes food it grows larger. The goal is to get the snake to eat as much food as possible without ending the game.

After the game was created we created a Deep Q-Learning network using PyTorch. We created a network with an input layer of size 11 which defines the current state of the snake, one hidden layer of 256 nodes, and an output layer of size 3 to determine which action to take. Pictured below is a visual representation of our network.



Due to the discrete time steps (frames) of the game, we are able to calculate a new state for every new frame of the game. We defined our state parameters to be 11 boolean values based on the direction the snakes moving, the location of danger which we define as a collision that would occur with an action in the next frame, and the location of food relative to the snake. Our 3 actions (the output layer) are the directions of movement for the snake to move relative to the direction it's facing: forward, left, or right. 

The state at every time step is passed to the Q-Learning network and the network makes a prediction on what it thinks is the best action. This information is then saved in both short term and long term memory. All the information learned from the previous states can be taken from memory and passed to the network to continue the training process. 


# Experiments

We will look into the experiments and state how well the deep-q learning parameters are performing well and also first our agent is procceding with the untrained policy and later on we will be implementing a policy and training our agent with the parameters we have taken.


To reduce the randomness in our experiments. At the beginning of the experiment we will be taking 3 agents and computed the average of the 3 agents. Due to the slow processing of the average computation. we will be taking the trained set of data over 300 games
We will be presenting our results with a double plot. The top plot will be the 5 game moving average of the score and the bottom plot will be the highest score the agent achieved.

## No Training (Base Case)

The untrained agent moved around sporadically and without purpose. The highest score it achieved was only one food. As expected, there was no improvement in performance.

## Default Parameters
---

![Default](graphs/default.png)

We decided to set our default parameters as the following, and made changes to individual parameters to see how they would change the performance: 

```python
  Gamma = 0.9
  Epsilon = 0.4
  Food_amount = 1
  Learning_Rate = 0.001
  Reward_for_food (score) = 10
  collision_with_Wall = -10
  collision_with_itself = -10
  Snake_going_in_a_loop = -10
```
 
#GAMMA_VALUE_EXPERIMENT 
---

![Gamma](graphs/gamma.png)

We decided to test gamma values of 0.00, 0.50, and 0.99. A gamma of 0.00 means that the agent is only focused on maximizing immediate rewards. We assumed a gamma of 0 would be ideal because the game of Snake is not a game where you have to think very far in the future. The results show that the best performance was with a gamma of 0.50 which showed much better performance than the other two. We are unsure why a gamma of 0.99 performed so badly. Our default value of gamma=0.90 performed the best. This demonstrates that it is necessary to fine tune gamma to balance the priority of short term reward vs long term.

#EPISLON_VALUE_EXPERIMENT 


Our epsilon decays by 0.005 every game until it reaches 0. This allows our agent to benefit from exploratory learning for a sufficient amount of time before switching to exploitation.
 
We wanted to test how changing the balance between exploration and exploitation impacts the performance. We decided to try no exploration (epsilon = 0), a balance of both (epsilon = 0.5), and the largest amount of exploration at the beginning (epsilon = 1).

As seen by the graph above, an epsilon of 0 performs poorly because without exploration the agent cannot learn the environment well enough to find the optimal policy. An epsilon of 0.5 provides an even balance of exploring and exploitation which greatly improves the learning of the agent. An epsilon of 1 maximizes the amount of time the agent explores in the beginning. This results in a slow rate of learning at the beginning but a large increase of score once the period of exploitation begins. This behaviour proves that high epsilon values are needed to get a higher score. To conclude, epsilon values of 0.5 and 1 both seem to be more performant than the default of 0.4.

#REWARDS


In this experiment we decided to change the immediate rewards. Our immediate rewards were (S) score for eating food, (C) collision with the wall or itself, and (L) moving in a loop. An interesting result we came across is that having a large difference between positive and negative rewards negatively affects the performance of the agent. This may be because the agent will learn to focus on the larger of the negative or positive rewards therefore making rewards of equal magnitude be more performant. We also found that having rewards that are small in scale do better than rewards that are large in scale. The best performance we found was with rewards of C=-5, L=-5, and S=5. Rewards of C=-1, L=-1, and S=1 performed very similarly to the previous agent. Larger rewards of 30 and -30 performed much worse. The performance of rewards C=-5, L=-5, and S=5 did slightly better than default.
 
#Learning Rate 


Changing the learning rate impacts the way our agent finds an optimal policy and how fast it learns. We found that a learning rate of 0.05 was too high since it performed similar to our untrained agent. This poor performance is likely because the agent was skipping over the optimal by taking too large a step size. This means that the agent moves too quickly from one suboptimal solution to another and fails to learn. We noticed a strong correlation between lower learning rates and higher scores. Our best performing agents had the lowest learning rates with lr = 0.0005 performing better than our default of lr = 0.001.

#OPTIMALAGENT

The performance of our optimal agent is slightly better than the default. This is because our default parameters were similar to the optimal parameters from our experiments. Further experimentation would allow for more finetuning of parameters to increase performance.
	From our experiments we found that the learning rate, epsilon, gamma, and immediate rewards were the parameters that had the biggest impact on performance. The experiments with direction, distance, and food generation were detrimental to performance and are not parameters that would help with the optimal performance.


Based on our experiments we decided to take the optimal values we found to see how the agent would perform over a 1000 games compared to the default. The optimal parameters we used were:
```python
Gamma = 0.9
Epsilon = 1
Food amount = 1
Learning Rate = 0.0005
Reward for food (score) = 5
Collision to wall = -5
Collision to self = -5
Snake going in a loop = -5
```

The performance of our optimal agent is slightly better than the default. This is because our default parameters were similar to the optimal parameters from our experiments. Further experimentation would allow for more finetuning of parameters to increase performance.

From our experiments we found that the learning rate, epsilon, gamma, and immediate rewards were the parameters that had the biggest impact on performance. The experiments with direction, distance, and food generation were detrimental to performance and are not parameters that would help with the optimal performance.


Above is a graph showing the high scores for each experiment for each parameter.
	
We combined the parameters that had the best impact on performance and used them as part of our optimal parameters. We found that small changes in the learning rate had the largest difference in performance. From the lowest to its highest result, the difference was a record of 79. The rewards had the second largest range, then epsilon, and then gamma. 

Our experiments were a starting point of looking at parameters that would impact Deep Q-Learning. Further research could be done to tune the main parameters to optimize the model further. 


#REFERENCE

1. https://arxiv.org/pdf/2007.08794.pdf 
2. https://www.researchgate.net/publication/221455879_High-level_reinforcement_learning_in_strategy_games
3. https://arxiv.org/abs/1312.5602 
4. https://papers.nips.cc/paper/2017/file/39ae2ed11b14a4ccb41d35e9d1ba5 d11-Paper.pdf
5. https://medium.com/@hugo.sjoberg88/using-reinforcement-learning-and-q-learning-to-play-snake-28423dd49e9b
6. https://towardsdatascience.com/learning-to-play-snake-at-1-million-fps-4aae8d36d2f1
7. http://cs229.stanford.edu/proj2016spr/report/060.pdf
8. https://mkhoshpa.github.io/RLSnake/
9. https://docs.python.org/3/library/multiprocessing.html
10. https://github.com/eidenyoshida/Snake-Reinforcement-Learning
11. https://github.com/python-engineer/snake-ai-pytorch/blob/main/model.py


