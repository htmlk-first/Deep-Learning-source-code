import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.envs.registration import register
import random

'''
Q-Table
        action  |   L   |   D   |   R   |   U   |
-------------------------------------------------
state:  0       |       |       |       |       |
-------------------------------------------------
state:  1       |       |       |       |       |
-------------------------------------------------
state:  2       |       |       |       |       |
-------------------------------------------------
state:  ...     |       |       |       |       |
-------------------------------------------------
'''

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={
        'map_name': '4x4',
        'is_slippery': False
    }
)

env = gym.make("FrozenLake-v3")

# Initialization with 0 in Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n]) # (16,4) where 16: 4*4 map, 4: actions
num_episodes = 1000 # Number of iterations

rList = []
successRate = []

def rargmax(vector):
    m = np.amax(vector) # Return the maximum of a array or maximum along an axis (0 or 1)
    indices = np.nonzero(vector == m)[0] # np.nonzero(True/False vector) => find the maximum
    return random.choice(indices) # Random selection

for i in range(num_episodes): # Updates with num_episodes iterations
    state = env.reset() # Reset
    total_reward = 0 # Reward graph (1: success, 0: failure)
    done = None
    
    while not done: # The agent is not in the goal yet
        action = rargmax(Q[state, :]) # Find maximum reward among 4 actions, find next action
        new_state, reward, done, _ = env.step(action) # Result of the chosen action
        
        Q[state, action] = reward + np.max(Q[new_state, :]) # Q-update
        total_reward += reward
        state = new_state
    
    rList.append(total_reward) # Reward appending
    successRate.append(sum(rList)/(i+1)) # Success rate appending
    


print(Q)
print("successRate: ", successRate[-1])