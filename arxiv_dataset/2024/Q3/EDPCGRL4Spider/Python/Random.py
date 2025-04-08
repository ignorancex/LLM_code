import sys
import numpy as np
import random
import gym
from SpiderEnv import SpiderEnv
from ManageCONST import readCONST
import Logger

def simulate(env):
    """Run the simulation using random actions."""
    CONST = readCONST()
    MAX_EPISODES = CONST["QLearning"]["MAX_EPISODES"]
    MAX_TRY = CONST["QLearning"]["MAX_TRY"]

    for episode in range(MAX_EPISODES):
        # Initialize environment for each episode
        state = env.reset()
        total_reward = 0

        # The agent tries up to MAX_TRY times
        for attempt in range(MAX_TRY):
            state = tuple(state)  # Convert state to tuple for indexing
            action = env.action_space.sample()  # Choose a random action
            
            # Perform action and receive the result
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Prepare for the next iteration
            state = next_state

            # If the episode is done or max attempts reached, break the loop
            if done or attempt >= MAX_TRY - 1:
                break

    return done

def Random():
    """Run the Random action selection algorithm in the Spider environment."""
    env = SpiderEnv()
    done = False

    # Initialize environment states
    obs = env.reset()
    env.States = {}
 
    done = simulate(env)
    print("Simulation completed:", done)
