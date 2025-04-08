import sys
import numpy as np
import random
import gym
from time import sleep

from SpiderEnv import SpiderEnv
from ManageCONST import readCONST, writeTargetStressLevel
import Logger

def simulate(q_table, env):
    """Run the Q-learning simulation."""
    CONST = readCONST()
    MAX_EPISODES = CONST["QLearning"]["MAX_EPISODES"]
    MAX_TRY = CONST["QLearning"]["MAX_TRY"]
    learning_rate = CONST["QLearning"]["learning_rate"]
    gamma = CONST["QLearning"]["gamma"]
    epsilon = CONST["QLearning"]["epsilon"]
    epsilon_decay = CONST["QLearning"]["epsilon_decay"]

    print("Starting simulation...")
    targetStress = CONST["ListTargetStress"][0]
    writeTargetStressLevel(targetStress)

    # Initialize the environment
    state = env.reset()
    total_reward = 0

    for episode in range(MAX_EPISODES):
        targetStress = CONST["ListTargetStress"][episode]
        writeTargetStressLevel(targetStress)
        print(f"Episode {episode}")

        # Set tries for the first episode
        env.setTry(1 if episode == 0 else 0)

        # Wait to gather data for signal length during the first episode
        if episode == 0:
            sleep(CONST["SignalLengthEDA"] - CONST["waitToUpdateSpider"])

        while env.getTry() < MAX_TRY:
            print(f"Attempt {env.getTry()}")
            state = tuple(state)

            # Choose action based on epsilon-greedy strategy
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            # Take action and observe result
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Update Q-value using the Q-learning formula
            q_value = q_table[state][action]
            best_q = np.max(q_table[next_state])
            q_table[state][action] = (1 - learning_rate) * q_value + learning_rate * (reward + gamma * best_q)

            # Update state for the next iteration
            state = next_state

        # Decay the exploration rate
        if epsilon >= 0.005:
            epsilon *= epsilon_decay

    print("Simulation finished!")
    return done

def QLearning():
    """Initialize the Spider environment and run the Q-learning algorithm."""
    env = SpiderEnv()

    # Create a Q-table with dimensions based on the observation and action space
    num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    q_table = np.zeros(num_box + (env.action_space.n,))

    done = False
    env.States = {}

    done = simulate(q_table, env)
    print("Simulation complete:", done)
