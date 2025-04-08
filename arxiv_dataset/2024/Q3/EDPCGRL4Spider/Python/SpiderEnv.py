import gym
from gym import spaces
import numpy as np
from collections import deque
from scipy.stats import norm
from PredictionModelEDA import getStressLevel
from time import sleep
from ManageCONST import readCONST, writeSpiderAttr
import Logger

class SpiderEnv(gym.Env):
    """Custom environment for managing stress levels in a spider simulation."""

    def __init__(self):
        """Initialize the Spider environment and configure action and observation spaces."""
        super(SpiderEnv, self).__init__()
        self.CONST = readCONST()
        self.reward_range = (0, 1)
        self.States = {}
        self.TRY = 0
        
        # Define action space based on configuration
        if self.CONST['ActionSpace'] == 1:
            self.action_space = spaces.Box(
                low=np.array([-1] * len(self.CONST['ATTR'])),
                high=np.array([1] * len(self.CONST['ATTR'])),
                dtype=np.int16
            )
        elif self.CONST['ActionSpace'] == 2:
            self.action_space = spaces.Discrete(2 * len(self.CONST['ATTR']))

        self.observation_space = spaces.Box(
            low=np.zeros(len(self.CONST['ATTR'])),
            high=np.array(self.CONST['MaxATTR']),
            dtype=np.int16
        )
        self.observation = self.initialRandomState()

    def initialRandomState(self):
        """Generate the initial random state for spider attributes."""
        spiderAttributes = np.zeros(len(self.CONST['ATTR']))
        for i, maxVal in enumerate(self.CONST['rangeATTR']):
            if self.CONST['StartState'] == 'AVG':
                spiderAttributes[i] = maxVal / 2
            elif self.CONST['StartState'] == 'MAX':
                spiderAttributes[i] = maxVal - 1
            elif self.CONST['StartState'] == 'MIN':
                spiderAttributes[i] = 0
        return spiderAttributes.astype(np.int16)

    def addStates(self):
        """Add the current observation and stress level to the states."""
        obs = str(self.observation)
        self.States[obs] = self.stressLevel

    def getStress(self):
        """Retrieve or calculate the stress level based on the current observation."""
        self.CONST = readCONST()
        obs = str(self.observation)

        if obs in self.States:
            self.stressLevel = self.States[obs]
        else:
            print("Observation:", self.observation, type(self.observation))
            writeSpiderAttr(self.observation)
            Logger.log("Observation", self.observation)
            self.calculateStressLevel()
            self.addStates()
            Logger.log("Reward", self.calculateReward())

        print("************* Stress:", self.stressLevel)
        return self.stressLevel

    def calculateStressLevel(self):
        """Calculate the stress level after a delay."""
        self.addOneTry()
        sleep(self.CONST["waitToUpdateSpider"])
        stressLevel = getStressLevel()

        while stressLevel < 0:  # Wait until a valid stress level is received
            sleep(1)
            stressLevel = getStressLevel()

        self.stressLevel = stressLevel

    def hitMaxStress(self):
        """Check if the current stress level hits the maximum threshold."""
        return self.getStress() == self.CONST['MaxSTRESS']

    def hitGoalStress(self):
        """Check if the current stress level meets the target."""
        if self.getStress() == self.CONST['TargetSTRESS']:
            print("Find the Target!")
            self.resetDone()
            return True
        return False

    def getNormalDistribution(self):
        """Generate a normal distribution for the stress levels."""
        x_axis = np.arange(0, self.CONST['MaxSTRESS'] + 1, 1)
        mean = self.CONST['TargetSTRESS']
        print("Target stress:", mean)
        sd = self.CONST['MaxSTRESS'] / 2
        y = norm.pdf(x_axis, mean, sd)

        # Normalize the output range to [-1, 1]
        oldMin, oldMax = np.min(y), np.max(y)
        newMax, newMin = 1, -1
        result = (((y - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin
        return result

    def calculateReward(self):
        """Calculate the reward based on the current stress level."""
        stressLevel = self.getStress()
        rewards = self.getNormalDistribution()
        reward = rewards[stressLevel]
        return reward

    def step(self, action):
        """Execute one time step within the environment based on the action taken."""
        self.CONST = readCONST()
        if self.CONST['ActionSpace'] == 2:
            new_action = np.zeros(len(self.CONST['ATTR']))
            new_action[int(action / 2)] = 1
            if action % 2 == 1:
                new_action *= -1
            action = new_action.astype(np.int16)

        self.next_observation = np.add(action, self.observation)
        self.next_observation = np.clip(self.next_observation, 0, self.CONST['MaxATTR'])
        self.next_observation = self.next_observation.astype(np.int16)
        
        self.observation = self.next_observation
        self.reward = self.calculateReward()
        self.done = self.hitGoalStress()
        info = {}
        
        return self.observation, self.reward, self.done, info

    def reset(self):
        """Reset the environment to an initial state."""
        self.epoch = 0
        self.setTry(0)
        self.observation = self.initialRandomState()
        self.reward = self.calculateReward()
        self.done = False
        return self.observation

    def resetDone(self):
        """Set the done flag to indicate the episode has ended."""
        global DONE
        DONE = True
        self.done = True

    def getTry(self):
        """Retrieve the current try count."""
        return self.TRY

    def setTry(self, val):
        """Set the current try count."""
        self.TRY = val

    def addOneTry(self):
        """Increment the try count by one."""
        self.TRY += 1
