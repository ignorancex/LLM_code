import numpy as np
from collections import deque
from scipy.stats import norm
from PredictionModelEDA import getStressLevel
from time import sleep
from ManageCONST import readCONST, writeSpiderAttr, writeTargetStressLevel
import Logger

class RuleBased:
    def __init__(self):
        # Load constants and reset the initial state
        self.CONST = readCONST()
        self.reset()

    def initialRandomState(self):
        """Initialize spider attributes to a random state based on configuration."""
        spiderAttributes = np.zeros(len(self.CONST['ATTR']))
        for i, max_val in enumerate(self.CONST['rangeATTR']):
            avg = int(max_val / 2)
            min_val = 0
            max_val -= 1
            if self.CONST['StartState'] == 'AVG':
                spiderAttributes[i] = avg
            elif self.CONST['StartState'] == 'MAX':
                spiderAttributes[i] = max_val
            elif self.CONST['StartState'] == 'MIN':
                spiderAttributes[i] = min_val
        return spiderAttributes.astype(np.int16)

    def getStress(self):
        """Retrieve and log the current stress level based on the observation."""
        print("Observation:", self.observation, type(self.observation))
        writeSpiderAttr(self.observation)
        Logger.log("Observation", self.observation)
        self.calculateStressLevel()
        print("*************Stress:", self.stressLevel)
        return self.stressLevel

    def calculateStressLevel(self):
        """Calculate the stress level after a delay to allow for signal processing."""
        sleep(self.CONST["waitToUpdateSpider"])
        stressLevel = getStressLevel()
        while stressLevel < 0:  # Ensure stress level is valid
            sleep(1)
            stressLevel = getStressLevel()
        self.stressLevel = stressLevel

    def reset(self):
        """Reset the environment to an initial random state."""
        self.observation = self.initialRandomState()
        writeSpiderAttr(self.observation)

    def calculateCorrection(self):
        """Calculate the correction factor based on the desired stress level."""
        current_stress = self.getStress()
        desired_stress = self.CONST["TargetSTRESS"]
        print("Target Stress:", desired_stress)
        self.corr = (current_stress - desired_stress) / 10  # Correction should be between [-1, +1]
        print("Correction:", self.corr)

    def rescale(self, x, xmin, xmax, ymin, ymax):
        """Rescale a value from one range to another."""
        y = (x - xmin) / (xmax - xmin)
        return y * (ymax - ymin) + ymin

    def getCorrRange(self, min_val=-1, max_val=1):
        """Get a range of correction values."""
        return np.linspace(min_val, max_val, 100)

    def getLargeness(self, corr):
        """Calculate the size based on the correction factor."""
        return 0.3 + 0.2 ** corr

    def getMovement(self, corr):
        """Calculate the movement velocity based on the correction factor."""
        return 0.006 - 0.003 * ((corr + 0.6) / 1.4)

    def getCloseness(self, corr):
        """Calculate the probability of moving toward the subject."""
        return (1 - ((corr + 0.6) / 1.4)) ** 2

    def getLocomotion(self, corr):
        """Calculate the locomotion jump formula."""
        return -0.03 + (0.364 / (corr + 4.6))

    def getHairiness(self, corr):
        """Calculate hairiness based on correction."""
        return -1 * corr

    def getColor(self, corr):
        """Calculate color based on correction factor."""
        return 0.3 + 0.2 ** corr

    def getFeatureScaled(self, getFeature, minFeature=0, maxFeature=2):
        """Scale a feature based on its correction."""
        value = getFeature(self.corr)
        corr_range = self.getCorrRange()
        value_min = min(getFeature(corr_range))
        value_max = max(getFeature(corr_range))
        return self.rescale(value, value_min, value_max, minFeature, maxFeature)

    def step(self):
        """Perform a single step in the rule-based logic, updating the observation."""
        self.CONST = readCONST()
        self.calculateCorrection()
        
        # Calculate features
        LocoMotion = self.getFeatureScaled(self.getLocomotion, 0, self.CONST["MaxATTR"][0])
        Motion = self.getFeatureScaled(self.getMovement, 0, self.CONST["MaxATTR"][1])
        Closeness = self.getFeatureScaled(self.getCloseness, 0, self.CONST["MaxATTR"][2])
        Large = self.getFeatureScaled(self.getLargeness, 0, self.CONST["MaxATTR"][3])
        Hairiness = self.getFeatureScaled(self.getHairiness, 0, self.CONST["MaxATTR"][4])
        Color = self.getFeatureScaled(self.getColor, 0, self.CONST["MaxATTR"][5])

        # Create a new observation
        new_observation = np.around([LocoMotion, Motion, Closeness, Large, Hairiness, Color]).astype(np.int16)
        print("New observation:", self.corr, new_observation)

        # Update the observation by changing one attribute randomly
        index_diff = [i for i in range(len(new_observation)) if self.observation[i] != new_observation[i]]
        if index_diff:
            random_index = np.random.choice(index_diff)
            self.observation[random_index] = new_observation[random_index]

        print("Updated observation:", self.corr, self.observation)
        self.observation = np.clip(self.observation, np.zeros(len(self.CONST['ATTR'])), np.array(self.CONST['MaxATTR']))

        return self.observation

def simulate(ruleBased):
    """Simulate the rule-based environment."""
    CONST = readCONST()
    MAX_EPISODES = CONST["RuleBased"]["MAX_EPISODES"]
    MAX_TRY = CONST["RuleBased"]["MAX_TRY"]

    targetStress = CONST["ListTargetStress"][0]
    writeTargetStressLevel(targetStress)

    # Init environment
    state = ruleBased.reset()

    for episode in range(MAX_EPISODES):
        targetStress = CONST["ListTargetStress"][episode]
        writeTargetStressLevel(targetStress)
        print("Episode", episode)

        # Wait more at the beginning to gather data for the signal length
        if episode == 0:
            sleep(CONST["SignalLengthEDA"] - CONST["waitToUpdateSpider"])

        # AI tries up to MAX_TRY times
        for t in range(MAX_TRY):
            print("Try:", t, episode)
            next_state = ruleBased.step()
            state = next_state

def Rule_Based():
    """Run the rule-based simulation."""
    ruleBased = RuleBased()
    simulate(ruleBased)
