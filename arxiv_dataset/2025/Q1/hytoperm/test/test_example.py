
# external imports
import unittest
import matplotlib.pyplot as plt

# internal imports
from hytoperm import *


class TestExample(unittest.TestCase):

    def testExample(self):
        ex = Experiment.generate()
        fig, ax = ex.plotWorld()
        ex.agent().plotSensorQuality()
        ex.agent().computeVisitingSequence()
        op = OptimizationParameters()
        op.optimization_iters = 3
        ex.agent().op = op
        ex.agent().optimizeCycle()
        ex.agent().plotCycle()


if __name__ == "__main__":
    unittest.main()
