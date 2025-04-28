
# external imports
import unittest

# internal imports
from hytoperm import *


class TestAbstraction(unittest.TestCase):
    def testCompleteGraph(self):
        ex = Experiment.generate(n_sets=20)
        assert(isinstance(ex, Experiment))
        gpp = GlobalPathPlanner(ex._world)
        opts = AbstractionOptions()
        opts.onlyDirectConnections = False
        ga = GraphAbstraction(ex._world, gpp, opts)
        
        fig, ax = plt.subplots()
        ga.plotAbstraction(ax=ax)
        plt.close()

    def testIncompleteGraph(self):
        ex = Experiment.generate(n_sets=20)
        assert(isinstance(ex, Experiment))
        gpp = GlobalPathPlanner(ex._world)
        opts = AbstractionOptions()
        opts.onlyDirectConnections = True
        ga = GraphAbstraction(ex._world, gpp, opts)
        
        fig, ax = plt.subplots()
        ga.plotAbstraction(ax=ax)
        plt.close()


if __name__ == "__main__":
    unittest.main()
