
# external imports
import unittest

# internal imports
from hytoperm import *


class TestGlobalPlanner(unittest.TestCase):
    
    def testRRBT(self):
        niter = 100; n_sets = 20
        ex = Experiment.generate(n_sets=n_sets)
        assert(isinstance(ex, Experiment))
        gpp = GlobalPathPlanner(ex._world)
        gpp.rrbt_iter = niter
        fig, ax = ex.plotWorld()

        t0 = ex._world.target(1).p()
        tf = ex._world.target(9).p()
        path, time = gpp.planPath(t0, tf)

        self.assertTrue(isinstance(path, Tree))
        self.assertTrue(isinstance(time, float))
        self.assertTrue(np.allclose(t0,path.getData().p()))
        self.assertTrue(np.allclose(tf,path.getRoot().getData().p()))
        
        ex._world.plotTravelCostPerRegion(ax)
        
    def testTSP(self):
        n_sets=20; plot = False
        ex = Experiment.generate(n_sets=n_sets, fraction=0.2)
        assert(isinstance(ex, Experiment))
        gpp = GlobalPathPlanner(ex._world)
        gpp._plot_options.toggleAllPlotting(plot)
        fig, ax = ex.plotWorld()
        gpp.solveTSP()
        gpp.plotTSPSolution(ax, color='red', linewidth=2)
        po = ex.agent().plotSensorQuality(grid_size=0.05, ax=ax)
        gpp.plotTSPSolution(ax, color='red', linewidth=2)


if __name__ == "__main__":
    unittest.main()
