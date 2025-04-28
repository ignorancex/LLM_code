
# external imports
import unittest
import random

# internal imports
from hytoperm import *


class TestAgent(unittest.TestCase):
    
    def testLocalController(self):
        n_sets = 20
        ex = Experiment.generate(n_sets=n_sets)
        assert(isinstance(ex, Experiment))
        
        target = ex._world.targets()[0]
        sensor = ex.agent().sensor()

        phi = SwitchingPoint(target.region().randomBoundaryPoint())
        psi = SwitchingPoint(target.region().randomBoundaryPoint())
        tf = 10
        Omega0 = {}
        for t in ex._world.targets():
            Omega0[t] = np.eye(1)
        lmp = SwitchingParameters(phi=phi,psi=psi,tf=tf,Omega0=Omega0)
        mc = MonitoringController(target, sensor)

        mc.buildOptimalMonitoringSolver(target, sensor)
        tp, tmse, tomega, tu, cost, lam = mc.optimalMonitoringControl(lmp)

        ex, ax = ex.plotWorld(with_sensor_quality=True)
        tp.plotStateVsState(0,1, ax)

    def testCycle(self):
        n_sets = 8
        ex = Experiment.generate(n_sets=n_sets)
        assert(isinstance(ex, Experiment))
        ex.agent().computeVisitingSequence()
        ex.agent().initializeCycle()
        ex.agent()._cycle.simulate()
        ex.agent().plotCycle()

    def testBilevelOptimization(self, n_sets=10):
        ex = Experiment.generate(n_sets=n_sets)
        assert(isinstance(ex, Experiment))
        ex.agent().computeVisitingSequence()
        ex.agent().op.alpha = 0.1
        ex.agent().op.beta = 0.9
        ex.agent().op.optimization_iters = 10
        ex.agent().optimizeCycle()

    def testMultiAgentCycles(self):
        n_sets = 8
        ex = Experiment.generate(n_sets=n_sets, n_agents=3)
        assert(isinstance(ex, Experiment))
        ex.agent(0).computeVisitingSequence()
        ex.agent(1).setTargetVisitingSequence(
            [ex.world().target(i) for i in range(ex.world().nTargets())]
            )
        ex.agent(2).setTargetVisitingSequence(
            random.shuffle(
                [ex.world().target(i) for i in range(ex.world().nTargets())]
                )
            )
        
        for agent in ex.agents():
            agent.computeVisitingSequence()
            agent.initializeCycle()
            agent._cycle.simulate()

    def testTrajectoryPointsProgrammatically(self):
        n_sets = 8
        ex = Experiment.generate(n_sets=n_sets)
        assert(isinstance(ex, Experiment))
        ex.agent().computeVisitingSequence()
        ex.agent().initializeCycle()
        ex.agent()._cycle.simulate()

        for i in range(1,len(ex.agent()._switchingSegments)):
            ## Check if switching segment starts near last monitoring segment
            ms = ex.agent()._monitoringSegments[i-1]
            ss = ex.agent()._switchingSegments[i]
            ms_end = ms.pTrajectory.x[:,-1]
            ss_start = ss.pTrajectory.x[:,0]
            dist = np.linalg.norm(ms_end-ss_start)
            self.assertLessEqual(dist, 1e-5, msg=f"Distance between ms {i-1} end and ss {i} start: {dist}")

            ## Check if switching segment ends near next monitoring segment
            ms_next = ex.agent()._monitoringSegments[i]
            ms_start = ms_next.pTrajectory.x[:,0]
            ss_end = ss.pTrajectory.x[:,-1]
            dist = np.linalg.norm(ms_start-ss_end)
            self.assertLessEqual(dist, 1e-5, msg=f"Distance between ms {i-1} end and ss {i} start: {dist}")

if __name__ == "__main__":
    unittest.main()
