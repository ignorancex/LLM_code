
# external imports
import unittest
import numpy as np

# internal imports
from hytoperm import *


class TestTrajectory(unittest.TestCase):
    
    def testShiftTime(self):
        traj = Trajectory(
            np.array([[-10,20,0.2,0.01,-1.0]]), 
            np.array([0,1,2,3,4])
            )
        traj.shiftTime(1)
        self.assertTrue(np.all(traj.t == np.array([1,2,3,4,5])))          


if __name__ == "__main__":
    unittest.main()
