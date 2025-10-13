from blackbox import WingFFD
from baseclasses import AeroProblem
import numpy as np

solverOptions = {
    # Common Parameters
    "monitorvariables": ["cl", "cd", "yplus"],
    "writeTecplotSurfaceSolution": True,
    "writeSurfaceSolution": False,
    "writeVolumeSolution": False,
    # Physics Parameters
    "equationType": "RANS",
    "smoother": "DADI",
    "MGCycle": "sg",
    "nsubiterturb": 10,
    "nCycles": 7000,
    # ANK Solver Parameters
    "useANKSolver": True,
    "ANKSubspaceSize": 400,
    "ANKASMOverlap": 3,
    "ANKPCILUFill": 4,
    "ANKJacobianLag": 5,
    "ANKOuterPreconIts": 3,
    "ANKInnerPreconIts": 3,
    # NK Solver Parameters
    "useNKSolver": True,
    "NKSwitchTol": 1e-6,
    "NKSubspaceSize": 400,
    "NKASMOverlap": 3,
    "NKPCILUFill": 4,
    "NKJacobianLag": 5,
    "NKOuterPreconIts": 3,
    "NKInnerPreconIts": 3,
    # Termination Criteria
    "L2Convergence": 1e-14
}

ap = AeroProblem(name="STW", alpha=2.5, mach=0.85, altitude=10000, areaRef=45.5, chordRef=3.56, evalFuncs=["cl", "cd"])

options = {
    "solver": "adflow",
    "solverOptions": solverOptions,
    "gridFile": "wing_vol_L3.cgns",
    "ffdFile": "fitted_ffd.xyz",
    "liftIndex": 3,
    "aeroProblem": ap,
    "noOfProcessors": 10,
    "sliceLocation": [0.14, 2.33, 4.67, 7.0, 9.33, 11.67, 13.86],
    "writeLiftDistribution": True,
    "writeDeformedFFD": True,
    "computeVolume": True,
    "leList": [[0.01, 0.001, 0.0], [7.51, 13.99, 0.0]],
    "teList": [[4.99, 0.001, 0.0], [8.99, 13.99, 0.0]],
    "samplingCriterion": "ese",
    "alpha": "implicit",
    "targetCLTol": 1e-4,
    "startingAlpha": 3.0,
}

class Wing():

    def __init__(self, seed=None):

        pass

    def __call__(self):

        pass