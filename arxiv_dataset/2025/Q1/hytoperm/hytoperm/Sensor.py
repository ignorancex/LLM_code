
# external imports
import casadi as cad

# internal imports
from .World import *


'''
SensingQualityFunction: Abstract class definition for any sensing function.

Created on: Mar 28 2024

@author: Jonas Hall
'''
class SensingQualityFunction:
    def __init__(self):
        pass

    def __call__(self, p, q):
        pass


class ConstantQualityFunction(SensingQualityFunction):
    def __init__(self, c = 1):
        self._c : float = None
        self.assignConstant(c)

    def assignConstant(self, c : float) -> None:
        if c < 0 or c > 1:
            raise ValueError("Constant must be in the interval [0, 1].")
        self._c = c

    def __call__(self, p, q):
        return self._c


class GaussianQualityFunction(SensingQualityFunction):
    def __init__(self, c1 = 50, c2 = 1):
        self._c1 : float = None
        self._c2 : float = None
        self.assignConstants(c1, c2)

    def assignConstants(self, c1 : float, c2 : float) -> None:
        if c1 <= 0 or c2 <= 0:
            raise ValueError("Constants must be positive.")
        try:
            self._c1 = float(c1)
            self._c2 = float(c2)
        except ValueError:
            raise ValueError("Constants must be a number.")            

    def __call__(self, p, q):
        delta = p - q
        sqr_dist = cad.dot(delta, delta)
        return self._c2*cad.exp(-self._c1*sqr_dist)


class SinusoidalQualityFunction(SensingQualityFunction):
    def __init__(self, c1 = 3.0, c2 = 20.0, c3 = 40.0):
        self._c1 : float = None
        self._c2 : float = None
        self._c3 : float = None
        self.assignConstants(c1, c2, c3)

    def assignConstants(self, c1 : float, c2 : float, c3 : float) -> None:
        if c3 <= 0:
            raise ValueError("Constant c3 must be positive.")
        try:
            self._c1 = float(c1)
            self._c2 = float(c2)
            self._c3 = float(c3)
        except ValueError:
            raise ValueError("Constants must be numbers.")

    def __call__(self, p, q):
        delta = p - q
        expArg = -self._c3*cad.dot(delta,delta)
        sinArg = self._c1*delta[0]
        cosArg = self._c2*delta[1]
        return 0.5*cad.exp(expArg)*(cad.sin(sinArg)**2 + cad.cos(cosArg)**2)


'''
Sensor: A collection of sensing quality functions and measurement models 
        for each target. 

Created on: Mar 28 2024

@author: Jonas Hall
'''
class Sensor:
    def __init__(self, p : np.ndarray = None) -> None:
        self._p : np.ndarray = None
        self._ttsqm : Dict[Target, SensingQualityFunction] = {}                 # target to sensing quality function mapper
        self._ttHm : Dict[Target, np.ndarray] = {}                              # target to measurement matrix mapper
        self._ttRm : Dict[Target, np.ndarray] = {}                              # target to measurement noise mapper
        self._ttRinvm : Dict[Target, np.ndarray] = {}                           # target to measurement noise inverse mapper
        
        if p is not None:
            self.setPosition(p)

    # getters
    def getPosition(self) -> np.ndarray:
        return self._p
    
    def targetToSQFMapper(self) -> Dict[Target, SensingQualityFunction]:
        return self._ttsqm
    
    def getQualityFunction(self, target : Target) -> SensingQualityFunction:
        return self.targetToSQFMapper()[target]
    
    def getSensingQuality(self, target : Target) -> float:
        p = self.getPosition()
        return self.getQualityFunction(target)(p, target.p())
    
    def getMeasurementMatrix(self, target : Target) -> np.ndarray:
        return self._ttHm[target]

    def getMeasurementNoiseMatrix(self, target : Target) -> np.ndarray:
        return self._ttRm[target]

    def getMeasurementNoiseInverse(self, target : Target) -> np.ndarray:
        return self._ttRinvm[target]

    def getMeasurement(self, p : np.ndarray, target : Target):
        q = target.p()
        quality = self.getSensingQuality(target)(p, q)
        H = self.getMeasurementMatrix(target)
        targetState = target.internalState()
        return quality * H @ targetState + np.random.normal(0, 0.1, H.shape[0])   
    
    def drawNoise(self, target : Target) -> np.ndarray:
        R = self.getMeasurementNoiseMatrix(target)
        return np.random.multivariate_normal(0, R)
    
    # setters
    def setPosition(self, p : np.ndarray) -> None:
        if not isinstance(p, np.ndarray):
            raise ValueError("Position must be a numpy array.")
        self._p = p

    def setTargetQualityFunction(
            self, 
            target : Target,
            sqf : SensingQualityFunction
            ) -> None:
        if not isinstance(target, Target):
            raise ValueError("Target must be of type Target.")
        if not isinstance(sqf, SensingQualityFunction):
            raise ValueError("sqf must be of type SensingQualityFunction.")
        self.targetToSQFMapper()[target] = sqf
      
    def setNoiseMatrix(self, target : Target, R : np.ndarray) -> None:
        if not isinstance(target, Target):
            raise ValueError("Target must be of type Target.")
        if not isinstance(R, np.ndarray):
            raise ValueError("R must be a numpy array.")
        self._ttRm[target] = R
        try:
            self._ttRinvm[target] = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            raise ValueError("R must be invertible.")

    def setMeasurementMatrix(self, target : Target, H : np.ndarray) -> None:
        if not isinstance(target, Target):
            raise ValueError("Target must be of type Target.")
        if not isinstance(H, np.ndarray):
            raise ValueError("H must be a numpy array.")
        self._ttHm[target] = H
