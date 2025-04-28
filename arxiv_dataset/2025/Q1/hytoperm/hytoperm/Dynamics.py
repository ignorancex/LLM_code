
# external imports
import numpy as np
import matplotlib.pyplot as plt

# internal imports
from .PyPlotHelpers.Plotters import PlotObject, getAxes


class Trajectory:
    def __init__(self, x0 : np.ndarray, t0 : np.ndarray) -> None:
        self.assertDims(x0, t0)
        self.x = x0.copy()
        self.t = t0.copy()
        self._po : PlotObject = PlotObject()

    def extend(self, x : np.ndarray, t : np.ndarray) -> None:
        self.assertDims(x, t)
        self.x = np.append(self.x, x, 1)
        self.t = np.append(self.t, t)

    def appendTrajectory(self, trj : 'Trajectory') -> None:
        self.extend(trj.x, trj.t)

    def assertDims(self, x : np.ndarray, t : np.ndarray) -> None:
        if x.ndim != 2:
            raise ValueError("State array must be two-dimensional")
        if t.ndim != 1:
            raise ValueError("Time array must be one-dimensional")
        if x.shape[1] != t.shape[0]:
            raise ValueError("State and time arrays must have the same length")

    def getInitialValue(self) -> np.ndarray:
        return self.x[:,0]

    def getEndPoint(self) -> np.ndarray:
        return self.x[:,-1]

    def plot(self, ax : plt.Axes = None, **kwargs) -> PlotObject:
        ax = getAxes(ax)
        for i in range(self.x.shape[0]):
            ln = ax.plot(self.t.flatten(), self.x[i,:].flatten(), **kwargs)
            self._po.add(PlotObject(ln))
        return self._po

    def plotStateVsTime(
            self, 
            idx, 
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        ln = ax.plot(self.t.flatten(), self.x[idx,:].flatten(), **kwargs)
        self._po.add(PlotObject(ln))
        return self._po
    
    def plotStateVsState(
            self, 
            idx1, 
            idx2, 
            ax : plt.Axes = None, 
            **kwargs
            ) -> PlotObject:
        ax = getAxes(ax)
        ln = ax.plot(self.x[idx1,:],self.x[idx2, :],**kwargs)
        self._po.add(PlotObject(ln))
        return self._po
    
    def shiftTime(self, delta : float) -> None:
        self.t = self.t + delta

    def getDuration(self) -> float:
        return self.t[-1] - self.t[0]


class ControlledTrajectory(Trajectory):
    def __init__(
            self, 
            x0 : np.ndarray, 
            t0 : np.ndarray, 
            u0 : np.ndarray
            ) -> None:
        super().__init__(x0, t0)
        self.u = u0.copy()

    def extend(
            self, 
            x : np.ndarray, 
            t : np.ndarray, 
            u : np.ndarray
            ) -> None:
        self.assertDims(x, t)
        self.x = np.append(self.x, x, 1)
        self.u = np.append(self.u, u, 1)
        self.t = np.append(self.t, t)
    