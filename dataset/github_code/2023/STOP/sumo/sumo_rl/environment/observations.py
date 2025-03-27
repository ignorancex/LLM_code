"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
#from gymnasium import spaces
import gym

from .traffic_signal import TrafficSignal


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def observation_space(self) -> gym.spaces.Box:
        """Return the observation space."""
        return gym.spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes), dtype=np.float32),
        )

class UnboundedObservation(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the default observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        #density = self.ts.get_lanes_density()
        #queue = self.ts.get_lanes_queue()
        queues = self.ts.get_per_lane_queued()
        #wait_time = self.ts.get_accumulated_waiting_time_per_lane()
        #observation = np.array(phase_id + min_green + wait_time + queues, dtype=np.float32)
        observation = np.array(phase_id + min_green + queues, dtype=np.float32)
        return observation

    def observation_space(self) -> gym.spaces.Box:
        """Return the observation space."""
        #dim = self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes)
        dim = self.ts.num_green_phases + 1 + len(self.ts.lanes)
        return gym.spaces.Box(low = 0, high = np.inf, shape = (dim,), dtype = float)
