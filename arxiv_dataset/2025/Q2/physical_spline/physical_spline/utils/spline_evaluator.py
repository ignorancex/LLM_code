from __future__ import annotations

import abc
from collections.abc import Iterable
from typing import overload

import numpy as np


class SplineEvaluator:
    @abc.abstractmethod
    def eval_basis(self, time: float, k: int, derivative: int = 0) -> float: ...

    @overload
    @abc.abstractmethod
    def __call__(self, time: float, derivative: int = ...) -> float: ...

    @overload
    @abc.abstractmethod
    def __call__(self, time: np.ndarray, derivative: int = ...) -> np.ndarray: ...

    @abc.abstractmethod
    def __call__(self, time: float | np.ndarray, derivative: int = 0) -> float | np.ndarray: ...

    @abc.abstractmethod
    def copy(self) -> object: ...


class PhysicalSpline(SplineEvaluator):
    """
    Spline model with physical parameters:

    t_knot: vector is structured [0,0,0,delta_t,2*delta_t, ...]+t_min at each position
    you have the acceleration weights[i] .
    weights: Acceleration[t_knot[i]] = weights[i] . With the exception of the first two
    weights which are initial position and initial velocity. Inbetween the acceleration gets linear interpolated

    callable(time, derivative)
    """

    def __init__(self, t_knot: np.ndarray | None = None, weights: np.ndarray | None = None):
        """Inits the Physical Spline t_knot and weight vector

        Args:
            t_knot (np.ndarray, optional): . Defaults to None.
            weights (np.ndarray, optional): _description_. Defaults to None.
        """
        self._t_knot = t_knot if t_knot is not None else np.array([0.0])
        self._weights = weights if weights is not None else np.array([0.0])

    def copy(self) -> PhysicalSpline:
        """Returns a copy of itself

        Returns:
            PhysicalSpline: copy of itself
        """
        return PhysicalSpline(self.t_knot, self.weights)

    @property
    def weights(self) -> np.ndarray:
        """specify the weight to the corresponding base. The resulting function will be build by
        weight[0]*base_0+weight[1]*base_1+...
        base_k = self.eval_basis(time, k)
        """
        return self._weights

    @weights.setter
    def weights(self, value: np.ndarray | None) -> None:
        """sets the weights of the spline model

        Args:
            weights (np.ndarray): weights of the spline model [initial_position, initial_velocity,
            initial_acceleration, acceleration(t_knot[3], acceleration(t_knot[4], ... ]
        """
        if value is None:
            value = np.array([0.0])
        self._weights = value

    @property
    def t_knot(self) -> np.ndarray:
        return self._t_knot

    @t_knot.setter
    def t_knot(self, value: np.ndarray | None) -> None:
        if value is None:
            value = np.array([0.0])
        self._t_knot = value

    def init_for_optimisation(
        self,
        time_min: float,
        time_max: float,
        delta_knot: float,
        time_data_to_limit: np.ndarray | None = None,
    ) -> None:
        """Inits the spline on a time intervall with the given resolution.
        All points that will be added later should be in this intervall

        Args:
            time_min (float): start time of the intervall
            time_max (float): end time of the intervall
            delta_knot (float): resolution of the intervall e.g. a resolution of 0.25 means new
            acceleration values will be added each 0.25 sec inbetween is a linear interpolation
            of acclerations (second derivative)
            time_data_to_limit (np.ndarray, optional): A time vector. That limits the resolution
            [0,0.1,0.2,0.3,0.8,0.9,1] paired with resolution = 0.2 would give you accelerations at [0,0.2,0.8,1]
        """
        t_knot = [time_min, time_min, time_min]
        last_time_stamp = time_min
        if time_data_to_limit is None:
            while t_knot[-1] + delta_knot < time_max:
                last_time_stamp += delta_knot
                t_knot.append(last_time_stamp)
        else:
            for i in range(len(time_data_to_limit) - 1):
                if (
                    time_data_to_limit[i] >= last_time_stamp + delta_knot
                    and last_time_stamp + delta_knot <= time_data_to_limit[-1] - delta_knot
                ):
                    last_time_stamp = max(last_time_stamp + delta_knot, time_data_to_limit[i])
                    t_knot.append(last_time_stamp)
        t_knot.extend([last_time_stamp + delta_knot, last_time_stamp + 2 * delta_knot])
        self.t_knot = np.array(t_knot)
        self.weights = np.array([0.0] * len(self.t_knot))

    @overload
    def __call__(self, time: float, derivative: int = ...) -> float: ...

    @overload
    def __call__(self, time: np.ndarray, derivative: int = ...) -> np.ndarray: ...

    def __call__(self, time: float | np.ndarray, derivative: int = 0) -> float | np.ndarray:
        """makes the spline a callable object that take time and derivative to evaluate the spline

        Args:
            time (Union[float,np.ndarray]): _description_
            derivative (int, optional): _description_. Defaults to 0.

        Returns:
            Union[float,np.ndarray]: fcn value
        """

        if isinstance(time, Iterable):
            np_all_basis_all_time = eval_all_base_all_time(self, time, derivative)
            return np.einsum("ji,i->j", np_all_basis_all_time, self.weights)
        np_all_basis_all_time = eval_all_base_at_time(self, float(time), derivative)
        return np.dot(np_all_basis_all_time, self.weights)

    def eval_basis(self, time: float, k: int, derivative: int = 0) -> float:
        """Evaluates derivative of the k-th basefunction of the spline.

        Args:
            time (float): time you want to evaluate the basefunction
            k (int): the k-th basefunction to evaluate
            derivative (int, optional): Which derivative you want to evaluate. Defaults to 0.

        Returns:
            _type_: f_k(time)
        """
        if time < self.t_knot[0] or time > self.t_knot[-1]:
            return 0
        if derivative == 0:
            if k == 0:
                return 1
            if k == 1:
                return time - self.t_knot[2]
            if time < self.t_knot[k - 1]:
                val = 0.0
            elif time < self.t_knot[k]:
                val = 1 / 6 * (time - self.t_knot[k - 1]) ** 3 / (self.t_knot[k] - self.t_knot[k - 1])
            elif k < len(self.t_knot) - 1 and time <= self.t_knot[k + 1]:
                val = (
                    -1 / 6 * (time - self.t_knot[k]) ** 3 / (self.t_knot[k + 1] - self.t_knot[k])
                    + 0.5 * (time - self.t_knot[k]) ** 2
                    + 0.5 * (self.t_knot[k] - self.t_knot[k - 1]) * (time - self.t_knot[k])
                    + 1 / 6 * (self.t_knot[k] - self.t_knot[k - 1]) ** 2
                )
            elif k < len(self.t_knot) - 1:
                val = (
                    0.5 * (self.t_knot[k + 1] - self.t_knot[k - 1]) * (time - self.t_knot[k + 1])
                    + 1 / 3 * (self.t_knot[k + 1] - self.t_knot[k]) ** 2
                    + 1 / 6 * (self.t_knot[k] - self.t_knot[k - 1]) ** 2
                    + 1 / 2 * (self.t_knot[k] - self.t_knot[k - 1]) * (self.t_knot[k + 1] - self.t_knot[k])
                )
            else:
                val = 0.0
            return val

        if derivative == 1:
            if k == 0:
                return 0.0
            if k == 1:
                return 1.0
            if time < self.t_knot[k - 1]:
                val = 0.0
            elif time < self.t_knot[k]:
                val = 0.5 * (time - self.t_knot[k - 1]) ** 2 / (self.t_knot[k] - self.t_knot[k - 1])
            elif k < len(self.t_knot) - 1 and time <= self.t_knot[k + 1]:
                val = (
                    -0.5 * (time - self.t_knot[k]) ** 2 / (self.t_knot[k + 1] - self.t_knot[k])
                    + (time - self.t_knot[k])
                    + 0.5 * (self.t_knot[k] - self.t_knot[k - 1])
                )
            elif k < len(self.t_knot) - 1:
                val = 0.5 * (self.t_knot[k + 1] - self.t_knot[k - 1])
            else:
                val = 0.0
            return val

        if derivative == 2:
            if k == 0:
                return 0.0
            if k == 1:
                return 0.0
            if k > 0 and time < self.t_knot[k - 1]:
                return 0.0
            if time < self.t_knot[k]:
                return (time - self.t_knot[k - 1]) / (self.t_knot[k] - self.t_knot[k - 1])
            if k < len(self.t_knot) - 1 and time <= self.t_knot[k + 1]:
                return -(time - self.t_knot[k]) / (self.t_knot[k + 1] - self.t_knot[k]) + 1
        return 0.0

    def eval_basis_vectorized(self, time: np.ndarray, k: int, derivative: int = 0) -> np.ndarray:
        """Evaluates derivative of the k-th basefunction of the spline for the time
        vector by vectorizing for a faster evaluation

        Args:
            time (np.ndarray): numpy array of the time vector to evaluate on
            k (int): the k-th basefunction to evaluate
            derivative (int, optional): Which derivative you want to evaluate. Defaults to 0.

        Returns:
            _type_: f_k(time)
        """
        if k == 0:
            if derivative == 0:
                return np.ones(len(time))
            if derivative == 1:
                return np.zeros(len(time))
            if derivative == 2:
                return np.zeros(len(time))
        if k == 1:
            if derivative == 0:
                return time - self.t_knot[2] * np.ones(len(time))
            if derivative == 1:
                return np.ones(len(time))
            if derivative == 2:
                return np.zeros(len(time))
        dt_last = self.t_knot[k] - self.t_knot[max(k - 1, 0)]
        dt_next = self.t_knot[min(k + 1, len(self.t_knot) - 1)] - self.t_knot[k]
        dt = time - self.t_knot[k]
        return eval_basis_despite_start_basis_vectorized(dt, dt_last, dt_next, derivative)

    def eval_all_basis_vectorized(self, time: np.ndarray, derivative: int = 0) -> np.ndarray:
        """Evaluates derivative of all basisfunctions over the time vector and stores it in a
        matrix with time dimension and the k_th base derivate for the other dimension
        Faster than looping by vectorizing

        Args:
            time (np.ndarray): numpy array of the time vector to evaluate on
            derivative (int, optional): Which derivative you want to evaluate. Defaults to 0.

        Returns:
            np.ndarray: Matrix with all basis at all times stamps
        """
        dt_last = self.t_knot[-1] - self.t_knot[-2]
        dt_next = dt_last
        dt = time * np.ones((len(self.t_knot[3:]), 1)) - np.transpose((self.t_knot[3:]) * np.ones((len(time), 1)))
        base_1 = np.zeros(len(time))
        base_2 = np.zeros(len(time))
        base_3 = eval_basis_despite_start_basis_vectorized(time - self.t_knot[2], 0, dt_next, derivative)
        if derivative == 0:
            base_1 = np.ones(len(time))
            base_2 = time - self.t_knot[2] * np.ones(len(time))
        if derivative == 1:
            base_1 = np.zeros(len(time))
            base_2 = np.ones(len(time))
        if derivative == 2:
            base_1 = np.zeros(len(time))
            base_2 = np.zeros(len(time))
        val = eval_basis_despite_start_basis_vectorized(dt, dt_last, dt_next, derivative)
        return np.c_[base_1, base_2, base_3, np.transpose(val)]


def eval_basis_despite_start_basis_vectorized(
    dt: np.ndarray, dt_last: float, dt_next: float, derivative: int
) -> np.ndarray:
    """Evals Basis after the 3 initial state basis for the Matrix dt. dt has the
    time vector shifted by the base center as rows.
    e.g.
    [-1,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6, ...
    -2,-1.8,-1.6,-1.4,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,...]

    For a basis each second (resolution t_knot)

    Args:
        dt (np.ndarray): Matrix with time vetors shifted around the base center
        dt_last (float): width of the basis typically for a resoultion of 1 this is 1
        dt_next (float): width of the basis typically for a resoultion of 1 this is 1
        derivative (int): Which derivatives of basefunctions do you want

    Returns:
        np.ndarray: basefunction values at timestamp for kth base
    """
    if derivative == 0:
        val1 = (
            np.where((dt >= -dt_last) & (dt < 0), 1 / 6 * (dt + dt_last) ** 3 / dt_last, 0) if (dt_last > 0) else dt * 0
        )
        val2 = (
            np.where(
                (dt >= 0) & (dt <= dt_next),
                -1 / 6 * dt**3 / dt_next + 0.5 * dt**2 + 0.5 * dt_last * dt + 1 / 6 * dt_last**2,
                0,
            )
            if (dt_next > 0)
            else np.zeros_like(dt)
        )
        val3 = np.where(
            (dt > dt_next),
            0.5 * (dt_next + dt_last) * (dt - dt_next)
            + 1 / 3 * dt_next**2
            + 1 / 6 * dt_last**2
            + 1 / 2 * dt_last * dt_next,
            0,
        )

    if derivative == 1:
        val1 = (
            np.where((dt >= -dt_last) & (dt < 0), 0.5 * (dt + dt_last) ** 2 / dt_last, 0) if (dt_last > 0) else dt * 0
        )
        val2 = (
            np.where((dt >= 0) & (dt <= dt_next), -0.5 * dt**2 / dt_next + dt + 0.5 * dt_last, 0)
            if (dt_next > 0)
            else np.zeros_like(dt)
        )
        val3 = np.where((dt > dt_next), 0.5 * (dt_next + dt_last), 0)

    if derivative == 2:
        val1 = np.where((dt >= -dt_last) & (dt < 0), (dt + dt_last) / dt_last, 0) if (dt_last > 0) else dt * 0
        val2 = np.where((dt >= 0) & (dt <= dt_next), -dt / dt_next + 1, 0) if (dt_next > 0) else dt * 0
        val3 = np.zeros_like(dt)

    return val1 + val2 + val3


class HeadingSplineShell:
    """Serves as a box that stores to fitted splines that represent sin an cos of the fitted signal.
    This class does the backtransformation when you call it
    """

    def __init__(self, spline_evaluator_sin: PhysicalSpline, spline_evaluator_cos: PhysicalSpline) -> None:
        self.spline_evaluator_sin = spline_evaluator_sin
        self.spline_evaluator_cos = spline_evaluator_cos

    @overload
    def __call__(self, time: float, derivative: int = ...) -> float: ...

    @overload
    def __call__(self, time: np.ndarray, derivative: int = ...) -> np.ndarray: ...

    def __call__(self, time: float | np.ndarray, derivative: int = 0) -> float | np.ndarray:
        """makes the spline a callable object that take time and derivative to evaluate the spline

                Args:
        <<<<<<< HEAD
                    time (Union[float,np.ndarray]): _description_
                    derivative (int, optional): _description_. Defaults to 0.

                Returns:
                    Union[float,np.ndarray]: fcn value
        =======
                    time (float | list[float]): _description_
                    derivative (int, optional): _description_. Defaults to 0.

                Returns:
                    float | list[float]: fcn value
        >>>>>>> d009900a15259d3364ebb0dc9cc8201a9a0d6875
        """
        fcn_sin = self.spline_evaluator_sin(time, 0)
        fcn_cos = self.spline_evaluator_cos(time, 0)
        if derivative == 0:
            return np.arctan2(fcn_sin, fcn_cos)
        fcn_sin_dot = self.spline_evaluator_sin(time, 1)
        fcn_cos_dot = self.spline_evaluator_cos(time, 1)
        base_L2_norm = fcn_sin**2 + fcn_cos**2
        if derivative == 1:
            return (fcn_sin_dot * fcn_cos - fcn_sin * fcn_cos_dot) / base_L2_norm
        fcn_sin_dot_dot = self.spline_evaluator_sin(time, 2)
        fcn_cos_dot_dot = self.spline_evaluator_cos(time, 2)
        if derivative == 2:
            value = (fcn_sin_dot_dot * fcn_cos - fcn_sin * fcn_cos_dot_dot) / base_L2_norm
            return (
                value
                - 2
                * (fcn_sin * fcn_sin_dot + fcn_cos * fcn_cos_dot)
                * (fcn_sin_dot * fcn_cos - fcn_sin * fcn_cos_dot)
                / base_L2_norm
                / base_L2_norm
            )
        return np.zeros_like(time)


def eval_all_base_at_time(base_function: PhysicalSpline, time_eval: float, derivative: int) -> np.ndarray:
    """Evaluates all basefunctions for a given time

    Args:
        base_function (PhysicalSpline): Basefunction object
        time_eval (float): time where you want to evaluate
        derivative (int): derivative you want to evaluate

    Returns :
        np.ndarray: all base evaluations in a vector
    """
    return np.array([base_function.eval_basis(time_eval, i, derivative) for i in range(len(base_function.weights))])


def eval_all_base_all_time(base_function: PhysicalSpline, time: np.ndarray, derivative: int) -> np.ndarray:
    """Evaluates derivative of all basisfunctions over the time vector and stores it
    in a matrix with time dimension and the k_th base derivate for the other dimension
    Faster than looping by vectorizing

    Args:
        base_function (PhysicalSpline): Basefunction object
        time (np.ndarray): numpy array of the time vector to evaluate on
        derivative (int, optional): Which derivative you want to evaluate. Defaults to 0.

    Returns:
        np.ndarray: Matrix with all basis at all times stamps
    """
    return base_function.eval_all_basis_vectorized(np.array(time), derivative)
    # TODO decide if we want to remove this function and just call it at the object
