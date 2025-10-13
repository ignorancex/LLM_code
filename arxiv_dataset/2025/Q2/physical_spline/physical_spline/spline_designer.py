import numpy as np
import scipy

from .utils import lgs_fitting_elements
from .utils import lgs_regularisation_elements
from .utils.spline_evaluator import HeadingSplineShell, PhysicalSpline


class SplineDesigner:
    """
    1D design for estimating f(t)
    Start with calling the "init_constant_resolution" call as many add functions as you like as often as
     you like. Call the get_spline to get the spline for all your added information
    Only call the get_spline if sufficient information is added.
    Minimum before calling "get_spline":
        Add at least one point (derivative = 0)
        Call the add_coefficient_regularisation once with low default weights
        Add another point (derivative = 0 or derivative = 1)
    """

    def __init__(self, base_function: PhysicalSpline | None = None):
        self.base_function = base_function or PhysicalSpline()

    def init_constant_resolution(
        self, time_min: float, time_max: float, delta_knot: float, time_data_to_limit: np.ndarray | None = None
    ) -> None:
        """Inits the spline on a time intervall with the given resolution. All points that
        will be added later should be in this intervall

        Args:
            time_min (float): start time of the intervall
            time_max (float): end time of the intervall
            delta_knot (float): resolution of the intervall e.g. a resolution of 0.25 means new acceleration
            values will be added each 0.25 sec inbetween is a linear interpolation of acclerations (second derivative)
            time_data_to_limit (np.ndarray, optional): A time vector. That limits the resolution
            [0,0.1,0.2,0.3,0.8,0.9,1] paired with resolution = 0.2 would give you accelerations at [0,0.2,0.8,1]
        """
        self.time_min = time_min
        self.time_max = time_max
        self.base_function = PhysicalSpline()
        self.base_function.init_for_optimisation(time_min, time_max, delta_knot, time_data_to_limit)
        self.Q = np.zeros((len(self.base_function.weights), len(self.base_function.weights)))
        self.b = np.zeros(len(self.base_function.weights))

    def add_points(
        self,
        time: np.ndarray,
        value: np.ndarray,
        weight: float = 1.0,
        derivative: int = 0,
        weight_vec: np.ndarray | None = None,
    ) -> None:
        """Adds points to the spline design, by setting derivative you can add velocities or accelerations

        Args:
            time (np.ndarray): time vector must have the same length as the value vector the pair is
            fcn(time[i]) = value[i]
            value (np.ndarray): value vector must have the same length as time vector
            weight (float, optional): How much weight you want to give the added points.
            A bigger weight means smaller deviations from the points. Defaults to 1.
            derivative (int, optional): indicates if the values are position, velocities or
            accelerations. Defaults to 0.
            weight_vec (np.ndarray, optional): Must have the same length as time and value vector.
            You can specify individual weights for each point. Defaults to None.
        """
        if weight_vec is None:
            weight_vec = np.ones(len(time)) * weight
        Q, b = lgs_fitting_elements.generate_1d_point_fitting_matrix_and_vector(
            self.base_function, time, value, derivative, weight_vec
        )
        self.Q = self.Q + Q
        self.b = self.b + b

    def add_coefficient_regularisation(
        self, weight: float = 1e-7, order: int = 0, weight_vec: np.ndarray | None = None
    ) -> None:
        """Regulize the coefficients of the fit.
        order = -1: means small sums of coeffs for the physical spline small velocities
        order = 0: means small coefficients are prefered. In the physicall spline that
        means small accelerations
        order = 1: means small coefficient differences are prefered In the physicall
        spline that means small acceleration changes
        order = 2: means small changes of the changes are prefered
        order = 3: means small changes of the (changes of the changes) are prefered

        Args:
            weight (float, optional): How much you want to weigh this regularisation/filter.
              Defaults to 10**-7.
            order (int, optional): order implemented for 0, 1, 2 and 3. Defaults to 0.
            weight_vec (lnp.ndarray, optional): must has the length of basefunction.t_knot -2 and controlls
            only for orders -1,0,1 implemented
        """
        # TODO check if we want a regularisation that is not constant over the duration
        if weight_vec is None:
            weight_vec = np.ones(len(self.base_function.t_knot) - 2)
        Q = lgs_regularisation_elements.generate_simple_regularisation_matrix(self.base_function, order, weight_vec)
        self.Q += weight * Q

    def get_spline(self) -> PhysicalSpline:
        """Calculates the spline after adding all the conditions

        Returns:
            PhysicalSpline: returns an object that can be called fcn(time,derivative)
        """
        try:
            c = np.linalg.solve(self.Q.astype(np.float64), self.b.astype(np.float64))
        except Exception as err:
            raise Exception("Add coefficient regularisation") from err
        self.base_function.weights = c
        return self.base_function


class SplineDesigner2D:
    """
    2D design for estimating x(t) and y(t)
    Start with calling the "init_constant_resolution" call as many add functions as you like
    as often as you like. Call the get_spline to get the spline for all your added information
    Only call the get_spline if sufficient information is added.
    Minimum before calling "get_spline":
        Add at least one point (derivative = 0)
        Call the add_coefficient_regularisation once with low default weights
        Add another point (derivative = 0 or derivative = 1)
    """

    def __init__(self, base_function: PhysicalSpline | None = None) -> None:
        self.base_function = base_function or PhysicalSpline()

    def init_constant_resolution(
        self, time_min: float, time_max: float, delta_knot: float, time_data_to_limit: np.ndarray | None = None
    ) -> None:
        """Inits the spline on a time intervall with the given resolution. All points that
        will be added later should be in this intervall

        Args:
            time_min (float): start time of the intervall
            time_max (float): end time of the intervall
            delta_knot (float): resolution of the intervall e.g. a resolution of 0.25 means new acceleration
            values will be added each 0.25 sec inbetween is a linear interpolation of acclerations (second derivative)
            time_data_to_limit (np.ndarray, optional): A time vector. That limits the resolution
            [0,0.1,0.2,0.3,0.8,0.9,1] paired with resolution = 0.2 would give you accelerations at [0,0.2,0.8,1]
        """
        self.time_min = time_min
        self.time_max = time_max
        self.base_function = PhysicalSpline()
        self.base_function.init_for_optimisation(time_min, time_max, delta_knot, time_data_to_limit)
        # 2 times because of 1 times for x and 1 time for y
        self.Q = np.zeros((len(self.base_function.weights) * 2, len(self.base_function.weights) * 2))
        self.b = np.zeros(len(self.base_function.weights) * 2)

    def add_points(
        self,
        time: np.ndarray,
        value_x: np.ndarray,
        value_y: np.ndarray,
        weight: float = 1.0,
        derivative: int = 0,
        weight_vec: np.ndarray | None = None,
    ) -> None:
        """Adds points to the spline design, by setting derivative you can add velocities or
        accelerations of the components

        Args:
            time (np.ndarray): time vector must have the same length as the value_x and value_y vector
            the pair is fcn_1(time[i]) = value_x[i], fcn_2(time[i]) = value_y[i]
            value_x (np.ndarray): x value vector must have the same length as time and value_y vector
            value_y (np.ndarray): y value vector must have the same length as time and value_x vector
            weight (float, optional): How much you want to weigh those points. Defaults to 1.
            derivative (int, optional): indicates if the value_x,value_y are position values or xDot,
            yDot or xDotDot, yDotDot values. Defaults to 0.
            weight_vec (np.ndarray, optional): weighing each point individuallly must have the same
            length as value_x,value_y and time. Defaults to None.
        """
        # TODO avoid unesscary evaluations of base functions that are zero anyway
        if weight_vec is None:
            weight_vec = np.ones(len(time)) * weight
        Q, b = lgs_fitting_elements.generate_2d_point_fitting_matrix_and_vector(
            self.base_function, time, value_x, value_y, derivative, weight_vec
        )
        self.Q = self.Q + Q
        self.b = self.b + b

    def add_errors_lateral_to_heading(
        self,
        time: np.ndarray,
        value_x: np.ndarray,
        value_y: np.ndarray,
        heading: np.ndarray,
        weight: float = 1.0,
        derivative: int = 0,
        weight_vec: np.ndarray | None = None,
    ) -> None:
        """Adds errors lateral to a given heading. The spline tries to not diviate from the handed
        points in the direction orthogonal to the heading. By setting derivative you can add velocities, accelerations.
        The heading value can for example from the object itself or from the road

        Args:
            time (np.ndarray): time vector must have the same length as the value_x and value_y vector
            the pair is fcn_1(time[i]) = value_x[i], fcn_2(time[i]) = value_y[i]
            value_x (np.ndarray): x value vector must have the same length as time and value_y vector
            value_y (np.ndarray): y value vector must have the same length as time and value_x vector
            heading (np.ndarray): heading at each point must have the same length as value_x, value_y
            weight (float, optional): How much to you want to weigh those lateral errors. Defaults to 1.
            derivative (int, optional): indicates if the value_x,value_y are position values or xDot,
            yDot or xDotDot, yDotDot values. Defaults to 0.
            weight_vec (np.ndarray, optional): weighing each point individuallly must have the same
            length as value_x,value_y and time. Defaults to None.
        """
        if weight_vec is None:
            weight_vec = np.ones(len(time)) * weight

        Q, b = lgs_fitting_elements.generate_lateral_point_fit_matrix_and_vector(
            self.base_function, time, value_x, value_y, heading, derivative, weight_vec
        )

        self.Q = self.Q + Q
        self.b = self.b + b

    def add_errors_longitudinal_to_heading(
        self,
        time: np.ndarray,
        value_x: np.ndarray,
        value_y: np.ndarray,
        heading: np.ndarray,
        weight: float = 1.0,
        derivative: int = 0,
        weight_vec: np.ndarray | None = None,
    ) -> None:
        """Adds errors longitudinal to a given heading. The spline tries to not diviate from the handed
        points in the direction of the heading. By setting derivative you can add velocities, accelerations.
        The heading value can for example from the object itself or from the road

        Args:
            time (np.ndarray): time vector must have the same length as the value_x and value_y vector
            the pair is fcn_1(time[i]) = value_x[i], fcn_2(time[i]) = value_y[i]
            value_x (np.ndarray): x value vector must have the same length as time and value_y vector
            value_y (np.ndarray): y value vector must have the same length as time and value_x vector
            heading (np.ndarray): heading at each point must have the same length as value_x, value_y
            weight (float, optional): How much to you want to weigh those lateral errors. Defaults to 1.
            derivative (int, optional): indicates if the value_x,value_y are position values or xDot, yDot
            or xDotDot, yDotDot values. Defaults to 0.
            weight_vec (np.ndarray, optional): weighing each point individuallly must have the same length
            as value_x,value_y and time. Defaults to None.
        """
        if weight_vec is None:
            weight_vec = np.ones(len(time)) * weight
        Q, b = lgs_fitting_elements.generate_longitudinal_point_fit_matrix_and_vector(
            self.base_function, time, value_x, value_y, heading, derivative, weight_vec
        )
        self.Q = self.Q + Q
        self.b = self.b + b

    def add_heading_regularisation(
        self, time: np.ndarray, heading: np.ndarray, weight: float = 1.0, weight_vec: np.ndarray | None = None
    ) -> None:
        """Prefers solution that have at time[i] the heading[i]

        Args:
            time (np.ndarray): time vector must have the same length as heading vector
            heading (np.ndarray): heading vector must have the same length as the time vector
            weight (float, optional): how much you want to weight the heading. Defaults to 1.
            weight_vec (np.ndarray, optional): weighing each heading point individually.
            Must have the same length as time and heading vector. Defaults to None.
        """
        # TODO avoid unesscary evaluations of base functions that are zero anyway
        if weight_vec is None:
            weight_vec = np.ones(len(time)) * weight
        Q = lgs_regularisation_elements.generate_heading_regularisation_matrix(
            self.base_function, time, heading, weight_vec
        )
        self.Q = self.Q + Q

    def add_yaw_rate_heading_regularisation(
        self,
        time: np.ndarray,
        heading: np.ndarray,
        yaw_rate: np.ndarray,
        weight: float = 1.0,
        weight_vec: np.ndarray | None = None,
    ) -> None:
        """Includes yaw rate information (derivative of heading reg)

        Args:
            time (np.ndarray): time vector must have the same length as heading vector
            heading (np.ndarray): heading vector must have the same length as the time vector
            yaw_rate (np.ndarray): yaw_rate vector must have the same length as the time vector
            weight (float, optional): how much you want to weight the heading. Defaults to 1.
            weight_vec (np.ndarray | None, optional): weighing each heading point individually.
            Must have the same length as time and heading vector. Defaults to None.
        """
        if weight_vec is None:
            weight_vec = np.ones(len(time)) * weight
        Q = lgs_regularisation_elements.generate_yaw_rate_heading_matrix(
            self.base_function, time, heading, yaw_rate, weight_vec
        )
        self.Q = self.Q + Q

    def add_coefficient_regularisation(
        self, weight: float = 1e-7, order: int = 0, weight_vec: np.ndarray | None = None
    ) -> None:
        """Regulize the coefficients of the fit.
        order = -1: means small sums of coeffs for the physical spline small velocities
        order = 0: means small coefficients are prefered. In the physicall spline that
        means small accelerations
        order = 1: means small coefficient differences are prefered In the physicall
        spline that means small acceleration changes
        order = 2: means small changes of the changes are prefered
        order = 3: means small changes of the (changes of the changes) are prefered

        Args:
            weight (float, optional): How much you want to weigh this regularisation/filter. Defaults to 10**-7.
            order (int, optional): order implemented for 0, 1, 2 and 3. Defaults to 0.
            weight_vec (np.ndarray, optional): must has the length of basefunction.t_knot -2 and controlls
            only for orders -1,0,1 implemented
        """
        # TODO check if we want a regularisation that is not constant over the duration
        if weight_vec is None:
            weight_vec = np.ones(len(self.base_function.t_knot) - 2)
        Q_one_dimension = lgs_regularisation_elements.generate_simple_regularisation_matrix(
            self.base_function, order, weight_vec
        )
        self.Q = self.Q + weight * scipy.linalg.block_diag(Q_one_dimension, Q_one_dimension)

    def get_spline(self) -> tuple[PhysicalSpline, PhysicalSpline]:
        """Calculates the splines after adding all the conditions

        Returns:
            tuple[PhysicalSpline,PhysicalSpline]: Returns a spline for x(t) and y(t). Both can be
            evaluated with x(time,derivative) and y(time,derivative)
        """
        try:
            c = np.linalg.solve(self.Q.astype(np.float64), self.b.astype(np.float64))
        except Exception as err:
            raise Exception("Add coefficient regularisation") from err
        c_x = c[: len(self.base_function.weights)]
        c_y = c[len(self.base_function.weights) :]
        base_x = self.base_function.copy()
        base_x.weights = c_x
        base_y = self.base_function.copy()
        base_y.weights = c_y
        return base_x, base_y


class SplineDesignerHeading:
    """
    estimate psi(t) . Uses a transformation to estimate sin(psi(t)) and cos(psi(t))
    Start with calling the "init_constant_resolution" call as many add functions as you like
    as often as you like. Call the get_spline to get the spline for all your added information
    Only call the get_spline if sufficient information is added.
    Minimum before calling "get_spline":
        Add at least one point (derivative = 0)
        Call the add_coefficient_regularisation once with low default weights
        Add another point (derivative = 0 or derivative = 1)
    """

    def __init__(self, base_function: PhysicalSpline | None = None) -> None:
        self.base_function = base_function or PhysicalSpline()

    def init_constant_resolution(
        self, time_min: float, time_max: float, delta_knot: float, time_data_to_limit: np.ndarray | None = None
    ) -> None:
        """Inits the spline on a time intervall with the given resolution. All points that
        will be added later should be in this intervall

        Args:
            time_min (float): start time of the intervall
            time_max (float): end time of the intervall
            delta_knot (float): resolution of the intervall e.g. a resolution of 0.25 means new acceleration
            values will be added each 0.25 sec inbetween is a linear interpolation of acclerations (second derivative)
            time_data_to_limit (np.ndarray, optional): A time vector. That limits the resolution
            [0,0.1,0.2,0.3,0.8,0.9,1] paired with resolution = 0.2 would give you accelerations at [0,0.2,0.8,1]
        """
        self.time_min = time_min
        self.time_max = time_max
        self.base_function = PhysicalSpline()
        self.base_function.init_for_optimisation(time_min, time_max, delta_knot, time_data_to_limit)
        # 2 times because of 1 times for x and 1 time for y
        self.Q = np.zeros((len(self.base_function.weights) * 2, len(self.base_function.weights) * 2))
        self.b = np.zeros(len(self.base_function.weights) * 2)

    def add_coefficient_regularisation(
        self, weight: float = 1e-7, order: int = 0, weight_vec: np.ndarray | None = None
    ) -> None:
        """Regulize the coefficients of the fit.
        order = -1: means small sums of coeffs for the physical spline small velocities
        order = 0: means small coefficients are prefered. In the physicall spline that
        means small accelerations
        order = 1: means small coefficient differences are prefered In the physicall
        spline that means small acceleration changes
        order = 2: means small changes of the changes are prefered
        order = 3: means small changes of the (changes of the changes) are prefered

        Args:
            weight (float, optional): How much you want to weigh this regularisation/filter. Defaults to 10**-7.
            order (int, optional): order implemented for 0, 1, 2 and 3. Defaults to 0.
            weight_vec (np.ndarray, optional): must has the length of basefunction.t_knot -2 and controlls
            only for orders -1,0,1 implemented
        """
        # TODO check if we want a regularisation that is not constant over the duration
        if weight_vec is None:
            weight_vec = np.ones(len(self.base_function.t_knot) - 2)
        Q_one_dimension = lgs_regularisation_elements.generate_simple_regularisation_matrix(
            self.base_function, order, weight_vec
        )
        self.Q = self.Q + weight * scipy.linalg.block_diag(Q_one_dimension, Q_one_dimension)


    def add_heading_points(
        self,
        time: np.ndarray,
        heading: np.ndarray,
        weight: float = 1.0,
        weight_vec: np.ndarray | None = None,
    ) -> None:
        """Adds heading points to the estimation

        Args:
            time (np.ndarray): time vector must have the same length as the value vector the pair is
            fcn(time[i]) = value[i]
            heading (np.ndarray): value vector must have the same length as time vector
            weight (float, optional): How much weight you want to give the added points.
            A bigger weight means smaller deviations from the points. Defaults to 1.
            weight_vec (list, optional): Must have the same length as time and value vector.
            You can specify individual weights for each point. Defaults to None.
        """
        if weight_vec is None:
            weight_vec = np.ones(len(time)) * weight
        Q, b = lgs_fitting_elements.generate_2d_point_fitting_matrix_and_vector(
            self.base_function, time, np.sin(np.array(heading)), np.cos(np.array(heading)), 0, weight_vec
        )
        self.Q = self.Q + Q
        self.b = self.b + b

    def add_velocities_to_heading(
        self,
        time: np.ndarray,
        value_x_dot: np.ndarray,
        value_y_dot: np.ndarray,
        weight: float = 1.0,
        weight_vec: np.ndarray | None = None,
    ) -> None:
        """Add cartesian velocities to the heading estimation

        Args:
            time (np.ndarray): time vector must have the same length as the value vector the pair is
            fcn(time[i]) = value[i]
            value_x_dot (np.ndarray): value vector must have the same length as time vector
            value_y_dot (np.ndarray): value vector must have the same length as time vector
            weight (float, optional): How much weight you want to give the added points.
            A bigger weight means smaller deviations from the points. Defaults to 1.
            weight_vec (list, optional): Must have the same length as time and value vector.
            You can specify individual weights for each point. Defaults to None.
        """
        if weight_vec is None:
            weight_vec = np.ones(len(time)) * weight
        Q, b = lgs_fitting_elements.generate_velocity_heading_matrix_and_vector(
            self.base_function, time, value_x_dot, value_y_dot, weight_vec
        )
        self.Q += Q
        self.b += b

    def add_positions_to_heading(
        self,
        time: np.ndarray,
        value_x: np.ndarray,
        value_y: np.ndarray,
        weight: float = 1.0,
        min_dist: float = 4.0,
        weight_vec: np.ndarray | None = None,
    ) -> None:
        """Add position information to the optimization. Effect depends on the helper function

        Args:
            time (np.ndarray): time vector must have the same length as the value vector the pair is
            fcn(time[i]) = value[i]
            value_x (np.ndarray): value vector must have the same length as time vector
            value_y (np.ndarray): value vector must have the same length as time vector
            weight (float, optional): How much weight you want to give the added points.
            A bigger weight means smaller deviations from the points. Defaults to 1.
            min_dist (float, optional): minimal distance between points for heading estimation. Defaults to 1.
            weight_vec (np.ndarray | None, optional): Must have the same length as time and value vector.
            You can specify individual weights for each point. Defaults to None.
        """
        if weight_vec is None:
            weight_vec = np.ones(len(time)) * weight
        Q, b = lgs_fitting_elements.generate_heading_position_matrix_and_vector(
            self.base_function, time, value_x, value_y, min_dist, weight_vec
        )
        self.Q += Q
        self.b += b

    def add_accelerations_to_heading(
        self,
        time: np.ndarray,
        value_x_dot: np.ndarray,
        value_y_dot: np.ndarray,
        value_x_dotdot: np.ndarray,
        value_y_dotdot: np.ndarray,
        weight: float = 1.0,
        weight_vec: np.ndarray | None = None,
    ) -> None:
        """Adds Acceleration information. Also needs velocity information because the affect of
        acceleration on yaw rate depends on velocity

        Args :
            time (np.ndarray): _description_
            value_x_dot (np.ndarray): _description_
            value_y_dot (np.ndarray): _description_
            value_x_dotdot (np.ndarray): _description_
            value_y_dotdot (np.ndarray): _description_
            weight (float, optional): _description_. Defaults to 1.0.
            weight_vec (np.ndarray | None, optional): _description_. Defaults to None.
        """
        if weight_vec is None:
            weight_vec = np.ones(len(time)) * weight
        Q, b = lgs_fitting_elements.generate_acceleration_heading_matrix_vector(
            self.base_function, time, value_x_dot, value_y_dot, value_x_dotdot, value_y_dotdot, weight_vec
        )
        self.Q += Q
        self.b += b

    def get_spline(self) -> HeadingSplineShell:
        """Calculates the splines after adding all the conditions

        Returns:
            HeadingSplineShell: Returns a spline for x(t) and y(t). Both can be
            evaluated with fcn(time,derivative)
        """
        try:
            c = np.linalg.solve(self.Q, self.b)
        except Exception as error:
            raise Exception("Add coefficient regularisation") from error
        c_x = c[: len(self.base_function.weights)]
        c_y = c[len(self.base_function.weights) :]
        base_sin = self.base_function.copy()
        base_sin.weights = c_x
        base_cos = self.base_function.copy()
        base_cos.weights = c_y
        return HeadingSplineShell(base_sin, base_cos)
