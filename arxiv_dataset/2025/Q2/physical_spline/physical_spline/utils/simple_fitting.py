import numpy as np
import pandas as pd


def estimate_locale_heading(
    time: np.ndarray, value_x: np.ndarray, value_y: np.ndarray, min_dist: float = 4.0
) -> np.ndarray:
    """Estimates the heading based on local information, by taking the current point, the first point
    infront that is at least min_dist away and the next point that is a least min_dist away

    Args:
        time (np.ndarray): time vector must have the same length as value vectors
        value_x (np.ndarray): value vector
        value_y (np.ndarray): value vector
        min_dist (float): minimal distance between points for heading estimation

    Returns:
        np.ndarray: heading estimation based on local information
    """
    time = time
    value_x = value_x
    value_y = value_y
    time_reduced = [time[0]]
    value_x_reduced = [value_x[0]]
    value_y_reduced = [value_y[0]]

    for i in range(1, len(time)):
        if np.linalg.norm([value_x[i] - value_x_reduced[-1], value_y[i] - value_y_reduced[-1]]) > min_dist:
            # Only large position deltas are considered for estimation. We could also include the hole intervall
            # of Points. The downside would be that standing points would be over represented. A way to
            # compensated this would be a weighing based on delta distance (TODO for improvement)
            value_x_reduced.append(value_x[i])
            value_y_reduced.append(value_y[i])
            time_reduced.append(time[i])
    heading = []
    for j in range(1, len(time_reduced) - 1):
        heading.append(
            estimate_heading_float(
                [value_x_reduced[j - 1] - value_x_reduced[j], 0, value_x_reduced[j + 1] - value_x_reduced[j]],
                [value_y_reduced[j - 1] - value_y_reduced[j], 0, value_y_reduced[j + 1] - value_y_reduced[j]],
            )
        )
        if len(heading) > 1:
            if (heading[-1] - heading[-2]) > 5:
                heading[-1] = heading[-1] - np.pi * 2
            if (heading[-2] - heading[-1]) > 5:
                heading[-1] = heading[-1] + np.pi * 2
    if len(time_reduced) <= 2 or len(heading) <= 1:
        return np.array([])

    df = pd.merge_asof(pd.DataFrame({"time": time}),pd.DataFrame({"time": time_reduced[1:-1], "heading": heading}), on = 'time')
    return df["heading"].bfill().ffill().to_numpy(dtype=np.float64)


def estimate_heading_float(value_x: list[float], value_y: list[float]) -> float:
    """Estimates heading through the point by fitting a straight line

    Args:
        value_x (np.ndarray): x points, must have the same length as y points
        value_y (np.ndarray): y points, must have the same length as x points

    Returns :
        float: estimated heading
    """
    deltaHeading = 0.0

    # Variance switch (rotate by 90° to avoid step slopes
    if (max(value_x) - min(value_x)) < (max(value_y) - min(value_y)):
        temp = value_x
        value_x = value_y
        value_y = list(map(lambda x: -x, temp))
        deltaHeading = np.pi / 2

    # regression of a straight line
    Q = np.zeros(2)
    b = np.array([0, 0])
    for j in range(len(value_x)):
        baseFunction = np.array([value_x[j], 1])
        Q = Q + np.outer(np.transpose(baseFunction), baseFunction)
        b = b + baseFunction * value_y[j]
    Reg = np.array([[1, 0], [0, 1]])
    alpha = 1e-7
    coeffs = list(np.linalg.solve(Q + alpha * Reg, b))
    heading = np.arctan2(coeffs[0], 1)
    if value_x[0] > value_x[len(value_x) - 1]:
        heading = heading + np.pi
    return heading + deltaHeading
