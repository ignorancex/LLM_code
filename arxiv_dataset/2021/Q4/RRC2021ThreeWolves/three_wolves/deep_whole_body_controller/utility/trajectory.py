import numpy as np


def Rotate(_rpy, xyz):
    xyz = np.matrix(xyz).T
    Rx = np.matrix([[1, 0, 0],
                    [0, np.cos(_rpy[0]), -np.sin(_rpy[0])],
                    [0, np.sin(_rpy[0]), np.cos(_rpy[0])]])

    Ry = np.matrix([[np.cos(_rpy[1]), 0, np.sin(_rpy[1])],
                    [0, 1, 0],
                    [-np.sin(_rpy[1]), 0, np.cos(_rpy[1])]])

    Rz = np.matrix([[np.cos(_rpy[2]), -np.sin(_rpy[2]), 0],
                    [np.sin(_rpy[2]), np.cos(_rpy[2]), 0],
                    [0, 0, 1]])
    R = Rx * Ry * Rz
    return np.array(R * xyz).flatten()

def Rotate_yaw(_rpy, xyz):
    xyz = np.matrix(xyz).T
    Rz = np.matrix([[np.cos(_rpy[2]), -np.sin(_rpy[2]), 0],
                    [np.sin(_rpy[2]), np.cos(_rpy[2]), 0],
                    [0, 0, 1]])
    return np.array(Rz * xyz).flatten()


def _gen_parabola(phase: float, start: float, mid: float, end: float) -> float:
    """Gets a point on a parabola y = a x^2 + b x + c.

  The Parabola is determined by three points (0, start), (0.5, mid), (1, end) in
  the plane.

  Args:
    phase: Normalized to [0, 1]. A point on the x-axis of the parabola.
    start: The y value at x == 0.
    mid: The y value at x == 0.5.
    end: The y value at x == 1.

  Returns:
    The y value at x == phase.
  """
    mid_phase = 0.5
    delta_1 = mid - start
    delta_2 = end - start
    delta_3 = mid_phase ** 2 - mid_phase
    coef_a = (delta_1 - delta_2 * mid_phase) / delta_3
    coef_b = (delta_2 * mid_phase ** 2 - delta_1) / delta_3
    coef_c = start

    return coef_a * phase ** 2 + coef_b * phase + coef_c


def compute_third_polynomial_trajectory(t, T):
    assert T > 0
    a0 = 0
    a1 = 0
    a2 = 3 / T ** 2
    a3 = - 2 / T ** 3
    s = a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3
    v = a1 + 2 * a2 * t + 3 * a3 * t ** 2
    a = 2 * a2 + 6 * a3 * t
    return s, v, a


def compute_fifth_polynomial_trajectory(t, T):
    assert T > 0
    # a0 = 0
    # a1 = 0
    # a2 = 0
    a3 = 10 / (T ** 3)
    a4 = - 15 / (T ** 4)
    a5 = 6 / (T ** 5)

    s = a3 * (t ** 3) + a4 * (t ** 4) + a5 * (t ** 5)
    v = 3 * a3 * (t ** 2) + 4 * a4 * (t ** 3) + 5 * a5 * (t ** 4)
    a = 6 * a3 * t + 12 * a4 * (t ** 2) + 20 * a5 * (t ** 3)
    return s, v, a


def get_path_planner(init_pos, tar_pos, start_time, reach_time):
    """Get a fifth order trajectory generator

  Args:
    init_pos: Initial position
    tar_pos: Target position
    start_time: Starting time
    reach_time: End time

  Returns:
    A function that output: position at current time, bool value for reach the target
  """
    dist = tar_pos - init_pos

    def tg(cur_time):
        t = cur_time - start_time
        T = reach_time - start_time
        if t <= T:
            return init_pos + compute_fifth_polynomial_trajectory(t, T)[0] * dist, False
        else:
            return tar_pos, t > T + 0.3     # for camera observation frequency

    return tg


def get_acc_planner(init_pos, tar_pos, start_time, reach_time):
    dist = tar_pos - init_pos

    def tg(cur_time):
        t = cur_time - start_time
        T = reach_time - start_time
        if t <= T:
            s, t_v, t_a = compute_fifth_polynomial_trajectory(t, T)
            t_s = init_pos + s * dist
            t_v *= dist
            t_a *= dist
            return [t_s, t_v, t_a]
        else:
            return [tar_pos, np.zeros(3), np.zeros(3)]

    return tg


def get_interpolation_planner(init_pos, tar_pos, start_time, reach_time):
    interpolation = (tar_pos - init_pos) / (reach_time - start_time)

    def tg(cur_time):
        t = cur_time - start_time
        T = reach_time - start_time
        if t <= T:
            return init_pos + t * interpolation
        else:
            return tar_pos

    return tg


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    path = get_acc_planner(np.array([0, 0, 0]), np.array([30, -10, 20]), 20, 50)
    q = np.array([path(t) for t in range(20, 80)])
    for i in range(3):
        plt.title(['displacement', 'velocity', 'acceleration'][i])
        # plt.plot(range(20, 80), q[:, i])
        plt.plot(range(20, 80), q[:, i, 0])
        plt.plot(range(20, 80), q[:, i, 1])
        plt.plot(range(20, 80), q[:, i, 2])
        plt.show()