import string
from typing import List, Tuple


def LinearizedTwoValues(val_1: float, val_2: float, vec_len: int) -> List[float]:
    """create a list of values according to a given length where the values go up 
    with the same step size from one given value to another.

    Args:
        val_1 (float): initial value
        val_2 (float): final value
        vec_len (int): length of new vector

    Returns:
        List[float]: A linear list of values
    """
    vector_values = []
    diff = (val_2 - val_1)/vec_len

    for i in range(vec_len):
        val_new = val_1 + i * diff
        vector_values.append(val_new)

    return vector_values


def ExpandPath(path):
    path_new = []
    for dot in path:
        x_val = (dot[0] + 0.5) * 20
        y_val = (dot[1] + 1) * 20
        new_dot = [x_val, y_val]
        path_new.append(new_dot)
    return path_new


def folderName2Sections(folder_name: string):
    if folder_name == "A_before" or folder_name == "csv_correction":
        sections = ["A"]
    elif folder_name == "A_after":
        sections = ["B", "C", "D", "E"]
    elif folder_name == "B_before":
        sections = ["A", "B"]
    elif folder_name == "B_after":
        sections = ["C", "D", "E"]
    elif folder_name == "C_before":
        sections = ["A", "B", "C"]
    elif folder_name == "C_after":
        sections = ["D", "E"]
    elif folder_name == "D_before":
        sections = ["A", "B", "C", "D"]
    elif folder_name == "D_after":
        sections = ["E"]
    elif folder_name == "full_path":
        sections = ["A", "B", "C", "D", "E"]

    return sections


def linearExperiments(val_1: float, val_2: float, resolution: float, section: string) -> List[float]:
    """Create a linear list of values for the waypoints toward the goal.

    Args:
        val_1 (float): initial value
        val_2 (float): final value
        vec_len (int): length of new vector
        section (string): section of the path of the real world experiments

    Returns:
        List[float]: A linear list of values
    """
    if section in {"C", "E"}:
        vec_len = round(resolution*1.5)
    else:
        vec_len = resolution

    return LinearizedTwoValues(val_1, val_2, vec_len)


def AECDDataLinear(path_exe_waypoints, file_name: string):
    path_exe = ExpandPath(path_exe_waypoints)
    resolution = 20  # TODO: change to true res
    sections = folderName2Sections(file_name)

    path_exe_linear = []
    for i in range(2):
        vec = []
        for j in range(len(sections)):
            vec = vec + linearExperiments(path_exe[j][i], path_exe[j+1][i],
                                          resolution, sections[j])
        path_exe_linear.append(vec)

    return path_exe_linear[0], path_exe_linear[1]


def VOADataExpanded(x_mean: List[float], y_mean: List[float], x_var: List[float], y_var: List[float]):
    x_new = []
    for x in x_mean:
        x_val = (x + 0.5) * 20
        x_new.append(x_val)
    x_mean = x_new

    y_new = []
    for y in y_mean:
        y_val = (y + 1) * 20
        y_new.append(y_val)
    y_mean = y_new

    x_var_new = []
    for x in x_var:
        x_val = x * 20
        x_var_new.append(x_val)
    x_var = x_var_new

    y_var_new = []
    for y in y_var:
        y_val = y * 20
        y_var_new.append(y_val)
    y_var = y_var_new

    return x_mean, y_mean, x_var, y_var


def VOADataExtractHelpAtI(i: int):
    x_mean, y_mean, x_var, y_var = VOADataRaw()
    x_mean, y_mean, x_var, y_var = VOADataExpanded(
        x_mean, y_mean, x_var, y_var)

    x_mean, y_mean, x_var_new, y_var_new = VOADataLinear(
        x_mean, y_mean, x_var, y_var)

    x_var_correction = x_var[i]
    y_var_correction = y_var[i]
    index_correction = x_var_new.index(x_var_correction)
    for j in range(len(x_var_new) - index_correction):
        x_var_new[j + index_correction] -= x_var_correction
        y_var_new[j + index_correction] -= y_var_correction

    return x_mean, y_mean, x_var_new, y_var_new


def VOADataLinear(x_mean: List[float], y_mean: List[float], x_var: List[float], y_var: List[float]):
    resolution = 20
    values = [x_mean, y_mean, x_var, y_var]

    values_linear = []
    for vec_value in values:
        vec_val_A = linearExperiments(
            vec_value[0], vec_value[1], resolution, "A")
        vec_val_B = linearExperiments(
            vec_value[1], vec_value[2], resolution, "B")
        vec_val_C = linearExperiments(
            vec_value[2], vec_value[3], resolution, "C")
        vec_val_D = linearExperiments(
            vec_value[3], vec_value[4], resolution, "D")
        vec_val_E = linearExperiments(
            vec_value[4], vec_value[5], resolution, "E")
        values_linear.append(vec_val_A + vec_val_B +
                             vec_val_C + vec_val_D + vec_val_E)

    return values_linear[0],  values_linear[1],  values_linear[2],  values_linear[3]


def VOADataRaw() -> Tuple[List[float], List[float], List[float], List[float]]:
    """Data to calculate the VOA according to the location uncertainty of the agent

    Returns:
        Tuple[List[float], List[float], List[float], List[float]]: 
           x_mean - location mean over the X axis
           y_mean - location mean over the Y axis
           x_sigma - location sigma over the X axis
           y_sigma - location sigma over the Y axis 
    """
    x_mean = [-0.028375, 0.954354167, 1.5180625,
              2.389979167, 3.347729167, 4.1971875]
    y_mean = [0.000333333, 0.001625, 0.7801875, -
              0.376895833, -0.3968125, 0.781916667]

    x_sigma = [0.000855785, 0.009477892, 0.022112197,
               0.03022457, 0.028011764, 0.036931409]
    y_sigma = [0.000465215, 0.014023169, 0.022906405,
               0.045606768, 0.071284017, 0.094513136]

    return x_mean, y_mean, x_sigma, y_sigma
