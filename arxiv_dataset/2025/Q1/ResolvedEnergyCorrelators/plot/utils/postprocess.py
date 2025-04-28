import numpy as np
from matplotlib.pyplot import Line2D

# Local imports
from plotter import Plotter
from utils.plot_utils import stamp
from utils.gen_utils import str_to_bool


# ====================================
# Code for processing parameters
# ====================================
def partonic_energy(fixed_param_values):
    energy = fixed_param_values.get('energy')
    pid_1 = int(fixed_param_values.get('PID_1'))
    pid_2 = int(fixed_param_values.get('PID_2'))
    for pid in [pid_1, pid_2]:
        if pid == 2212:
            energy /= np.sqrt(13)
    return energy


# ====================================
# Code for processing text files
# ====================================
def read_xy_columns(file_path):
    """
    Reads a file with two columns of data and returns two NumPy arrays:
    x and y.

    Parameters:
    file_path (str): The path to the file containing the data.

    Returns:
    x_array (numpy.ndarray): NumPy array of x values.
    y_array (numpy.ndarray): NumPy array of y values.
    """
    x_values = []
    y_values = []

    # Open the file and read each line
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line into x and y
            x, y = line.split()
            # Convert strings to floats and append to lists
            x_values.append(float(x))
            y_values.append(float(y))

    # Convert lists to NumPy arrays
    x_array = np.array(x_values)
    y_array = np.array(y_values)

    return x_array, y_array

# =====================================
# Code for processing figures
# =====================================
_default_fontsizes = {
    'title': 18,
    'process': 14,
    'stamp': 12,
    'peak': 18,
    'main_legend': 12,
    'line_legend': 9,
    'x_axis': 24,
    'y_axis': 24,
}

_default_locs = {
    'title': (0.5, 0.945),
    'process': (0.02, 0.93),
    'stamp': (0.02, 0.93),
    'peak_text': (0.5, 0.94),
    'main_legend': (0.67, 0.60),
    'line_legend': (.72, 0.45)
}


def collision_stamp(axes, fixed_param_values: dict,
                    show_level: bool = False,
                    fontsize=_default_fontsizes,
                    loc=_default_locs,
                    **kwargs):
    """Adds a stamp to the given axes associated
    with the parameters that are fixed for the plot
    on those axes.
    """
    # - - - - - - - - - - - - - - - - -
    # Stamp configuration
    # - - - - - - - - - - - - - - - - -
    # Basic options for stamps
    if isinstance(loc, dict):
        loc = loc['stamp']
    if isinstance(fontsize, dict):
        fontsize = fontsize['stamp']

    stamp_text_options = {
        'fontsize': fontsize,
        'ha': 'left',
     }

    # Text for the stamp
    stamp_text = {f'line_{i}': '' for i in range(7)}
    line = 0
    line_text = ''

    # - - - - - - - - - - - - - - - - -
    # Extracting relevant parameters
    # - - - - - - - - - - - - - - - - -
    shower_model = fixed_param_values.get('shower_model',
                                          None)
    level = fixed_param_values.get('level', None)
    if shower_model is None and level == 'data':
        shower_model = 'data'

    pid_1 = int(fixed_param_values.get('pid_1', 0))
    pid_2 = int(fixed_param_values.get('pid_2', 0))
    outstate = fixed_param_values.get('outstate_str', 0)
    energy = fixed_param_values.get('energy', None)

    pid_dict = {11: r'e^-', -11: r'e^+',
                2212: 'p', 1000822080: 'Pb',
                0: 'X'}
    pidstr_1, pidstr_2 = pid_dict[pid_1], pid_dict[pid_2]
    is_proton_collision = (int(pid_1) == 2212 and \
                           int(pid_2) == 2212) or \
                          level == 'data'

    outstate_dict = {'top': r'$t\bar{t}$',
                     'w': r'$W^+W^-$',
                     'qcd': 'hadrons',
                     0: 'XX'}
    outstate_str = outstate_dict[outstate.lower()]

    pt_min = fixed_param_values.get('pt_min', None)
    pt_max = fixed_param_values.get('pt_max', None)

    n_events = fixed_param_values.get('n_events', None)
    jet_algorithm = fixed_param_values.get('jet_alg', None)
    subjet_algorithm = fixed_param_values.get('sub_alg',
                                              None)
    if subjet_algorithm:
        subjet_algorithm = subjet_algorithm.lower()
    if jet_algorithm.lower() in ['anti-kt', 'antikt'] and \
            subjet_algorithm is None:
        jet_algorithm = 'AK'

    jet_radius = fixed_param_values.get('jet_rad', None)
    subjet_radius = fixed_param_values.get('sub_rad', None)
    if jet_radius is not None:
        jet_radius = float(jet_radius)
    if subjet_radius is not None:
        subjet_radius = float(subjet_radius)

    isr = fixed_param_values.get('isr', None)
    fsr = fixed_param_values.get('fsr', None)
    mpi = fixed_param_values.get('mpi', None)

    def needs_comma(line_text):
        if line_text != '' and line_text[-2:] != ', ':
            return True
        return False

    # *:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*
    # Shower Model
    # *:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*
    if shower_model is not None:
        line_text = (r"\texttt{Pythia 8.307}"
                    if shower_model == 'pythia'\
                      else (r"\texttt{CMS Open Data}"))
        line_text += ", "
    else:
        line_text = ""

    # *:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*
    # Process
    # *:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*
    # PIDs
    if level != 'data' and (pid_1 != 0 and pid_2 != 0):
        line_text += (r"$" +pidstr_1+pidstr_2 +r"$")
        # Outstates
        if outstate is not None:
            line_text += (" to " + outstate_str)

    if energy is not None:
        if needs_comma(line_text):
            line_text += ", "
        line_text += (r"$\sqrt{s}$="
                      +f"{energy/1000:.1f}"
                      +" TeV")

    # ---------------------------------
    stamp_text[f'line_{line}'] = line_text; line_text = ''; line += 1
    # ---------------------------------

    # ---------------------------------
    # Monte Carlo Information:
    # ---------------------------------
    # Level (Parton, Hadron, Hadron Level + MPI)
    if show_level and level is not None and level != 'data':
        if needs_comma(line_text):
            line_text += ', '
        line_text += f"{level} level ".capitalize()
        if mpi is not None:
            if str_to_bool(mpi):
                line_text += "+ MPI"

    # *:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*
    # Jet Information:
    # *:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*
    # ---------------------------------
    # Jet algorithm string
    # ---------------------------------
    if jet_algorithm is not None:
        if jet_algorithm.startswith('ee_'):
            jet_algorithm = jet_algorithm[3:]
        if jet_algorithm == 'antikt':
            jet_algorithm = 'akt'
        if jet_algorithm.lower() == "ca":
            jet_algorithm = "c/a"
        jet_alg_str = f"{jet_algorithm}".upper()\
                            .replace("_", "-")
    else:
        jet_alg_str = ''

    # Deciding whether to print jet info
    show_jet_info = \
        (jet_algorithm is not None\
        or subjet_algorithm is not None\
        or jet_radius is not None\
        or subjet_radius is not None)

    if show_jet_info:
        if jet_radius == 1000.:
            if line_text == '':
                line_text = "Full event"
            else:
                line_text += ", full event"
        # Ex: AKT8 Jets,
        elif jet_algorithm is not None:
            if needs_comma(line_text):
                line_text += ", "
            line_text += f"{jet_alg_str}"
            if jet_radius is not None:
                line_text += f"{10*jet_radius:.0f}"
            line_text += " Jets"
        else:
            assert jet_radius is not None,\
                "Unsupported use case: "\
                "jet_algorithm and "\
                "jet_radius are both None."

        # Ex: kt4 subjets
        if subjet_algorithm is not None\
                and subjet_radius is not None:
            if needs_comma(line_text):
                line_text += ", "
            # DEBUG: case-by-case soln for now
            if subjet_radius < .1:
                line_text += \
                    fr"$r^{{({subjet_algorithm})}}_{{\rm sub}}$"\
                    +fr"$=\,${subjet_radius:.2f}"
            else:
                line_text += \
                    fr"$r^{{({subjet_algorithm})}}_{{\rm sub}}$"\
                    +fr"$=\,${subjet_radius:.1f}"


    # ---------------------------------
    # p_T Range
    # ---------------------------------
    show_pt_range = False

    if is_proton_collision:
        pt_str = r"$p_{T}$"
    else:
        pt_str = r"$E_{\mathrm{jet}}$"

    if pt_min is not None and pt_min > 0:
        if needs_comma(line_text):
            line_text += ', '
        show_pt_range = True
        line_text += (f"{int(pt_min)} GeV "
                      r"$ < \,$"+pt_str)
    else:
        pt_min = None
    if pt_max is not None and\
            energy is not None\
            and pt_max < energy/2:
        if needs_comma(line_text) and not pt_min:
            line_text += ', '
        show_pt_range = True
        if not pt_min:
            line_text += pt_str
        line_text += (r"$\, < \,$" + f"{pt_max:.1f} GeV")

    # ---------------------------------
    stamp_text[f'line_{line}'] = line_text; line_text = ''; line += 1
    # ---------------------------------

    # *:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*
    # Adding stamp to the axes
    # *:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*:*
    stamp(*loc, axes,
          textops_update=stamp_text_options,
          **stamp_text)
    return
