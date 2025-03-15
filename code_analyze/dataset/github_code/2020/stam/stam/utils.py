from configparser import ConfigParser
import logging
import os


def get_config(config_file="config.ini"):
    """
    get_config(config_file="config.ini")

    Read configuration file.

    Parameters
    ----------
    config_file : str, optional
        The configuration file name, including path (default: "config.ini").

    Returns
    -------
    config : A ConfigParser object.
    """

    config = ConfigParser(inline_comment_prefixes=';')
    config.read(config_file)
    return config


def init_log(filename="log.log", log_path=None, console_log_level="DEBUG", file_log_level="DEBUG",
             config_file="config.ini"):
    """
    init_log(filename="log.log", config_file="config.ini")

    Initialize logger.

    Parameters
    ----------
    filename : str, optional
        Log file name.
    log_path : str, optional
        Log file path (default: None). If neither `config_file` nor `log_path` are provided, don't save log to file.
    console_log_level : str, optional
        Console log level (DEBUG, INFO, WARNING, ERROR, CRITICAL; default: "DEBUG"). Only used if no `config_file` is provided.
    file_log_level : str, optional
        Console log level (DEBUG, INFO, WARNING, ERROR, CRITICAL; default: "DEBUG"). Only used if no `config_file` is provided.
    config_file : str, optional
        The configuration file name, including path (default: "config.ini"). If neither `config_file` nor `log_path` are provided, don't save log to file.

    Returns
    -------
    log : Logger
        Logger object.
    """

    write_logfile = True
    if config_file is None:
        if log_path is None:
            write_logfile = False
    else:
        config = get_config(config_file)
        log_path = config.get('LOG', 'PATH')  # log file path
        console_log_level = config.get('LOG', 'CONSOLE_LEVEL')  # logging level
        file_log_level = config.get('LOG', 'FILE_LEVEL')  # logging level

    if write_logfile:
        # create log folder
        if not os.path.exists(log_path):
            os.makedirs(log_path)

    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    # console handler
    h = logging.StreamHandler()
    h.setLevel(logging.getLevelName(console_log_level))
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    h.setFormatter(formatter)
    log.addHandler(h)

    # log file handler
    if write_logfile:
        h = logging.FileHandler(log_path + filename + ".log", "w", encoding=None, delay=True)
        h.setLevel(logging.getLevelName(file_log_level))
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s [%(filename)s:%(lineno)s]: %(message)s", "%Y-%m-%d %H:%M:%S")
        h.setFormatter(formatter)
        log.addHandler(h)

    return log


def close_log(log):
    """
    close_log(log)

    Close logger.

    Parameters
    ----------
    log : Logger object

    Returns
    -------

    """
    handlers = list(log.handlers)
    for h in handlers:
        log.removeHandler(h)
        h.flush()
        h.close()
    return


def write_config(config_file="config.ini",
                 log_path="./log/", log_console_level="INFO", log_file_level="DEBUG",
                 general_save="TRUE", general_path=".", general_output_type="csv", general_csv_format=".8f",
                 gaia_path="./Gaia/", gaia_file="Gaia.fits", gaia_correct_extinction="TRUE",
                 binary_consider_twins="TRUE", binary_flux_ratio_min=1.9, binary_flux_ratio_max=2,
                 models_source="PARSEC", models_m_min=0.15, models_m_max=1.05, models_m_step=0.05, models_age=5,
                 models_mh_pre_ms=0.7, models_smooth="True", models_smooth_sigma=3, models_exclude_pre_ms_masses=[],
                 parsec_path="./PARSEC/",
                 interp_method="rbf", interp_rbf_fun="linear",
                 mass_n_realizations=10, mh_n_realizations=10):
    """
    write_config(config_file="config.ini",
                 log_path="./log/", log_console_level="INFO", log_file_level="DEBUG",
                 general_save="TRUE", general_path=".", general_output_type="csv", general_csv_format=".8f",
                 gaia_path="./Gaia/", gaia_file="Gaia.fits", gaia_correct_extinction="TRUE",
                 binary_consider_twins="TRUE", binary_flux_ratio_min=1.9, binary_flux_ratio_max=2,
                 models_source="PARSEC", models_m_min=0.15, models_m_max=1.05, models_m_step=0.05, models_age=5,
                 models_mh_pre_ms=0.7, models_smooth="True", models_smooth_sigma=3, models_exclude_pre_ms_masses=[],
                 parsec_path="./PARSEC/",
                 interp_method="rbf", interp_rbf_fun="linear",
                 mass_n_realizations=10, mh_n_realizations=10)

    Write config file.

    Parameters
    ----------
    config_file : str, optional
        The configuration file name, including path (default: "config.ini").
    log_path : str, optional
        Log file path (default: "./log/").
    log_console_level : str, optional
        Console log level (DEBUG, INFO, WARNING, ERROR, CRITICAL; default: "INFO").
    log_file_level : str, optional
        File log level (DEBUG, INFO, WARNING, ERROR, CRITICAL; default: "DEBUG").
    general_save : str, optional
        save results to file? (default: "TRUE").
    general_path : str, optional
        Result file destination path (default: ".").
    general_output_type : str, optional
        Output type (npy, csv; default: "csv").
    general_csv_format : str, optional
        CSV format (without "%"; default: ".8f").
    gaia_path : str, optional
        Path to Gaia data folder (default: "./Gaia/").
    gaia_file : str, optional
        Gaia data file name (only works for FITS files; default: "Gaia.fits").
    gaia_correct_extinction : str, optional
        Correct Gaia data for extinction? (default: "TRUE").
    binary_consider_twins : str, optional
        Consider equal-mass binary sequence when assigning mass/metallicity/age? (default: "TRUE").
    binary_flux_ratio_min : float, optional
        Minimal binary twin flux ratio (default: 1.9).
    binary_flux_ratio_max : float, optional
        Maximal binary twin flux ratio (default: 2).
    models_source : str, optional
        Which isochrone models to use? (default: "PARSEC").
    models_m_min : float, optional
        Minimum track mass in Msun (default: 0.15).
    models_m_max : float, optional
        Maximum track mass in Msun (default: 1.05).
    models_m_step : float, optional
        Track mass step in Msun (default: 0.05).
    models_age : float, optional
        Age of the MS tracks in Gyr (default: 5).
    models_mh_pre_ms : float, optional
        Pre-MS track [M/H] metallicity (default: 0.7).
    models_smooth : str, optional
        Smooth track? (default: "TRUE").
    models_smooth_sigma : float, optional
        Gaussian smoothing sigma (default: 3).
    models_exclude_pre_ms_masses : list, optional
        Exclude the pre-MS tracks of these masses in Msun (to avoid crossing other tracks), leave empty to include all (default: []).
    parsec_path : str, optional
        Path to the PARSEC *.dat tables (concatenates all *.dat files in the folder to a single table) (default: "./PARSEC/").
    interp_method : str, optional
        Interpolation method (rbf, griddata, nurbs; default: "rbf").
    interp_rbf_fun : str, optional
        SciPy's RBF interpolation function (default: "linear").
    mass_n_realizations : float, optional
        Number of realizations for mass assignment (default: 10).
    mh_n_realizations : float, optional
        Number of realizations for metallicity assignment (default: 10).

    Returns
    -------

    """

    config = f"""
; config.ini
[LOG]
PATH = {log_path}
CONSOLE_LEVEL = {log_console_level} ; DEBUG, INFO, WARNING, ERROR, CRITICAL
FILE_LEVEL = {log_file_level} ; DEBUG, INFO, WARNING, ERROR, CRITICAL

[GENERAL]
SAVE = {general_save} ; save results to file?
PATH = {general_path} ; result file destination path
OUTPUT_TYPE = {general_output_type} ; npy, csv
CSV_FORMAT = {general_csv_format} ; CSV format (without "%")

[GAIA]
PATH = {gaia_path} ; path to Gaia data folder
FILE = {gaia_file} ; Gaia data file name (only works for FITS files)
CORRECT_EXTINCTION = {gaia_correct_extinction} ; correct Gaia data for extinction?

[BINARY]
CONSIDER_TWINS = {binary_consider_twins} ; consider equal-mass binary sequence when assigning mass/metallicity/age?
FLUX_RATIO_MIN = {binary_flux_ratio_min} ; minimal binary twin flux ratio
FLUX_RATIO_MAX = {binary_flux_ratio_max} ; maximal binary twin flux ratio

[MODELS]
SOURCE = {models_source} ; which isochrone models to use?
M_MIN = {models_m_min} ; [Msun] minimum track mass
M_MAX = {models_m_max} ; [Msun] maximum track mass
M_STEP = {models_m_step} ; [Msun] track mass step
AGE = {models_age} ; [Gyr] age of the MS tracks
MH_PRE_MS = {models_mh_pre_ms} ; pre-MS track [M/H]
SMOOTH = {models_smooth} ; smooth track?
SMOOTH_SIGMA = {models_smooth_sigma} ; Gaussian smoothing sigma
EXCLUDE_PRE_MS_MASSES = {', '.join(str(x) for x in models_exclude_pre_ms_masses)} ; [Msun] exclude the pre-MS tracks of these masses (to avoid crossing other tracks),
                                                       ; separate values by a comma, leave empty to include all

[PARSEC]
PATH = {parsec_path} ; path to the PARSEC *.dat tables (concatenates all *.dat files in the folder to a single table)

[INTERP]
METHOD = {interp_method} ; rbf, griddata, nurbs
RBF_FUN = {interp_rbf_fun} ; linear, cubic, ...

[MASS]
N_REALIZATIONS = {mass_n_realizations} ; number of realizations

[MH]
N_REALIZATIONS = {mh_n_realizations} ; number of realizations
    """

    with open(config_file, "w") as file:
        print(config, file=file)

    return
