import glob
from astropy.io import ascii
from astropy.table import vstack
from .utils import get_config


def read_parsec(files=None, path=None, config_file="config.ini"):
    """
    read_parsec(files=None, path=None, config_file="config.ini")

    Read PARSEC isochrone *.dat files, as downloaded from http://stev.oapd.inaf.it/cgi-bin/cmd .

    Parameters
    ----------
    files : list of str, optional
        PARSEC *.dat file names. If None, read all *.dat files in `path` (default: None).
    path : str, optional
        Path to PARSEC *.dat files. If None, get from the configuration file (default: None).
    config_file : str, optional
        The configuration file name, including path (default: "config.ini").

    Returns
    -------
    models : Table
        All PARSEC models in a single astropy table
    """

    if config_file is None:
        if path is None:
            raise Exception("If no config file is given, a path must be set!")
    else:
        config = get_config(config_file)
    if path is None:
        path = config.get('PARSEC', 'PATH')  # PARSEC models path
    print(f"Taking PARSEC files from {path}")

    if files is None:
        # read all .dat files in the path
        files = glob.glob(path + "*.dat")
    else:
        # attach path to filename
        files = [path + file for file in files]

    for i, file in enumerate(files):
        # get the header line number
        header_start = 0
        for line in open(file):
            li = line.strip()
            if li.startswith("#"):
                header_start += 1
            else:
                break

        iso = ascii.read(file, header_start=header_start)
        if i == 0:
            models = iso
        else:
            models = vstack([models, iso])

    return models


def colname(param, model_name="parsec"):
    """
    colname(param, model_name="parsec")

    Return a specific model table column name, according to the model table type.

    Parameters
    ----------
    param : str
        Parameter name.
    model_name : str, optional
        Model table type (currently only supports PARSEC)

    Returns
    -------
    str
        The relevant column name.
    """

    if model_name.lower() == "parsec":
        # Zini     MH   logAge Mini  int_IMF         Mass   logL    logTe  logg  label   McoreTP C_O  period0 period1
        # pmode  Mloss  tau1m   X   Y   Xc  Xn  Xo  Cexcess  Z 	 mbolmag  Gmag    G_BPmag  G_RPmag
        cols = {
            "Z0": "Zini",
            "mh": "MH",
            "log_age": "logAge",
            "m0": "Mini",
            "m": "Mass",
            "phase": "label",
            "Gmag": "Gmag",
            "G_BPmag": "G_BPmag",
            "G_RPmag": "G_RPmag",
            "B": "Bmag",
            "V": "Vmag",
            "I": "Imag"
        }
    else:
        raise Exception(f"Unknown model {model_name}!")

    if param in cols:
        return cols[param]
    else:
        return param
