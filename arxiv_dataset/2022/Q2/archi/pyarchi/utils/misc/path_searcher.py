import glob
import re
import os 

from pyarchi.utils import create_logger

logger = create_logger("utils")


def path_finder(mode, off_curve=None, **kwargs):
    """
    Searches inside the base folder for the desired files

    Parameters
    ----------
    mode:
        - subarray: retrieves the subarray file path
        - default: default lightcurve
        - stars : path for the Star catalogue file
    kwargs:
        - config values

    Returns
    -------

    """
    searching_for_real = False 
    if kwargs['data_type'] == 'real':
        searching_for_real = True 

    if mode == "subarray":
        
        extension_to_base = r"*eduction*/COR/**" if not searching_for_real else r"**/**"
        possible_paths = glob.glob(os.path.join(kwargs["base_folder"], extension_to_base))
        regex_patern = re.compile(r"\S+_Sub\S+.fits")

    elif mode == "default":
        extension_to_base = r"*eduction*/PHE/**" if not searching_for_real else r"**/**"

        possible_paths = glob.glob(os.path.join(kwargs["base_folder"], extension_to_base))
        curve_2_search = kwargs["official_curve"] if not off_curve else off_curve
        regex_patern = re.compile(r"\S+{}\S+.fits".format(curve_2_search))

    elif mode == "stars":
        extension_to_base = r"CH_*/data/**" if not searching_for_real else r"**/**"

        possible_paths = glob.glob(os.path.join(kwargs["base_folder"],extension_to_base))

        regex_patern = re.compile(r"\S+Star\S+.fits")
        
    try:
        return list(filter(regex_patern.findall, possible_paths))[0]
    except IndexError:
        extra_info = '' if mode != 'default' else curve_2_search
        logger.critical("Failed to find DRP {} output".format(mode +' - '+ extra_info))
        raise Exception("Missing the correct DRP output. Please verify the input folder") 




if __name__ == '__main__':
    print(path_finder('default', **{'data_type':'real', 'base_folder':'/home/amiguel/archi/data_files/cheopsDownload', 'official_curve':'OPTIMAL'}))