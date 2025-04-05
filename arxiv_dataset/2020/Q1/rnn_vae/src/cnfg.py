"""
#############################################################################################################

Utilities for handling command-line arguments, and for parsing configuration files.

The config files should be textual, in which each line can contain:
    - a comment, starting with char '#'
    - an empty line
    - the name of the variable and its value, separated with tab or spaces

The content of config files should be fed to the config dictionaries in the main python scripts.

    Alice   2019

#############################################################################################################
"""

import  os
import  sys
import  argparse

import  mesg            as ms
from    arch            import layer_code


DEBUG0  = False         # enable debugging print
DEBUG1  = False


def get_args():
    """ -----------------------------------------------------------------------------------------------------
    Parse the command-line arguments defined by flags
    
    return:         [dict] args (keys) and their values
    ----------------------------------------------------------------------------------------------------- """
    parser      = argparse.ArgumentParser()

    parser.add_argument(
            '-a',
            '--accur',
            action          = 'count',
            dest            = 'ACCUR',
            default         = 0,
            help            = "Execute accuracy evaluation on selected samples [-a] or on all test set \
                                    (WARNING: may take a while)[-aa]"
    )
    parser.add_argument(
            '-c',
            '--config',
            action          = 'store',
            dest            = 'CONFIG',
            type            = str,
            required        = True,
            help            = "Config file describing the model architecture and training parameters"
    )
    parser.add_argument(
            '-e',
            '--eval',
            action          = 'store_true',
            dest            = 'EVAL',
            help            = "Execute evaluation routines"
    )
    parser.add_argument(
            '-f',
            '--fgpu',
            action          = 'store',
            dest            = 'FGPU',
            type            = float,
            default         = 0.90,
            help            = "Fraction of GPU memory to allocate"
    )
    parser.add_argument(
            '-g',
            '--gpu',
            action          = 'store',
            dest            = 'GPU',
            required        = True,
            help            = "Number of GPUs to use (0 if CPU) or list of GPU indices"
    )
    parser.add_argument(
            '-i',
            '--intrp',
            action          = 'store_true',
            dest            = 'INTRP',
            help            = "Interpolation between two frames"
    )
    parser.add_argument(
            '-l',
            '--load',
            action          = 'store',
            dest            = 'LOAD',
            type            = str,
            default         = None,
            help            = "Folder or HDF5 file to load as weights or entire model"
    )
    parser.add_argument(
            '-p',
            '--pred',
            action          = 'store_true',
            dest            = 'PRED',
            help            = "Compute predictions over a selected set of images"
    )
    parser.add_argument(
            '-r',
            '--redir',
            action          = 'store_true',
            dest            = 'REDIRECT',
            help            = "Redirect stderr and stdout to log file"
    )
    parser.add_argument(
            '-s',
            '--save',
            action          = 'count',
            dest            = 'ARCHIVE',
            default         = 0,
            help            = "Archive config files [-s] and python scripts [-ss]"
    )
    parser.add_argument(
            '-T',
            '--train',
            action          = 'store_true',
            dest            = 'TRAIN',
            help            = "Execute training of the model"
    )
    parser.add_argument(
            '-t',
            '--test',
            action          = 'store_true',
            dest            = 'TEST',
            help            = "Execute testing routines"
    )
    parser.add_argument(
            '-x',
            '--hallx',
            action          = 'store_true',
            dest            = 'HALLX',
            help            = "Execute hallucination routine"
    )

    return vars( parser.parse_args() )



def get_args_eval():
    """ -----------------------------------------------------------------------------------------------------
    Parse the command-line arguments defined by flags.
    This version is created for exec_eval.py
    
    return:         [dict] args (keys) and their values
    ----------------------------------------------------------------------------------------------------- """
    parser_e    = argparse.ArgumentParser()

    parser_e.add_argument(
            '-m',
            '--model',
            action          = 'store',
            dest            = 'MODEL',
            type            = str,
            required        = False,
            default         = None,
            help            = "Pathname of folder containing the model result"
    )
    parser_e.add_argument(
            '-l',
            '--list',
            action          = 'store',
            dest            = 'MODELS',
            type            = str,
            required        = False,
            default         = None,
            help            = "List of pathnames of several folders of models result"
    )
    parser_e.add_argument(
            '-i',
            '--img',
            action          = 'store',
            dest            = 'IMAGE',
            type            = str,
            required        = False,
            default         = None,
            help            = "Pathname of image file to use as input for prediction"
    )
    parser_e.add_argument(
            '-s',
            '--seqs',
            action          = 'store',
            dest            = 'IMAGES',
            type            = str,
            required        = False,
            default         = None,
            help            = "List of pathnames of image folders image to use as input for prediction"
    )

    return vars( parser_e.parse_args() )



def get_config( fname ):
    """ -----------------------------------------------------------------------------------------------------
    Return the content of a config file in the form of a dictionary

    fname:          [str] path of config file

    return:         [dict] content of the file
    ----------------------------------------------------------------------------------------------------- """
    cnfg    = dict()

    if not os.path.isfile( fname ):
        ms.print_err( "Configuration file \"{}\" not found.".format( fname ) )

    if DEBUG0:
        ms.print_msg( cnfg[ 'log_msg' ], "Reading configuration file \"{}\".\n".format( fname ) )
        os.system( "cat %s" % fname )

    with open( fname ) as doc:
        for line in doc:
            if line[ 0 ] == '#': continue   # comment line

            if DEBUG1:
                print( line )

            c   = line.split( '#' )[ 0 ]    # remove any following comment
            c   = c.split()
            if len( c ) == 0: continue      # empty line

            cnfg[ c[ 0 ] ] = eval( str().join( c[ 1: ] ) )

    return cnfg



def load_config( cnfg, dest ):
    """ -----------------------------------------------------------------------------------------------------
    Use the first dict to fill the value of the second dict, in case of common keys

    cnfg:           [dict] one with all configs
    dest:           [dict] one to be filled
    ----------------------------------------------------------------------------------------------------- """
    for k in dest.keys():
        if k in cnfg.keys():
            dest[ k ]   = cnfg[ k ]
