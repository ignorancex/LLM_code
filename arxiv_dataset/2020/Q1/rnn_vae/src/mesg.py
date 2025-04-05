"""
#############################################################################################################

Utilities for printing messages

    Alice   2019

#############################################################################################################
"""

import      os
import      sys
import      inspect


def print_err( msg, exit=True ):
    """ -----------------------------------------------------------------------------------------------------
    Print an error messagge in stderr, including the file and line number where the print is executed

    msg:        [str] message to print
    exit:       [bool]
    ----------------------------------------------------------------------------------------------------- """
    LINE    = inspect.currentframe().f_back.f_lineno
    FILE    = os.path.basename( inspect.getfile( inspect.currentframe().f_back ) )

    sys.stderr.write( "ERROR [{}:{}] --> {}\n".format( FILE, LINE, msg ) )

    if exit:
        sys.exit( 1 )



def print_wrn( msg ):
    """ -----------------------------------------------------------------------------------------------------
    Print a warning messagge in stderr, including the file and line number where the print is executed

    msg:        [str] message to print
    ----------------------------------------------------------------------------------------------------- """
    LINE    = inspect.currentframe().f_back.f_lineno
    FILE    = os.path.basename( inspect.getfile( inspect.currentframe().f_back ) )

    sys.stderr.write( "WARNING [{}:{}] --> {}\n".format( FILE, LINE, msg ) )
    sys.stderr.flush()



# TODO check if this still works...
def print_msg( log, msg ):
    """ -----------------------------------------------------------------------------------------------------
    Print a messagge in log file

    log:        [str] path to log file
    msg:        [str] message to print
    ----------------------------------------------------------------------------------------------------- """
    with open( log, 'a' ) as f:
        f.write( msg + '\n' )



def print_line( l=70 ):
    """ -----------------------------------------------------------------------------------------------------
    Print a separation line
    ----------------------------------------------------------------------------------------------------- """
    sys.stdout.write( "\n# " + ( l * '=' ) + " #\n" )
