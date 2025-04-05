"""
#############################################################################################################

Analysis of the latent space


    Alice   2019

#############################################################################################################
"""

import  os
import  sys
import  h5py
import  argparse
import  numpy       as np
from    math        import atan2, sqrt
from    matplotlib  import pyplot

INTERACTIVE     = True

root_dir        = 'latent'

# -----------------------------------------------------------------------------------------------------------

synthia_seqs    = [ 1, 2, 4, 5, 6 ]     # indices of available SYNTHIA sequences
synthia_names   = [                     # all possible names of variations of a SYNTHIA sequence
    # default sequence folder               # my sequence folder        # my prefix for each image file
    ( 'SYNTHIA-SEQS-{:02d}-DAWN',           'S{:02d}_dawn',             'S{:02d}DW_' ),
    ( 'SYNTHIA-SEQS-{:02d}-FALL',           'S{:02d}_fall',             'S{:02d}FL_' ),
    ( 'SYNTHIA-SEQS-{:02d}-FOG',            'S{:02d}_fog',              'S{:02d}FG_' ),
    ( 'SYNTHIA-SEQS-{:02d}-NIGHT',          'S{:02d}_night',            'S{:02d}NT_' ),
    ( 'SYNTHIA-SEQS-{:02d}-RAIN',           'S{:02d}_rain',             'S{:02d}RN_' ),
    ( 'SYNTHIA-SEQS-{:02d}-RAINNIGHT',      'S{:02d}_rainnight',        'S{:02d}RT_' ),
    ( 'SYNTHIA-SEQS-{:02d}-SOFTRAIN',       'S{:02d}_softrain',         'S{:02d}SR_' ),
    ( 'SYNTHIA-SEQS-{:02d}-SPRING',         'S{:02d}_spring',           'S{:02d}SP_' ),
    ( 'SYNTHIA-SEQS-{:02d}-SUMMER',         'S{:02d}_summer',           'S{:02d}SM_' ),
    ( 'SYNTHIA-SEQS-{:02d}-SUNSET',         'S{:02d}_sunset',           'S{:02d}ST_' ),
    ( 'SYNTHIA-SEQS-{:02d}-WINTER',         'S{:02d}_winter',           'S{:02d}WR_' ),
    ( 'SYNTHIA-SEQS-{:02d}-WINTERNIGHT',    'S{:02d}_winternight',      'S{:02d}WN_' )
]

synthia_codes   = [ c[ 2 ][ -3 : ] for c in synthia_names ]


def get_args():
    """ -----------------------------------------------------------------------------------------------------
    Parse the command-line arguments defined by flags.
    rather useless for now, maybe for future expansions
    
    return:         [dict] args (keys) and their values
    ----------------------------------------------------------------------------------------------------- """
    parser      = argparse.ArgumentParser()

    parser.add_argument(
            '-l',
            '--latent',
            action          = 'store',
            dest            = 'LATENT',
            type            = str,
            required        = True,
            default         = None,
            help            = "Pathname of the file with latent vectors"
    )
    return vars( parser.parse_args() )


def read_latent( fname ):
    """ -----------------------------------------------------------------------------------------------------
    read latent vectors from file

    fname:          [str] basename of a latent file

    return:         ( [list] list of keys, [numpy.array] numer of samples x latent size )
    ----------------------------------------------------------------------------------------------------- """
    fn      = os.path.join( root_dir, fname )
    if not os.path.isfile( fn ):
        print( "Error: file {} not found".format( fn ) )
        sys.exit( 0 )
    f       = h5py.File( fn, 'r' )
    k       = sorted( list( f.keys() ) )
    l       = np.array( [ f[ fr ].value for fr in k ] )
    n, _, s = l.shape
    l       = l.reshape( n, s )
    f.close()
    return k, l


def base_stat( latent ):
    """ -----------------------------------------------------------------------------------------------------
    basic statistics on latent vectors
    computes means and std's on each element of the latent vector, and their global statistics

    latent:         [numpy.ndarray] numer of samples x latent size

    return:         [dict] dictionay with statistics
    ----------------------------------------------------------------------------------------------------- """
    me              = latent.mean( axis=0 )
    st              = latent.std( axis=0 )
    stat            = { "means" : me, "stds" : st }
    stat[ "mmean" ] = me.mean()
    stat[ "mstd" ]  = st.mean()
    return stat


def code_stat( fnames, latent ):
    """ -----------------------------------------------------------------------------------------------------
    statistics on latent vectors, separated by synthia_codes
    computes the deviation of the means on a specific codes, with respect to the global mean,
    and the deviation of the std's on a specific codes, with respect to the global std,
    separated for each element of the latent space

    latent:         [numpy.ndarray] lantent size x numer of samples values
    fnames:         [list] list of frame names

    return:         ( [numpy.array] # codes x latent size, [numpy.array] # codes x latent size )
    ----------------------------------------------------------------------------------------------------- """

    b_stat  = base_stat( latent )
    b_mean  = b_stat[ "means" ]
    b_std   = b_stat[ "stds" ]

    n       = len( fnames )
    means   = []
    stds    = []
    for code in synthia_codes:
        ic      = [ i for i in range( n ) if code in fnames[ i ][ : 6 ] ]
        dc      = latent[ ic ]
        sc      = base_stat( dc )
        means.append( ( sc[ "means" ] - b_mean ).clip( 0.0 ) )
        stds.append(  ( sc[ "stds" ]  - b_std  ).clip( 0.0 ) )

    return np.array( means ), np.array( stds )


def seq_stat( fnames, latent ):
    """ -----------------------------------------------------------------------------------------------------
    statistics on latent vectors, separated by synthia sequences
    computes the deviation of the means on a specific sequence, with respect to the global mean,
    and the deviation of the std's on a specific sequence, with respect to the global std,
    separated for each element of the latent space

    latent:         [numpy.ndarray] lantent size x numer of samples values
    fnames:         [list] list of frame names

    return:         ( [numpy.array] # sequences x latent size, [numpy.array] # sequences x latent size )
    ----------------------------------------------------------------------------------------------------- """

    b_stat  = base_stat( latent )
    b_mean  = b_stat[ "means" ]
    b_std   = b_stat[ "stds" ]

    n       = len( fnames )
    means   = []
    stds    = []
    for seq in synthia_seqs:
        seqstr  = "S{:02d}".format( seq )
        ic      = [ i for i in range( n ) if seqstr in fnames[ i ][ : 4 ] ]
        dc      = latent[ ic ]
        sc      = base_stat( dc )
        means.append( ( sc[ "means" ] - b_mean ).clip( 0.0 ) )
        stds.append(  ( sc[ "stds" ]  - b_std  ).clip( 0.0 ) )

    return np.array( means ), np.array( stds )


def plot_code_stat( means, stds ):
    """ -----------------------------------------------------------------------------------------------------
    ----------------------------------------------------------------------------------------------------- """

    codes   = [ c[ : -1 ] for c in synthia_codes ]
    nc      = len( codes )

    pyplot.yticks( 0.5 + np.arange( nc ), codes )
    pyplot.axes().set_aspect( 3.0 )
    pyplot.pcolormesh( means )
    pyplot.savefig( "code_means.pdf" )
    pyplot.close()

    pyplot.yticks( 0.5 + np.arange( nc ), codes )
    pyplot.axes().set_aspect( 3.0 )
    pyplot.pcolormesh( stds )
    pyplot.savefig( "code_stds.pdf" )
    pyplot.close()


def plot_seq_stat( means, stds ):
    """ -----------------------------------------------------------------------------------------------------
    ----------------------------------------------------------------------------------------------------- """

    ns      = len( synthia_seqs )

    pyplot.yticks( 0.5 + np.arange( ns ), synthia_seqs )
    pyplot.axes().set_aspect( 4.0 )
    pyplot.pcolormesh( means )
    pyplot.savefig( "seq_means.pdf" )
    pyplot.close()

    pyplot.yticks( 0.5 + np.arange( ns ), synthia_seqs )
    pyplot.axes().set_aspect( 4.0 )
    pyplot.pcolormesh( stds )
    pyplot.savefig( "seq_stds.pdf" )
    pyplot.close()


# ===========================================================================================================

if __name__ == '__main__':
    if not INTERACTIVE:
        args    = get_args()
        if args[ 'LATENT' ] is not None:
            fnames, latent  = read_latent( args[ 'LATENT' ] )
            means, stds     = code_stat( fnames, latent )
            plot_code_stat( means, stds )
            means, stds     = seq_stat( fnames, latent )
            plot_seq_stat( means, stds )
