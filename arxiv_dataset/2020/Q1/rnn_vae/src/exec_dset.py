"""
#############################################################################################################

Utilities for handling the dataset image folders, by creating symbolic links

The Keras preprocessing.image.ImageDataGenerator methods require a very complex folder structure,
depending also from the type of model architecture. That is why is more convinient to keep a single
folder with all the actual frames, and then to generate the specific folders with symbolic links of
selected frames.

The structure should be something like:

    -------------------------------------------------------------------------------
    dataset_name/               main folder

        dset_noclass/           dataset for NN where input and target are the same
            train/              training set
                img/            linked images
            valid/              validation set
                img/            linked images
            test/               test set
                img/            linked images


        dset_class/             dataset for NN where input and target are different

            train/              training set
                class_1/        images of first class (i.e. input)
                    img/        linked images
                class_2/        images of second class (i.e. target)
                    img/        linked images

            valid/              validation set
                class_1/        images of first class (i.e. input)
                    img/        linked images
                class_2/        images of second class (i.e. target)
                    img/        linked images

            test/               test set
                class_1/        images of first class (i.e. input)
                    img/        linked images
                class_2/        images of second class (i.e. target)
                    img/        linked images


        dset_multi/             dataset for NN where input is RBG and there are multiple targets

            train/              training set
                rgb/            input images as well as one of the target
                    img/        linked images
                class_1/        images used as second target
                    img/        linked images
                class_2/        images used as third target
                    img/        linked images

            valid/              validation set
                same substracture as train

            test/               test set
                same substracture as train


        dset_seq_noclass/       dataset for recursive NN where input and target are the same

            train/              training set
                f0/             images of first time step
                    img/        linked images
                f1/             images of second time step
                    img/        linked images
                f2/             images of third time step
                    img/        linked images
                ...

            valid/              validation set
                same substracture as train

            test/               test set
                same substracture as train


        dset_seq_multi/         dataset for recursive NN where there are multiple targets

            train/              training set
                f0/             images of first time step
                    rgb/        input images as well as one of the target
                        img/    linked images
                    class_1/    images used as second target
                        img/    linked images
                    class_2/    images used as third target
                        img/    linked images
                f1/             images of second time step
                    rgb/        input images as well as one of the target
                        img/    linked images
                    class_1/    images used as second target
                        img/    linked images
                    class_2/    images used as third target
                        img/    linked images
                ...

            valid/              validation set
                same substracture as train

            test/               test set
                same substracture as train

    -------------------------------------------------------------------------------

    Alice   2019

#############################################################################################################
"""

import  os
import  sys
import  shutil
import  random
import  png
import  h5py
import  deepdish
import  numpy   as np
from    PIL         import Image
from    math        import atan2, sqrt

from    sample_sel  import  sample_sel


PLAIN_LINK_SYNTHIA  = True
SCALE_SYNTHIA       = False 
LINK_SYNTHIA        = False
ODOM_SYNTHIA        = False
TIME_SYNTHIA        = False

NORM_ODOM       = False

DEBUG           = True
SEED            = 3

frac_train      = 0.70                                  # fraction of training set
frac_valid      = 0.25                                  # fraction of validation set
frac_test       = 0.05                                  # fraction of test set
batch_msize     = 256                                   # NOTE max batch size that will be used during training

odo_step        = [ 1, 2, 4 ]                           # step used to compute odometry
latent_size     = 128

file_ext        = ( '.jpg', '.jpeg', '.png' )           # accepted image file formats
cond_ext        = lambda x: x.endswith( file_ext )      # condition selecting only image files

root_dir            = 'dataset'
dset_alice          = os.path.join( root_dir, 'alice' )
dset_synthia        = os.path.join( root_dir, 'synthia' )
synthia_orig        = os.path.join( dset_synthia, 'orig' )
synthia_orig_neat   = os.path.join( dset_synthia, 'orig_neat' )
synthia_scaled      = os.path.join( dset_synthia, 'scaled_256_128' )
#synthia_scaled     = os.path.join( dset_synthia, 'scaled_1024_512' )
synthia_link        = os.path.join( dset_synthia, 'link' )
synthia_latent      = os.path.join( dset_synthia, 'latent' )
synthia_time        = os.path.join( dset_synthia, 'time' )
synthia_odom        = os.path.join( dset_synthia, 'odom' )
name_odom           = 'step_{:02d}.h5'                      # odometry file (speed, steer rate)
name_seq_dir        = 'f{}'                                 # base format of folder names for sequences

warn            = "Directory {} does not exist. Don't worry, I'm skipping to the next one..."

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

class_synthia   = {                     # see README.txt of SYNTHIA
        'VOID'          : 0,
        'SKY'           : 1,
        'BUILDING'      : 2,
        'ROAD'          : 3,
        'SIDEWALK'      : 4,
        'FENCE'         : 5,
        'VEGETATION'    : 6,
        'POLE'          : 7,
        'CAR'           : 8,
        'SIGN'          : 9,
        'PEDESTRIAN'    : 10,
        'BICYLE'        : 11,
        'LANE'          : 12,
        'TLIGHT'        : 15
}



# ===========================================================================================================
#
#   Function for composing the test set made of manually selected samples
#   the manually selected samples are those currently written in sample_sel.py
#
#   - sel_test
#   - dset_sel
#
# ===========================================================================================================

def sel_test( suffix='_RGB_FL' ):
    """ -----------------------------------------------------------------------------------------------------
    The function generates the strings identifying the first frame in a sample of the manual
    selection in sample_sel, and associates this string with the category of event of the sample

    suffix:         [str] final part of the string identifying a frame

    return:         [tuple] of a dictionary with frame strings as key and category as values,
                            the list of all categories
    ----------------------------------------------------------------------------------------------------- """
    stest       = {}
    categories  = []

    for s in synthia_seqs:
        for n in synthia_names:
            seq     = n[ 0 ].format( s )
            if seq not in sample_sel:
                continue
            prfx        = n[ 2 ].format( s )
            for f, c in sample_sel[ seq ]:
                stest[ prfx + f + suffix ]  = c
                categories.append( c )

    return stest, list( set( categories ) )



def dset_sel( stest, n_inp, n_out, step=1 ):
    """ -----------------------------------------------------------------------------------------------------
    Generate a dataset of string names for temporal prediction of latent spaces on the selected samples
    The structure is compatible with the string component returned by dset_time(), see comments there

    stest:          [dic] dictionary with frame strings as key, as returned by sel_test()
    n_inp:          [int] number of input frames
    n_out:          [int] number of output frames
    step:           [int] incremental step between consecutive frames

    return:         [tuple]     xs:     [list of numpy.array] string names of input images
                                ys:     [list of numpy.array] string names of target images
    ----------------------------------------------------------------------------------------------------- """

    xs  = [ [] for i in range( n_inp ) ]    # support list for saving the string name of frames
    ys  = [ [] for i in range( n_out ) ]    # support list for saving the string name of frames


    # fill the lists in the right order
    for k in sorted( stest ):
        f       = k                                     # initial frame code
        for i in range( n_inp ):
            xs[ i ].append( f )                         # save string name
            f   = next_timeframe( f, step )

        for i in range( n_out ):
            ys[ i ].append( f )                         # save string name
            f   = next_timeframe( f, step )

    xs  = [ np.array( l ) for l in xs ]
    ys  = [ np.array( l ) for l in ys ]

    return xs, ys




# ===========================================================================================================
#
#   Function for extracting odometry from SYNTHIA dataset
#
#   - next_file
#   - read_pose
#   - get_pitch
#   - get_pos
#   - get_odom
#   - dict_odom
#
# ===========================================================================================================

def next_file( frame, step=1 ):
    """ -----------------------------------------------------------------------------------------------------
    The function returns the pathname of a successor file of the current frame file

    frame:          [str] pathname of current frame file
    step:           [int] step to the successor frame

    return:         [str] pathname of the next frame file
    ----------------------------------------------------------------------------------------------------- """
    dr          = os.path.dirname( frame )
    frame_num   = os.path.splitext( os.path.basename( frame ) )[ 0 ]        # frame index
    frmt        = '{:0' + str( len( frame_num ) ) + '}.txt'                 # filename format

    return os.path.join( dr, frmt.format( int( frame_num ) + step ) )



def read_pose( frame ):
    """ -----------------------------------------------------------------------------------------------------
    This function is used to parse the text files contained in the CameraParams folder of each sequence of
    the SYNTHIA dataset.

    See 'get_odom()' for description of data format of the files.

    frame:          [str] pathname of a frame file

    return:         [numpy.array] the 4x4 pose matrix
    ----------------------------------------------------------------------------------------------------- """
    data    = open( frame, 'r' ).read().split()
    m       = np.array( data, dtype=float )
    m       = np.reshape( m, ( 4, 4 ) )
    m       = np.swapaxes( m, 0, 1 )
    return m



def get_pitch( m ):
    """ -----------------------------------------------------------------------------------------------------
    Compute the pitch angle (radians) from a pose matrix.
    The pitch is considered because the axis pointing up is the Y.

    Uses equation (3.48) p.99 in LaValle, 2006, "Planning Algorithms", Cambridge University Press

    m:              [numpy.array] 4x4 pose matrix

    return:         [float] radians
    ----------------------------------------------------------------------------------------------------- """
    r_31    = m[ 2, 0 ]
    r_32    = m[ 2, 1 ]
    r_33    = m[ 2, 2 ]
    return atan2( sqrt( r_32**2 + r_33**2 ), - r_31 )



def get_pos( m ):
    """ -----------------------------------------------------------------------------------------------------
    Retrieve the ego position from a pose matrix

    m:              [numpy.array] 4x4 pose matrix

    return:         [numpy.array] 3 vector
    ----------------------------------------------------------------------------------------------------- """
    return m[ :-1 , 3 ]



def get_odom( frame, step=1, fps=5 ):
    """ -----------------------------------------------------------------------------------------------------
    Compute the ego speed and steering rate related to a frame and its successor, for the SYNTHIA dataset.
    The steering rate is positive for counter-clockwise rotation along the Y axis
    (i.e. when steering the car left).

    SYNTHIA odometry files (one for each frame) contain 16 values, which made up the following matrix
    (by columns):

        |  r11 r12  r13 X |
        |  r21 r22  r23 Y |
        |  r31 r32  r33 Z |
        |  0   0    0   1 |

    where X, Y, Z are absolute coordinates, initially Z typically points in the initial direction of the car,
    Y points up, and X is horizontal (ortogonal to Z);
    rXX are components of the rotation matrix defining car headings with respect to the absolute reference.

    Note that the computation of speed and steering rate requires two frames.

    Note that the SYNTHIA dataset is generated using 5 FPS.

    frame:          [str] pathname of current frame file
    step:           [int] step between frames
    fps:            [int] frame per second

    return:         [tuple] ego speed (m/s) and steering angle rate (rad/s)
    ----------------------------------------------------------------------------------------------------- """
    if not os.path.exists( frame ):
        print( "Error: file {} not found".format( frame ) )
        sys.exit( 1 )

    # get the pathname of the next frame
    fnext   = next_file( frame, step=step )

    # and check if existing
    if not os.path.exists( fnext ):
        print( "Warning: file {} not found".format( fnext ) )
        return False

    # read pose matrices for each frame
    m1      = read_pose( frame )
    m2      = read_pose( fnext )

    # compute speed
    p1      = get_pos( m1 )
    p2      = get_pos( m2 )
    speed   = fps * np.linalg.norm( p2 - p1 ) / step

    # compute steering angle rate
    steer   = fps * ( get_pitch( m2 ) - get_pitch( m1 ) ) / step

    return speed, steer



def dict_odom( step=1 ):
    """ -----------------------------------------------------------------------------------------------------
    Generate a dictionary with odometry data (speed, steer) for all frames in the SYNTHIA dataset

    step:           [int] step between frames

    return:         [dict] keys are filenames (without extension), values are tuple of (speed, steer)
    ----------------------------------------------------------------------------------------------------- """
    odom    = {}
    tot     = 0

    for s in synthia_seqs:
        for n in synthia_names:
            src_dir     = os.path.join( synthia_orig, n[ 0 ].format( s ) )
            if not os.path.isdir( src_dir ):
                print( warn.format( src_dir ) )
                continue

            dir_odom_L  = os.path.join( src_dir, 'CameraParams/Stereo_Left/Omni_F' )
            dir_odom_R  = os.path.join( src_dir, 'CameraParams/Stereo_Right/Omni_F' )
            prfx        = n[ 2 ].format( s )
            files       = [ f for f in sorted( os.listdir( dir_odom_L ) ) if f.endswith( '.txt' ) ]

            cnt         = len( files )
            tot         += 2 * cnt
            if DEBUG: print( "Doing odometry for dataset {} with {:d} files".format( src_dir, cnt ) )

            for f in files:
                f_name          = f.split( '.' )[ 0 ]

                key             = prfx + f_name + '_RGB_FL'
                sp_st           = get_odom( os.path.join( dir_odom_L, f ), step=step )
                if sp_st:
                    odom[ key ] = sp_st

                key             = prfx + f_name + '_RGB_FR'
                sp_st           = get_odom( os.path.join( dir_odom_R, f ), step=step )
                if sp_st:
                    odom[ key ] = sp_st

    if DEBUG: print( "{:d} odometry files done".format( tot ) )
 
    return odom



# ===========================================================================================================
#
#   Dataset for temporal prediction in feature space
#
#   - next_timeframe
#   - check_timeseq
#   - dset_time
#   - save_dset_time
#   - load_dset_time
#
# ===========================================================================================================

def next_timeframe( sframe, step ):
    """ -----------------------------------------------------------------------------------------------------
    Return the code of the successor timeframe

    sframe:         [str] code of current timeframe
    step:           [int] incremental step between consecutive frames

    return:         [str] code of next timeframe
    ----------------------------------------------------------------------------------------------------- """
    seq     = sframe.split( '_' )[ 0 ]
    frame   = int( sframe.split( '_' )[ 1 ] )
    dclass  = '_'.join( sframe.split( '_' )[ 2 : -1 ] )
    side    = sframe.split( '_' )[ -1 ]

    p       = "{}_{:06d}_{}_{}"
    n       = frame + step
    return p.format( seq, n, dclass, side )



def check_timeseq( dkeys, sframe, n_inp, n_out, step ):
    """ -----------------------------------------------------------------------------------------------------
    Check if there are enough timeframes ahead in the dataset to generate a new sequence entry

    dkeys:          [set] str keys in the dataset 
    sframe:         [str] code of current timeframe
    n_inp:          [int] number of input frames
    n_out:          [int] number of output frames
    step:           [int] incremental step between consecutive frames

    return:         [bool]
    ----------------------------------------------------------------------------------------------------- """
    step_end    = step * ( n_inp + n_out )
    f           = next_timeframe( sframe, step_end )
    return f in dkeys



def dset_time( dset_latent, n_inp, n_out, dset_odom=None, step=1 ):
    """ -----------------------------------------------------------------------------------------------------
    Generate a dataset for temporal prediction of latent spaces and (optional) odometries.

    Each entry of the dataset is made of (input, target) where:
        - input is a list where:
            ~ (OPTIONAL) the first 'n_inp-1' np.array are tuple (ego speed, steering angle rate)
            ~ the other 'n_inp' np.array are the latent space encodings of image frames
        - output is a list of 'n_out' np.array with the target latent space encodings

    The input is made of N latent spaces and (OPTIONALLY) the corresponding N-1 in-between odometries.

    dset_latent:    [str] pathname of HDF5 file containing feature codes
    n_inp:          [int] number of input frames
    n_out:          [int] number of output frames
    dset_odom:      [str] pathname of HDF5 file containing odometries
    step:           [int] incremental step between consecutive frames

    return:         [tuple]     x:      [list of numpy.array] odometry and input latent spaces
                                y:      [list of numpy.array] target latent spaces
                                xs:     [list of numpy.array] string names of input images
                                ys:     [list of numpy.array] string names of target images
    ----------------------------------------------------------------------------------------------------- """

    # NOTE it is EXTREMELY DANGEROUS to initialize a list of empty lists using integer multiplier
    # and then append to such lists, because they will be all duplicates of the same one
    # for example       >>> x = 3 * [ [] ]
    #                   >>> x[ 0 ].append( 12 )
    # the result will be    [ [12], [12], [12] ]
    # 
    # usign the 'for' construct instead works just fine
    x   = [ [] for i in range( n_inp ) ] if dset_odom is None else [ [] for i in range( 2 * n_inp - 1 ) ]
    y   = [ [] for i in range( n_out ) ]

    xs  = [ [] for i in range( n_inp ) ]    # support list for saving the string name of frames
    ys  = [ [] for i in range( n_out ) ]    # support list for saving the string name of frames

    if not os.path.exists( dset_latent ):
        print( "File {} not found".format( dset_latent ) )
    h_latent    = h5py.File( dset_latent, 'r' )
    k_latent    = set( h_latent.keys() )

    if dset_odom is not None:
        if not os.path.exists( dset_odom ):
            print( "File {} not found".format( dset_odom ) )
        h_odom      = h5py.File( dset_odom, 'r' )
        k_odom      = set( h_odom.keys() )

        # all elements of h_odom must be in h_latent
        # NOTE viceversa is not true because h_odom does not include extremitites
        assert not ( k_odom - k_latent )

    h5_keys     = k_latent if dset_odom is None else k_odom
    
    # read from HDF5 files to collect all the data
    d_latent    = {}
    d_odom      = {}

    for k in h5_keys:
        d_latent[ k ]       = h_latent[ k ].value.reshape( ( latent_size, ) )

        if dset_odom is not None:
            if NORM_ODOM:
                norm        = np.array( ( 0.1, 1.0 ) )
            d_odom[ k ]     = norm * h_odom[ k ].value if NORM_ODOM else h_odom[ k ].value

            h_odom.close()
    h_latent.close()

    # fill the lists in the right order
    for k in sorted( h5_keys ):

        # check if there is enough data for a new sequence
        if not check_timeseq( h5_keys, k, n_inp, n_out, step ):          
            continue

        if dset_odom is not None:
            f       = k                                 # initial frame code
            amount  = amount
            for i in range( amount ):
                x[ i ].append( d_odom[ f ] )            # insert odometry in input data
                f   = next_timeframe( f, step )
        else:
            amount  = 0

        f       = k                                     # restore initial frame code
        for i in range( n_inp ):
            x[ i + amount ].append( d_latent[ f ] )     # insert latent in input data
            xs[ i ].append( f )                         # save string name
            f   = next_timeframe( f, step )

        for i in range( n_out ):
            y[ i ].append( d_latent[ f ] )              # insert latent in output data
            ys[ i ].append( f )                         # save string name
            f   = next_timeframe( f, step )             # <<<<<<<< !!!!!!!! <<<<<<<< !!!!!!!! <<<<<<<<< !!!!!!!

    # NOTE it is MUCH FASTER working with list and then converting to np.array
    # because np.append is very inefficient
    x   = [ np.array( l ) for l in x ]
    y   = [ np.array( l ) for l in y ]
    xs  = [ np.array( l ) for l in xs ]
    ys  = [ np.array( l ) for l in ys ]

    return x, y, xs, ys



def save_dset_time( dset_latent, n_inp, n_out, dset_odom=None, step=1, size=None ):
    """ -----------------------------------------------------------------------------------------------------
    Generate a dataset (with training, validation and test) for temporal prediction
    of latent spaces and (optionally) odometries. Then store it on a HDF5 file.

    dset_latent:    [str] pathname of HDF5 file containing feature codes
    n_inp:          [int] number of input frames
    n_out:          [int] number of output frames
    dset_odom:      [str] pathname of HDF5 file containing odometries
    step:           [int] incremental step between consecutive frames
    size:           [int] amount of data to consider (if None consider all data)
    ----------------------------------------------------------------------------------------------------- """
    print( "Starting executing: {}{}, inp={} out={} step={}".format(
                '' if size is None else 'small ',
                dset_latent.split( '/' )[ -1 ].split( '_' )[ 0 ],
                n_inp,
                n_out,
                step
    ) )

    x, y, xs, ys    = dset_time( dset_latent, n_inp, n_out, dset_odom=dset_odom, step=step )

    tsize           = len( x[ 0 ] ) if size is None else size
    bsize           = batch_msize   if size is None else 1

    random.seed( SEED )
    indx            = random.sample( range( len( x[ 0 ] ) ), tsize )

    # randomly permute the input and output data
    for i in range( len( x ) ):
        x[ i ]  = x[ i ][ indx ]
    for i in range( len( xs ) ):
        xs[ i ] = xs[ i ][ indx ]
    for i in range( len( y ) ):
        y[ i ]  = y[ i ][ indx ]
        ys[ i ] = ys[ i ][ indx ]

    # partition the data into the 3 subsets, for intput and output
    i_tr, i_vd, i_ts    = frac_dataset( frac_train, frac_valid, frac_test, tsize, batch_size=bsize )

    x_tr                = [ x[ i ][ i_tr ] for i in range( len( x ) ) ]
    x_vd                = [ x[ i ][ i_vd ] for i in range( len( x ) ) ]
    x_ts                = [ x[ i ][ i_ts ] for i in range( len( x ) ) ]

    y_tr                = [ y[ i ][ i_tr ] for i in range( len( y ) ) ]
    y_vd                = [ y[ i ][ i_vd ] for i in range( len( y ) ) ]
    y_ts                = [ y[ i ][ i_ts ] for i in range( len( y ) ) ]

    xs_tr               = [ xs[ i ][ i_tr ] for i in range( len( xs ) ) ]
    xs_vd               = [ xs[ i ][ i_vd ] for i in range( len( xs ) ) ]
    xs_ts               = [ xs[ i ][ i_ts ] for i in range( len( xs ) ) ]

    ys_tr               = [ ys[ i ][ i_tr ] for i in range( len( ys ) ) ]
    ys_vd               = [ ys[ i ][ i_vd ] for i in range( len( ys ) ) ]
    ys_ts               = [ ys[ i ][ i_ts ] for i in range( len( ys ) ) ]

    dd  = { 'dataset': (
                ( x_tr, y_tr ),         # train input and target data
                ( x_vd, y_vd ),         # valid input and target data
                ( x_ts, y_ts ),         # test input and target data
                ( xs_tr, ys_tr ),       # train input and target strings
                ( xs_vd, ys_vd ),       # valid input and target strings
                ( xs_ts, ys_ts )        # test input and target strings
    ) }

    s   = dset_latent.split( '/' )[ -1 ].split( '.' )[ 0 ]
    s   += '__{}_inp{:02d}_out{:02d}_step{:02d}_{}.h5'
    ss  = os.path.join( synthia_time, s.format(
                ( 'lat' if dset_odom is None else 'odo' ),
                n_inp,
                n_out,
                step,
                ( 'L' if size is None else 'S' )
    ) )

    deepdish.io.save( ss, dd )



def load_dset_time( h5_file ):
    """ -----------------------------------------------------------------------------------------------------
    Load the full dataset for temporal prediction from HDF5 file

    h5_file:    [str] name of HDF5 file

    return:     [tuple of tuple]    training input and target data
                                    validation input and target data
                                    testing input and target data
                                    training input and target strings
                                    validation input and target strings
                                    testing input and target strings
    ----------------------------------------------------------------------------------------------------- """
    h5_file = os.path.join( synthia_time, h5_file )
    dd      = deepdish.io.load( h5_file )
    return dd[ 'dataset' ]



# ===========================================================================================================
#
#   General fuctions for handling frame files
#
#   - frac_dataset
#   - make_symlink
#   - check_valid
#
#   - resize_img
#   - convert_gt
#   - resize_synthia
#
# ===========================================================================================================

def frac_dataset( f_train, f_valid, f_test, size, batch_size=1 ):
    """ -----------------------------------------------------------------------------------------------------
    Return 3 slices dividing a dataset into training, validation and test sets.

    Each subset size is computed to be divisible by the 'batch_size' (or the largest one)
    that will be used during training. This because Keras functions prefer dataset like that...

    f_train:        [float] fraction of images to be used as training set
    f_valid:        [float] fraction of images to be used as validation set
    f_test:         [float] fraction of images to be used as test set
    size:           [int] total number of images in the dataset
    batch_size:     [int] maximum size of the minibatch (if 1 consider no limitation to the size)

    return:         [list of slice]
    ----------------------------------------------------------------------------------------------------- """
    assert ( f_train + f_valid + f_test - 1 ) < 0.001

    f_tr        = int( batch_size * ( f_train * size // batch_size ) )
    f_vd        = int( batch_size * ( f_valid * size // batch_size ) )
    f_ts        = int( batch_size * ( f_test * size // batch_size ) )

    indx_train  = slice( 0,             f_tr )
    indx_valid  = slice( f_tr,          f_tr + f_vd )
    indx_test   = slice( f_tr + f_vd,   f_tr + f_vd + f_ts )

    if DEBUG:
        print( "Total:\t{}\nTrain:\t{}\nValid:\t{}\nTest:\t{}\nUnused:\t{}\n".format(
                    size, f_tr, f_vd, f_ts, size-f_tr-f_vd-f_ts ) )
    
    return indx_train, indx_valid, indx_test



def make_symlink( src_dir, dest_dir, files ):
    """ -----------------------------------------------------------------------------------------------------
    Make symbolic links of files from a folder to a second one

    NOTE: this assumes that, in case of multiple source folders, each of them has unique file names

    src_dir:        [str or list of str] source folder(s) containing original images
    dest_dir:       [str] destination folder
    files:          [list of str] name of files to be linked
    ----------------------------------------------------------------------------------------------------- """
    if isinstance( src_dir, str ):
        src_rel     = os.path.relpath( src_dir, dest_dir )          # get 'src' path relative to 'dest'
        for f in files:
            os.symlink( os.path.join( src_rel, f ), os.path.join( dest_dir, f ) )

    elif isinstance( src_dir, ( list, tuple ) ):
        src_rel     = [ os.path.relpath( sd, dest_dir ) for sd in src_dir ]
        for f in files:
            for i in range( len( src_dir ) ):
                if os.path.isfile( os.path.join( src_dir[ i ], f ) ):
                    os.symlink( os.path.join( src_rel[ i ], f ), os.path.join( dest_dir, f ) )



def check_valid( img, threshold ):
    """ -----------------------------------------------------------------------------------------------------
    Check if an image has at least 'threshold' white pixels

    img:            [str] path of image file
    threshold:      [int] required number of white pixels
    ----------------------------------------------------------------------------------------------------- """
    f       = Image.open( img )
    i       = np.array( f )
    f.close()
    return ( i == 255 ).sum() > threshold



def resize_img( img_i, img_o, w, h ):
    """ -----------------------------------------------------------------------------------------------------
    Resize an image to a target size, and save the result.
    If the original aspect ratio does not match the target one, the image is scaled and then cropped to
    avoid stretch deformations.

    img_i:          [str] path to the image to resize
    img_o:          [str] path of output image
    w:              [int] new width
    h:              [int] new height

    return:         [list of int] scaling factor, horizontal padding, vertical padding
    ----------------------------------------------------------------------------------------------------- """
    i1      = Image.open( img_i ) 
    w1, h1  = i1.size
    scale   = min( w1 // w, h1 // h )                       # scaling factor

    ws      = scale * w
    hs      = scale * h
    wm      = ( w1 - ws ) // 2
    hm      = ( h1 - hs ) // 2

    i2      = i1.crop( ( wm, hm, w1-wm, h1-hm ) )           # cropped image
    i3      = i2.resize( ( w, h ), Image.ANTIALIAS )        # scaled image

    i3      = i3.convert( 'RGB' )
    i3.save( img_o )

    return ( scale, wm, hm )



def convert_gt( img_i, img_o, cls, scale, w, h, x0, y0 ):
    """ -----------------------------------------------------------------------------------------------------
    Convert and rescale an image from the LABELS folder of Synthia dataset, to obtain a binary image with
    true pixels belonging to a single class (e.g. only cars, only lane, etc)

    img_i:          [str] path of input image
    img_o:          [str] path of output image
    cls:            [str] one of the categories of 'class_synthia'
    scale:          [int] scaling factor
    w:              [int] final width
    h:              [int] final height
    x0:             [int] horizontal padding
    y0:             [int] vertical padding
    ----------------------------------------------------------------------------------------------------- """
    if not cls in class_synthia:
        print( "Class {} not found for SYNTHIA dataset".format( cls ) )
        sys.exit( 1 )
    
    x1      = x0 + scale * w
    y1      = y0 + scale * h

    i       = png.Reader( img_i )
    i       = i.read()
    i       = np.array( list( i[ 2 ] ) )

    # rearrange the numpy array, discarding the useless channels, cropping and resizing
    a       = ( i.shape[ 0 ], i.shape[ 1 ] // 3, 3 )
    a       = i.reshape( a )[ y0 : y1 : scale, x0 : x1 : scale, 0 ]

    # set to 1 the true pixels, convert to image and save using PIL
    gt      = a == class_synthia[ cls ]
    gt      = np.uint8( gt ) * 255
    img     = Image.fromarray( gt )
    img.save( img_o )



def resize_synthia( src_dir, dest_dir, w, h, prfx='' ):
    """ -----------------------------------------------------------------------------------------------------
    Generate a downscaled version of a SYNTHIA dataset sequence

    src_dir:        [str] source folder (one of the sequence subdirectories, like 'SYNTHIA-SEQS-01-SPRING'
    dest_dir:       [str] destination folder
    w:              [int] new width
    h:              [int] new height
    prfx:           [str] prefix for the name of the new files
    ----------------------------------------------------------------------------------------------------- """
    if not os.path.isdir( src_dir ):
        print( warn.format( src_dir ) )
        return

    dir_rgb_L   = os.path.join( src_dir, 'RGB/Stereo_Left/Omni_F' )         # RGB images
    dir_rgb_R   = os.path.join( src_dir, 'RGB/Stereo_Right/Omni_F' )
    dir_dpt_L   = os.path.join( src_dir, 'Depth/Stereo_Left/Omni_F' )       # depth maps
    dir_dpt_R   = os.path.join( src_dir, 'Depth/Stereo_Right/Omni_F' )
    dir_sgm_L   = os.path.join( src_dir, 'GT/LABELS/Stereo_Left/Omni_F' )   # semantic segments
    dir_sgm_R   = os.path.join( src_dir, 'GT/LABELS/Stereo_Right/Omni_F' )

    if os.path.exists( dest_dir ):
        shutil.rmtree( dest_dir )
    os.makedirs( dest_dir )

    # NOTE this assumes that all dirs have the same number of files with the same names
    files       = [ f for f in sorted( os.listdir( dir_rgb_L ) ) if f.endswith( '.png' ) ]
    
    for f in files:
        f_name          = f.split( '.' )[ 0 ]

        scale, wm, hm   = resize_img(
                os.path.join( dir_rgb_L, f ),
                os.path.join( dest_dir, prfx + f_name + '_RGB_FL.jpg' ),
                w, h
        )
        resize_img(
                os.path.join( dir_rgb_R, f ),
                os.path.join( dest_dir, prfx + f_name + '_RGB_FR.jpg' ),
                w, h
        )
        resize_img(         # TODO is the resize of the depth map truly preserving its information ? ? ?
                os.path.join( dir_dpt_L, f ),
                os.path.join( dest_dir, prfx + f_name + '_DEPTH_FL.png' ),
                w, h
        )
        resize_img(
                os.path.join( dir_dpt_R, f ),
                os.path.join( dest_dir, prfx + f_name + '_DEPTH_FR.png' ),
                w, h
        )
        convert_gt(
                os.path.join( dir_sgm_L, f ),
                os.path.join( dest_dir, prfx + f_name + '_SEGM_CAR_FL.png' ),
                'CAR',
                scale,
                w,  h,
                wm, hm
        )
        convert_gt(
                os.path.join( dir_sgm_L, f ),
                os.path.join( dest_dir, prfx + f_name + '_SEGM_LANE_FL.png' ),
                'LANE',
                scale,
                w,  h,
                wm, hm
        )
        convert_gt(
                os.path.join( dir_sgm_R, f ),
                os.path.join( dest_dir, prfx + f_name + '_SEGM_CAR_FR.png' ),
                'CAR',
                scale,
                w,  h,
                wm, hm
        )
        convert_gt(
                os.path.join( dir_sgm_R, f ),
                os.path.join( dest_dir, prfx + f_name + '_SEGM_LANE_FR.png' ),
                'LANE',
                scale,
                w,  h,
                wm, hm
        )

        

def plain_link_synthia( src_dir, dest_dir, prfx='', copyfile=False ):
    """ -----------------------------------------------------------------------------------------------------
    Make links of original SYNTHIA files and organize them in folders

    src_dir:        [str] source folder (one of the sequence subdirectories, like 'SYNTHIA-SEQS-01-SPRING'
    dest_dir:       [str] destination folder
    prfx:           [str] prefix for the name of the new files
    ----------------------------------------------------------------------------------------------------- """
    if not os.path.isdir( src_dir ):
        print( warn.format( src_dir ) )
        return

    d_tmp       = os.path.join( src_dir, 'RGB/Stereo_Left/Omni_F' )         # RGB images

    if not copyfile:
        src_dir     = src_dir.replace( 'dataset/synthia', '../..' )

    dir_rgb_L   = os.path.join( src_dir, 'RGB/Stereo_Left/Omni_F' )         # RGB images
    dir_rgb_R   = os.path.join( src_dir, 'RGB/Stereo_Right/Omni_F' )

    if os.path.exists( dest_dir ):
        shutil.rmtree( dest_dir )
    os.makedirs( dest_dir )

    # NOTE this assumes that all dirs have the same number of files with the same names
    files       = [ f for f in sorted( os.listdir( d_tmp ) ) if f.endswith( '.png' ) ]

    func        = shutil.copyfile if copyfile else os.symlink
    
    for f in files:
        f_name          = f.split( '.' )[ 0 ]

        func(
                os.path.join( dir_rgb_L, f ),
                os.path.join( dest_dir, prfx + f_name + '_RGB_FL.jpg' )
        )
        func(
                os.path.join( dir_rgb_R, f ),
                os.path.join( dest_dir, prfx + f_name + '_RGB_FR.jpg' ),
        )
        


# ===========================================================================================================
#
#   Generate structured datasets
#
#   - dset_noclass
#   - dset_class
#   - dset_seq
#   - dset_seq_class
#
# ===========================================================================================================

def get_files_noclass( src_dir, cond=None ):
    """ -----------------------------------------------------------------------------------------------------
    Utility for retrieving all files necessary for a dataset without classes

    src_dir:        [str or list of str] source folder(s) containing original images
    cond:           [function] condition for selecting only some of the image files
    ----------------------------------------------------------------------------------------------------- """
    # if no condition on file selection was given, select all
    c       = ( lambda x: True ) if cond is None else cond

    # get list of all files - in the case of a single source folder
    if isinstance( src_dir, str ):
        # if the name of a SYNTHIA sub-sequence is not available, skip to the next one
        if not os.path.isdir( src_dir ):
            print( warn.format( src_dir ) )
            return

        files   = [ f for f in sorted( os.listdir( src_dir ) ) if c( f ) and cond_ext( f ) ]

    # get list of all files - in the case of multiple source folders
    elif isinstance( src_dir, ( list, tuple ) ):
        files   = []
        for d in src_dir:
            # if the name of a SYNTHIA sub-sequence is not available, skip to the next one
            if not os.path.isdir( d ):
                print( warn.format( d ) )
                continue

            for f in sorted( os.listdir( d ) ):
                if c( f ) and cond_ext( f ):
                    files.append( f )

    return files


def get_files_class( src_dir, cond_list ):
    """ -----------------------------------------------------------------------------------------------------
    Utility for retrieving all files necessary for a dataset with multiple classes

    src_dir:        [str or list of str] source folder(s) containing original images
    cond_list:      [list of function] condition for selecting files for each sub-category
    ----------------------------------------------------------------------------------------------------- """
    n_class     = len( cond_list )

    files       = [ [] for i in range( n_class ) ]

    # get list of all files, for each class - in the case of a single source folder
    if isinstance( src_dir, str ):
        # if the name of a SYNTHIA sub-sequence is not available, skip to the next one
        if not os.path.isdir( src_dir ):
            print( warn.format( src_dir ) )
            return

        for f in sorted( os.listdir( src_dir ) ):
            for c in range( n_class ):
                if cond_list[ c ]( f ) and cond_ext( f ):
                    files[ c ].append( f )

    # get list of all files, for each class - in the case of multiple source folders
    elif isinstance( src_dir, ( list, tuple ) ):
        for dr in src_dir:
            # if the name of a SYNTHIA sub-sequence is not available, skip to the next one
            if not os.path.isdir( dr ):
                print( warn.format( dr ) )
                continue

            for f in sorted( os.listdir( dr ) ):
                for c in range( n_class ):
                    if cond_list[ c ]( f ) and cond_ext( f ):
                        files[ c ].append( f )

    return files


def dset_noclass( src_dir, dest_dir, size=None, cond=None, batch_size=1 ):
    """ -----------------------------------------------------------------------------------------------------
    Create a structured dataset made of simbolic links to image files.
    The dataset has no sub-categories (input equal to target).

    In case of multiple source folders, it assumes that files in each folder have unique names.

    src_dir:        [str or list of str] source folder(s) containing original images
    dest_dir:       [str] destination folder
    size:           [int] amount of files to link (if None consider all files)
    cond:           [function] condition for selecting only some of the image files
    batch_size:     [int] max allowed size of the minibatch (if 1 consider no limitation to the size)
    ----------------------------------------------------------------------------------------------------- """
    dest_train  = os.path.join( dest_dir, 'train/img' )
    dest_valid  = os.path.join( dest_dir, 'valid/img' )
    dest_test   = os.path.join( dest_dir, 'test/img' )
    
    # remove the folder if already existed, and create a fresh one
    if os.path.exists( dest_dir ):
        shutil.rmtree( dest_dir )
    os.makedirs( dest_dir )
    os.makedirs( dest_train )
    os.makedirs( dest_valid )
    os.makedirs( dest_test )

    files   = get_files_noclass( src_dir, cond )

    # randomly permute the file list
    random.seed( SEED )
    s                   = len( files ) if size is None else size
    files               = random.sample( files, s )

    # partition file list into the 3 subsets
    i_tr, i_vd, i_ts    = frac_dataset( frac_train, frac_valid, frac_test, s, batch_size=batch_size )
    files_train         = files[ i_tr ]
    files_valid         = files[ i_vd ]
    files_test          = files[ i_ts ]

    # make symbolic links
    make_symlink( src_dir, dest_train, files_train )
    make_symlink( src_dir, dest_valid, files_valid )
    make_symlink( src_dir, dest_test, files_test )


def dset_class( src_dir, dest_dir, class_list, cond_list, size=None, batch_size=1 ):
    """ -----------------------------------------------------------------------------------------------------
    Create a structured dataset made of simbolic links to image files.
    The dataset has multiple sub-categories (input different fromt the target).

    In case of multiple source folders, it assumes that files in each folder have unique names.

    src_dir:        [str or list of str] source folder(s) containing original images
    dest_dir:       [str] destination folder
    class_list:     [list of str] names of the sub-categories
    cond_list:      [list of function] condition for selecting files for each sub-category
    size:           [int] amount of files to link (if None consider all files)
    batch_size:     [int] max allowed size of the minibatch (if 1 consider no limitation to the size)
    ----------------------------------------------------------------------------------------------------- """
    assert len( class_list ) == len( cond_list )

    n_class         = len( class_list )
    dest_train      = [ os.path.join( dest_dir, 'train/{}/img'.format( c ) ) for c in class_list ]
    dest_valid      = [ os.path.join( dest_dir, 'valid/{}/img'.format( c ) ) for c in class_list ]
    dest_test       = [ os.path.join( dest_dir, 'test/{}/img'.format( c ) ) for c in class_list ]

    # remove the folder if already existed, and create a fresh one
    if os.path.exists( dest_dir ):
        shutil.rmtree( dest_dir )
    os.makedirs( dest_dir )
    for d in dest_train + dest_valid + dest_test:
        os.makedirs( d )

    files   = get_files_class( src_dir, cond_list )

    # randomly permute the file list, for each class
    random.seed( SEED )
    s       = len( files[ 0 ] ) if size is None else size
    indx    = random.sample( range( len( files[ 0 ] ) ), s )
    for c in range( n_class ):
        files[ c ]  = np.array( files[ c ] )
        files[ c ]  = files[ c ][ indx ]

    # partition file list into the 3 subsets, for each class
    i_tr, i_vd, i_ts    = frac_dataset( frac_train, frac_valid, frac_test, s, batch_size=batch_size )
    files_train         = [ files[ c ][ i_tr ] for c in range( n_class ) ]
    files_valid         = [ files[ c ][ i_vd ] for c in range( n_class ) ]
    files_test          = [ files[ c ][ i_ts ] for c in range( n_class ) ]

    # make symbolic links
    for c in range( n_class ):
        make_symlink( src_dir, dest_train[ c ], files_train[ c ] )
        make_symlink( src_dir, dest_valid[ c ], files_valid[ c ] )
        make_symlink( src_dir, dest_test[ c ], files_test[ c ] )


def dset_seq( src_dir, dest_dir, seq=(1,2), cond=None, size=None, batch_size=1 ):
    """ -----------------------------------------------------------------------------------------------------
    Create a structured dataset made of simbolic links to image files.
    The dataset has as many sub-categories as the items in the sequence seq

    In case of multiple source folders, it assumes that files in each folder have unique names.

    src_dir:        [str or list of str] source folder(s) containing original images
    dest_dir:       [str] destination folder
    seq:            [tuple] items in a sequence, with relative spacing, omitting the first one
                    for example (1,2) means there are 3 consecutive frames, (1,3) means that there
                    are 3 frames, with the first two consecutive, and the third two steps ahead
    cond:           [function] condition for selecting only some of the image files
    size:           [int] amount of files to link (if None consider all files)
    batch_size:     [int] max allowed size of the minibatch (if 1 consider no limitation to the size)
    ----------------------------------------------------------------------------------------------------- """
    seq_list        = [ name_seq_dir.format( 0 ) ]
    for i in seq:
        seq_list.append( name_seq_dir.format( i ) )
    seq_len         = len( seq_list )
    dest_train      = [ os.path.join( dest_dir, 'train/{}/img'.format( c ) ) for c in seq_list ]
    dest_valid      = [ os.path.join( dest_dir, 'valid/{}/img'.format( c ) ) for c in seq_list ]
    dest_test       = [ os.path.join( dest_dir, 'test/{}/img'.format( c ) ) for c in seq_list ]

    # remove the folder if already existed, and create a fresh one
    if os.path.exists( dest_dir ):
        shutil.rmtree( dest_dir )
    os.makedirs( dest_dir )
    for d in dest_train + dest_valid + dest_test:
        os.makedirs( d )

    frames      = [ [] for i in range( seq_len ) ]
    files       = get_files_noclass( src_dir, cond )

    for f in files:
        if check_timeseq( files, f, 1, seq[ -1 ], 1 ):
            frames[ 0 ].append( f )
            for i, s in enumerate( seq ):
                frames[ i+1 ].append( next_timeframe( f, s ) )

    # randomly permute the file list, for each class
    random.seed( SEED )
    s       = len( frames[ 0 ] ) if size is None else size
    indx    = random.sample( range( len( frames[ 0 ] ) ), s )
    for c in range( seq_len ):
        frames[ c ] = np.array( frames[ c ] )
        frames[ c ] = frames[ c ][ indx ]

    # partition file list into the 3 subsets, for each class
    i_tr, i_vd, i_ts    = frac_dataset( frac_train, frac_valid, frac_test, s, batch_size=batch_size )
    files_train         = [ frames[ c ][ i_tr ] for c in range( seq_len ) ]
    files_valid         = [ frames[ c ][ i_vd ] for c in range( seq_len ) ]
    files_test          = [ frames[ c ][ i_ts ] for c in range( seq_len ) ]

    # make symbolic links
    for c in range( seq_len ):
        make_symlink( src_dir, dest_train[ c ], files_train[ c ] )
        make_symlink( src_dir, dest_valid[ c ], files_valid[ c ] )
        make_symlink( src_dir, dest_test[ c ], files_test[ c ] )



def set_mseq_tree( dest_dir, seq, class_list ):
    """ -----------------------------------------------------------------------------------------------------
    Utilify for creating the directory tree in the case of sequences with multiple classes

    dest_dir:       [str] destination folder
    seq:            [tuple] items in a sequence, see description in dset_seq_multi()
    class_list:     [list of str] names of the sub-categories
    ----------------------------------------------------------------------------------------------------- """
    seq_list        = [ name_seq_dir.format( 0 ) ]
    for i in seq:
        seq_list.append( name_seq_dir.format( i ) )
    dest_train      = [ [ os.path.join( dest_dir, 'train/{}/{}/img'.format( s, c ) )
            for c in class_list ]
                for s in seq_list
    ]
    dest_valid      = [ [ os.path.join( dest_dir, 'valid/{}/{}/img'.format( s, c ) )
            for c in class_list ]
                for s in seq_list
    ]
    dest_test       = [ [ os.path.join( dest_dir, 'test/{}/{}/img'.format( s, c ) )
            for c in class_list ]
                for s in seq_list
    ]

    # remove the folder if already existed, and create a fresh one
    if os.path.exists( dest_dir ):
        shutil.rmtree( dest_dir )
    os.makedirs( dest_dir )
    for tree in dest_train + dest_valid + dest_test:
        for d in tree:
            os.makedirs( d )

    return dest_train, dest_valid, dest_test


def dset_seq_multi( src_dir, dest_dir, seq, class_list, cond_list, size=None, batch_size=1 ):
    """ -----------------------------------------------------------------------------------------------------
    Create a structured dataset made of simbolic links to image files.
    The dataset has as many sub-categories as the items in the sequence seq

    In case of multiple source folders, it assumes that files in each folder have unique names.

    src_dir:        [str or list of str] source folder(s) containing original images
    dest_dir:       [str] destination folder
    seq:            [tuple] items in a sequence, with relative spacing, omitting the first one
                    for example (1,2) means there are 3 consecutive frames, (1,3) means that there
                    are 3 frames, with the first two consecutive, and the third two steps ahead
    class_list:     [list of str] names of the sub-categories
    cond_list:      [list of function] condition for selecting files for each sub-category
    size:           [int] amount of files to link (if None consider all files)
    batch_size:     [int] max allowed size of the minibatch (if 1 consider no limitation to the size)
    ----------------------------------------------------------------------------------------------------- """
    assert len( class_list ) == len( cond_list )

    seq_len = 1 + len( seq )
    n_class = len( class_list )
    dest_train, dest_valid, dest_test   = set_mseq_tree( dest_dir, seq, class_list )


    # get all files, for all classes, at frame 0
    files   = get_files_class( src_dir, cond_list )

    frames  = [ [ [] for c in range( n_class ) ] for f in range( seq_len ) ]
    # now get all files for the subsequent frames, if available
    for i, f in enumerate( files[ 0 ] ):
        if check_timeseq( files[ 0 ], f, 1, seq[ -1 ], 1 ):
            for c in range( n_class ):
                frames[ 0 ][ c ].append( files[ c ][ i ] )
                for j, s in enumerate( seq ):
                    frames[ j+1 ][ c ].append( next_timeframe( files[ c ][ i ], s ) )

    # randomly permute the file list, for each class
    random.seed( SEED )
    size    = len( frames[ 0 ][ 0 ] ) if size is None else size
    indx    = random.sample( range( len( frames[ 0 ][ 0 ] ) ), size )
    for s in range( seq_len ):
        for c in range( n_class ):
            frames[ s ][ c ] = np.array( frames[ s ][ c ] )
            frames[ s ][ c ] = frames[ s ][ c ][ indx ]

    # partition file list into the 3 subsets, for each class
    i_tr, i_vd, i_ts    = frac_dataset( frac_train, frac_valid, frac_test, size, batch_size=batch_size )
    files_train         = [ [ frames[ s ][ c ][ i_tr ] for c in range( n_class ) ] for s in range( seq_len ) ]
    files_valid         = [ [ frames[ s ][ c ][ i_vd ] for c in range( n_class ) ] for s in range( seq_len ) ]
    files_test          = [ [ frames[ s ][ c ][ i_ts ] for c in range( n_class ) ] for s in range( seq_len ) ]

    # make symbolic links
    for s in range( seq_len ):
        for c in range( n_class ):
            make_symlink( src_dir, dest_train[ s ][ c ], files_train[ s ][ c ] )
            make_symlink( src_dir, dest_valid[ s ][ c ], files_valid[ s ][ c ] )
            make_symlink( src_dir, dest_test[ s ][ c ], files_test[ s ][ c ] )



# ===========================================================================================================
#
#   MAIN
#
#   NOTE to be executed from the main folder (above 'src' and 'dataset')
#
# ===========================================================================================================
if __name__ == '__main__':

    # -------------------------------------------------------------------------------------------------------
    # generate a downscaled version of the SYNTHIA dataset images
    # -------------------------------------------------------------------------------------------------------
    if PLAIN_LINK_SYNTHIA:
        for s in synthia_seqs:
            for n in synthia_names:
                print( "Starting: plain linking SYNTHIA {}".format( n[ 1 ].format( s ) ) )
                plain_link_synthia( 
                        os.path.join( synthia_orig, n[ 0 ].format( s ) ),
                        os.path.join( synthia_orig_neat, n[ 1 ].format( s ) ),
                        prfx        = n[ 2 ].format( s ),
                        copyfile    = True
                )

    # -------------------------------------------------------------------------------------------------------
    # generate odometry data (speed, steer) for the SYNTHIA dataset
    # -------------------------------------------------------------------------------------------------------
    if ODOM_SYNTHIA:
        if os.path.exists( synthia_odom ):
            shutil.rmtree( synthia_odom )
        os.makedirs( synthia_odom )

        for oo in odo_step:
            odo = dict_odom( step=oo )
            fn  = os.path.join( synthia_odom, name_odom.format( oo ) )
            f   = h5py.File( fn, 'w' )
            for k in odo.keys():
                f.create_dataset( k, data=np.array( odo[ k ] ) )
            f.close()

    # -------------------------------------------------------------------------------------------------------
    # generate dataset for time prediction
    # -------------------------------------------------------------------------------------------------------
    if TIME_SYNTHIA:
        if not os.path.exists( synthia_time ):
            #shutil.rmtree( synthia_time )
            os.makedirs( synthia_time )

        """
        save_dset_time(
                os.path.join( synthia_latent, "MVAE_19-07-22_08-43-42.h5" ),   # dset_latent
                4,                                                              # n_inp
                2,                                                              # n_out
                step        = 1,
                dset_odom   = None,                                             # no odometry
                size        = None
        )
        
        """
        save_dset_time(
                os.path.join( synthia_latent, "RMVAE_19-10-01_13-20-08.h5" ),   # dset_latent
                10,                                                             # n_inp
                2,                                                              # n_out
                step        = 1,
                dset_odom   = None,                                             # no odometry
                size        = None
        )

        """
        save_dset_time(
                os.path.join( synthia_latent, "RMVAE_19-10-01_13-20-08.h5" ),   # dset_latent
                3,                                                              # n_inp
                2,                                                              # n_out
                step        = 1,
                dset_odom   = None,                                             # no odometry
                size        = 500
        )

        save_dset_time(
                os.path.join( synthia_latent, "RMVAE_19-10-01_13-20-08.h5" ),   # dset_latent
                8,                                                              # n_inp
                1,                                                              # n_out
                step        = 4,
                dset_odom   = None,                                             # no odometry
                size        = None
        )

        save_dset_time(
                os.path.join( synthia_latent, "RGB_19-04-22_16-00-24.h5" ),     # dset_latent
                2,                                                              # n_inp
                2,                                                              # n_out
                step        = 1,
                dset_odom   = os.path.join( synthia_odom, "step_01.h5" ),
                size        = None
        )        
        """

    # -------------------------------------------------------------------------------------------------------
    # generate a downscaled version of the SYNTHIA dataset images
    # -------------------------------------------------------------------------------------------------------
    if SCALE_SYNTHIA:
        for s in synthia_seqs:
            for n in synthia_names:
                print( "Starting: scaling SYNTHIA {}".format( n[ 1 ].format( s ) ) )
                resize_synthia( 
                        os.path.join( synthia_orig, n[ 0 ].format( s ) ),
                        os.path.join( synthia_scaled, n[ 1 ].format( s ) ),
                        256, 128,
                        #1024, 512,
                        prfx    = n[ 2 ].format( s )
                )
 
    # -------------------------------------------------------------------------------------------------------
    # generate structured datasets made of simbolic links of the SYNTHIA images
    # for each dataset, produce a 'small' version of few images, and a full 'large' version
    # -------------------------------------------------------------------------------------------------------
    if LINK_SYNTHIA:
        small_size      = 500   # size of the 'small' version of the datasets
        synthia_full    = []
        for s in synthia_seqs:
            for n in synthia_names:
                synthia_full.append( os.path.join( synthia_scaled, n[ 1 ].format( s ) ) )

        print( "Starting executing: SYNTHIA mseq12_S" )
        dset_seq_multi(
                synthia_full,
                os.path.join( synthia_link, 'mseq12_S' ),
                ( 1, 2 ),
                [ 'rgb', 'car', 'lane' ],
                [
                    lambda x: '_RGB_' in x,
                    lambda x: '_SEGM_CAR_' in x,
                    lambda x: '_SEGM_LANE_' in x
                ],
                size        = small_size
        )

        print( "Starting executing: SYNTHIA mseq12_L" )
        dset_seq_multi(
                synthia_full,
                os.path.join( synthia_link, 'mseq12_L' ),
                ( 1, 2 ),
                [ 'rgb', 'car', 'lane' ],
                [
                    lambda x: '_RGB_' in x,
                    lambda x: '_SEGM_CAR_' in x,
                    lambda x: '_SEGM_LANE_' in x
                ],
                batch_size  = batch_msize
        )

        """
        print( "Starting executing: SYNTHIA seq12_S" )
        dset_seq(
                synthia_full,
                os.path.join( synthia_link, 'seq12_S' ),
                ( 1, 2 ),
                lambda x: '_RGB_' in x,
                size        = small_size
        )

        print( "Starting executing: SYNTHIA seq12_L" )
        dset_seq(
                synthia_full,
                os.path.join( synthia_link, 'seq12_L' ),
                ( 1, 2 ),
                lambda x: '_RGB_' in x,
                batch_size  = batch_msize
        )

        print( "Starting executing: SYNTHIA rgb_S" )
        dset_noclass(
                synthia_full,
                os.path.join( synthia_link, 'rgb_S' ),
                cond    = lambda x: '_RGB_' in x,
                size    = small_size
        )

        print( "Starting executing: SYNTHIA rgb_L" )
        dset_noclass(
                synthia_full,
                os.path.join( synthia_link, 'rgb_L' ),
                cond        = lambda x: '_RGB_' in x,
                batch_size  = batch_msize
        )

        print( "Starting executing: SYNTHIA car_S" )
        dset_class(
                synthia_full,
                os.path.join( synthia_link, 'car_S' ),
                [ 'rgb', 'car' ],
                [
                    lambda x: '_RGB_' in x,
                    lambda x: '_SEGM_CAR_' in x
                ],
                size        = small_size
        )

        print( "Starting executing: SYNTHIA car_L" )
        dset_class(
                synthia_full,
                os.path.join( synthia_link, 'car_L' ),
                [ 'rgb', 'car' ],
                [
                    lambda x: '_RGB_' in x,
                    lambda x: '_SEGM_CAR_' in x
                ],
                batch_size  = batch_msize
        )

        print( "Starting executing: SYNTHIA lane_S" )
        dset_class(
                synthia_full,
                os.path.join( synthia_link, 'lane_S' ),
                [ 'rgb', 'lane' ],
                [
                    lambda x: '_RGB_' in x,
                    lambda x: '_SEGM_LANE_' in x
                ],
                size        = small_size
        )

        print( "Starting executing: SYNTHIA lane_L" )
        dset_class(
                synthia_full,
                os.path.join( synthia_link, 'lane_L' ),
                [ 'rgb', 'lane' ],
                [
                    lambda x: '_RGB_' in x,
                    lambda x: '_SEGM_LANE_' in x
                ],
                batch_size  = batch_msize
        )

        print( "Starting executing: SYNTHIA multi_S" )
        dset_class(
                synthia_full,
                os.path.join( synthia_link, 'multi_S' ),
                [ 'rgb', 'car', 'lane' ],
                [
                    lambda x: '_RGB_' in x,
                    lambda x: '_SEGM_CAR_' in x,
                    lambda x: '_SEGM_LANE_' in x
                ],
                size        = small_size
        )

        print( "Starting executing: SYNTHIA multi_L" )
        dset_class(
                synthia_full,
                os.path.join( synthia_link, 'multi_L' ),
                [ 'rgb', 'car', 'lane' ],
                [
                    lambda x: '_RGB_' in x,
                    lambda x: '_SEGM_CAR_' in x,
                    lambda x: '_SEGM_LANE_' in x
                ],
                batch_size  = batch_msize
        )
        """
