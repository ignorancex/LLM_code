"""
#############################################################################################################

Functions for parsing large datasets during training in Keras

    Alice   2019

#############################################################################################################
"""

import  os
import  h5py
import  random
import  numpy       as np
from    keras       import preprocessing
from    PIL         import Image

import  mesg        as ms
from    tester      import array_to_image, save_image
from    exec_dset   import frac_dataset, frac_train, frac_valid, frac_test, batch_msize


DEBUG0      = False
DEBUG1      = False

file_ext    = ( '.png', '.jpg', '.jpeg' )
cond_ext    = lambda x: x.lower().endswith( file_ext )      # the condition an image file must satisfy

cnfg        = []                                            # NOTE initialized by 'nn_main.py'


# ===========================================================================================================
#
#   Folder iterators
#
#   - iter_simple
#   - iter_double
#   - iter_multiple
#   - iter_seq
#   - iter_multi_seq
#
# ===========================================================================================================

def iter_simple( dr, mode, color, shuffle=True ):
    """ -----------------------------------------------------------------------------------------------------
    Simple generic dataset iterator

    https://keras.io/preprocessing/image/#imagedatagenerator-methods

    dr:             [str] folder of dataset (it must contain a subfolder for each class)
    mode:           [str] 'input' for dataset with input equal to target, None otherwise
    color:          [str] 'rgb' or 'grayscale'
    shuffle:        [bool] whether to shuffle the data

    return:         [keras_preprocessing.image.DirectoryIterator]
    ----------------------------------------------------------------------------------------------------- """

    # 'rescale' to normalize pixels in [0..1]
    idg     = preprocessing.image.ImageDataGenerator( rescale=1./255 )

    flow    = idg.flow_from_directory(
            directory   = dr,
            target_size = cnfg[ 'img_size' ][ :-1 ],
            color_mode  = color,
            class_mode  = mode,
            batch_size  = cnfg[ 'batch_size' ],
            shuffle     = shuffle,
            seed        = cnfg[ 'seed' ]
    )

    return flow



def iter_double( dir_in, dir_out, color_in, color_out ):
    """ -----------------------------------------------------------------------------------------------------
    Custom dataset iterator, combining two Iterators

    dir_in:         [str] folder of inputs
    dir_out:        [str] folder of target outputs (labels)
    color_in:       [str] 'rgb' or 'grayscale'
    color_out:      [str] 'rgb' or 'grayscale'

    return:         [generator] (python built-in type)
    ----------------------------------------------------------------------------------------------------- """
    flow_in     = iter_simple( dir_in, None, color_in, shuffle=True )
    flow_out    = iter_simple( dir_out, None, color_out, shuffle=True )

    if DEBUG0:
        cnt     = 0

    while True:
        inp     = flow_in.next()
        out     = flow_out.next()

        if DEBUG0:
            for i in range( cnfg[ 'batch_size' ] ):
                save_image( inp[ i ], True, "TEST/b{:03d}_i{:03d}.jpg".format( cnt, i ) )
                save_image( out[ i ], False, "TEST/b{:03d}_o{:03d}.jpg".format( cnt, i ) )
            cnt     += 1

        yield [ inp, out ]



def iter_multiple( dir_in, multi_dir_out, color_in, multi_color_out ):
    """ -----------------------------------------------------------------------------------------------------
    Custom dataset iterator, combining a single input Iterator and multiple output Iterators

    dir_in:             [str] folder of inputs
    multi_dir_out:      [list of str] folders of each target outputs (labels)
    color_in:           [str] 'rgb' or 'grayscale'
    multi_color_out:    [list of str] 'rgb' or 'grayscale' for each output class

    return:             [generator] (python built-in type)
    ----------------------------------------------------------------------------------------------------- """
    assert len( multi_dir_out ) == len( multi_color_out )

    flow_in             = iter_simple( dir_in, None, color_in, shuffle=True )
    multi_flow_out      = []

    for i in range( len( multi_dir_out ) ):
        multi_flow_out.append( iter_simple( multi_dir_out[ i ], None, multi_color_out[ i ], shuffle=True ) )

    if DEBUG0:
        cnt     = 0

    while True:
        inp     = flow_in.next()
        out     = [ fo.next() for fo in multi_flow_out ]

        if DEBUG0:
            for i in range( len( inp ) ):
                save_image( inp[ i ], True, "TEST/b{:03d}_i{:03d}.jpg".format( cnt, i ) )
                for j in range( len( multi_color_out ) ):
                    c   = multi_color_out[ j ] == 'rgb'
                    save_image( out[ j ][ i ], c, "TEST/b{:03d}_o{:03d}_{:1d}.jpg".format( cnt, i, j ))
            cnt     += 1

        yield [ inp, out ]



def iter_seq( multi_dir_in, dir_out ):
    """ -----------------------------------------------------------------------------------------------------
    Custom dataset iterator, combining multiple input Iterators and an output Iterator
    it assumes that all inputs and the output are of color type 'rgb'

    multi_dir_in:       [list of str] folders of each input
    dir_in:             [str] folder of target frame in the sequence

    return:             [generator] (python built-in type)
    ----------------------------------------------------------------------------------------------------- """

    multi_flow_in       = []
    flow_out            = iter_simple( dir_out, None, 'rgb', shuffle=True )

    for i in range( len( multi_dir_in ) ):
        multi_flow_in.append( iter_simple( multi_dir_in[ i ], None, 'rgb', shuffle=True ) )

    if DEBUG0:
        cnt     = 0

    while True:
        inp     = [ fi.next() for fi in multi_flow_in ]
        out     = inp.copy()
        out.append( flow_out.next() )

        if DEBUG0:
            for i,f in enumerate( out ):
                save_image( f, True, "TEST/b{:03d}_i{:03d}.jpg".format( cnt, i ) )
            cnt     += 1

        yield [ inp, out ]


def iter_multi_seq( multi_dir_in, multi_dir_out, multi_color_out ):
    """ -----------------------------------------------------------------------------------------------------
    Custom dataset iterator, combining multiple input Iterators and multiple output Iterators
    it assumes that all inputs are of color type 'rgb'

    multi_dir_in:       [list of str] folders of each input
    multi_dir_out:      [list of str] folders of each target outputs
    multi_color_out:    [list of str] 'rgb' or 'grayscale' for each output class

    return:             [generator] (python built-in type)
    ----------------------------------------------------------------------------------------------------- """

    multi_flow_in       = []
    multi_flow_out      = []

    for i in range( len( multi_dir_in ) ):
        multi_flow_in.append( iter_simple( multi_dir_in[ i ], None, 'rgb', shuffle=True ) )
    for i in range( len( multi_dir_out ) ):
        multi_flow_out.append( iter_simple( multi_dir_out[ i ], None, multi_color_out[ i ], shuffle=True ) )

    if DEBUG0:
        cnt     = 0

    while True:
        inp     = [ fi.next() for fi in multi_flow_in ]
        out     = [ fo.next() for fo in multi_flow_out ]


        if DEBUG0:
            for i in range( len( inp[ 0 ] ) ):
                for j in range( len( inp ) ):
                    save_image( inp[ j ][ i ], True, "TEST/b{:03d}_i{:03d}_{:1d}.jpg".format( cnt, i, j ) )
                for j, col in enumerate( multi_color_out ):
                    c   = col == 'rgb'
                    save_image( out[ j ][ i ], c, "TEST/b{:03d}_o{:03d}_{:1d}.jpg".format( cnt, i, j ))
            cnt     += 1

        yield [ inp, out ]



# ===========================================================================================================
#
#   Dataset iterators
#
#   - dset_same
#   - dset_class
#   - dset_multiple
#   - dset_sequence
#   - dset_multi_seq
#
# ===========================================================================================================

def dset_same( dir_dset ):
    """ -----------------------------------------------------------------------------------------------------
    Iterate over a training and a validation set, where target images are equal to input images

    dir_dset:       [str] folder of dataset (subfolders must include train/valid)

    return:         [list] of keras_preprocessing.image.DirectoryIterator
    ----------------------------------------------------------------------------------------------------- """
    color       = 'grayscale' if cnfg[ 'img_size' ][ -1 ] == 1 else 'rgb'

    train_dir   = os.path.join( dir_dset, 'train' )
    valid_dir   = os.path.join( dir_dset, 'valid' )

    train_flow  = iter_simple( train_dir, 'input', color, shuffle=True )
    valid_flow  = iter_simple( valid_dir, 'input', color, shuffle=True )

    return train_flow, valid_flow



def dset_class( dir_dset, data_class ):
    """ -----------------------------------------------------------------------------------------------------
    Iterate over a training and a validation set, where target images are different from input images

    dir_dset:       [str] folder of dataset (subfolders must include train/valid)
    data_class:     [str] name of the target class (car, lane)

    return:         [list of generator] (python built-in type)
    ----------------------------------------------------------------------------------------------------- """
    color_in        = 'grayscale' if cnfg[ 'img_size' ][ -1 ] == 1 else 'rgb'
    color_out       = 'grayscale'                                   # NOTE it always supposes grayscale target

    train_dir_in    = os.path.join( dir_dset, 'train', 'rgb' )
    train_dir_out   = os.path.join( dir_dset, 'train', data_class )
    valid_dir_in    = os.path.join( dir_dset, 'valid', 'rgb' )
    valid_dir_out   = os.path.join( dir_dset, 'valid', data_class )

    train_flow      = iter_double( train_dir_in, train_dir_out, color_in, color_out )
    valid_flow      = iter_double( valid_dir_in, valid_dir_out, color_in, color_out )

    return train_flow, valid_flow



def dset_multiple( dir_dset ):
    """ -----------------------------------------------------------------------------------------------------
    Iterate over a training and a validation set, where there are multiple classes of target images

    dir_dset:       [str] folder of dataset (subfolders must include train/valid)

    return:         [list of generator] (python built-in type)
    ----------------------------------------------------------------------------------------------------- """
    color_in        = 'rgb'                                         # NOTE it always supposes RGB input
    color_out       = [ 'rgb', 'grayscale', 'grayscale' ]

    train_dir_in    = os.path.join( dir_dset, 'train', 'rgb' )
    valid_dir_in    = os.path.join( dir_dset, 'valid', 'rgb' )

    output_classes  = [ 'rgb', 'car', 'lane' ]
    train_dir_out   = [ os.path.join( dir_dset, 'train', dc ) for dc in output_classes ]
    valid_dir_out   = [ os.path.join( dir_dset, 'valid', dc ) for dc in output_classes ]

    train_flow      = iter_multiple( train_dir_in, train_dir_out, color_in, color_out )
    valid_flow      = iter_multiple( valid_dir_in, valid_dir_out, color_in, color_out )

    return train_flow, valid_flow


def dset_sequence( dir_dset, seq_folders=('f0','f1','f2') ):
    """ -----------------------------------------------------------------------------------------------------
    Iterate over a training and a validation set, where there are multiple frames in sequence

    dir_dset:       [str] folder of dataset (subfolders must include train/valid)
    seq_folders:    [list of str] with names of subfolders of each frame in the sequence

    return:         [list of generator] (python built-in type)
    ----------------------------------------------------------------------------------------------------- """


    train_dir_in    = [ os.path.join( dir_dset, 'train', f ) for f in seq_folders[ : -1 ] ]
    valid_dir_in    = [ os.path.join( dir_dset, 'valid', f ) for f in seq_folders[ : -1 ] ]
    train_dir_out   = os.path.join( dir_dset, 'train', seq_folders[ -1 ] )
    valid_dir_out   = os.path.join( dir_dset, 'valid', seq_folders[ -1 ] )

    train_flow      = iter_seq( train_dir_in, train_dir_out )
    valid_flow      = iter_seq( valid_dir_in, valid_dir_out )

    return train_flow, valid_flow


def dset_multi_seq( dir_dset, seq_folders=('f0','f1','f2') ):
    """ -----------------------------------------------------------------------------------------------------
    Iterate over a training and a validation set, where there are multiple frames in sequence,
    each with multiple classes

    In the current use (as of 19 Sep 2019), the structure of the dataset folders is the following:
    dir_in:         [ f0/rgb, f1/rgb, f2/rgb ]
    dir_out:        [ f0/rgb, f0/car, f0/lane, f1/rgb, f1/car, f1/lane, f2/rgb, f2/car, f2/lane ]
    a corresponding structure of images is yield by the returned generators

    dir_dset:       [str] folder of dataset (subfolders must include train/valid)
    seq_folders:    [list of str] with names of subfolders of each frame in the sequence

    return:         [list of generator] (python built-in type)
    ----------------------------------------------------------------------------------------------------- """


    input_class     = 'rgb'
    output_classes  = [ 'rgb', 'car', 'lane' ]
    color_out       = [ 'rgb', 'grayscale', 'grayscale' ] * len( seq_folders )
    seq_in          = seq_folders[ : -1 ]
    train_dir_in    = [ os.path.join( dir_dset, 'train', f, input_class ) for f in seq_in ]
    valid_dir_in    = [ os.path.join( dir_dset, 'valid', f, input_class ) for f in seq_in ]
    train_dir_out   = [ os.path.join( dir_dset, 'train', s, c ) for s in seq_folders for c in output_classes ]
    valid_dir_out   = [ os.path.join( dir_dset, 'valid', s, c ) for s in seq_folders for c in output_classes ]

    train_flow      = iter_multi_seq( train_dir_in, train_dir_out, color_out )
    valid_flow      = iter_multi_seq( valid_dir_in, valid_dir_out, color_out )

    return train_flow, valid_flow



# ===========================================================================================================
#
#   - len_dataset
#   - gen_dataset
#
# ===========================================================================================================

def len_dataset( dir_dset ):
    """ -----------------------------------------------------------------------------------------------------
    Return the number of samples in each subset (train/valid/test) of a dataset

    dir_dset:       [str] folder of dataset (subfolders must include train/valid)

    return:         [list of int]
    ----------------------------------------------------------------------------------------------------- """
    train   = None
    valid   = None
    test    = None

    for dirpath, dirname, filename in os.walk( dir_dset ):
        if not dirname:
            if "train" in dirpath:
                train   = dirpath
            elif "valid" in dirpath:
                valid   = dirpath
            elif "test" in dirpath:
                test    = dirpath

    tr  = len( [ f for f in os.listdir( train ) if cond_ext( f ) ] )
    vl  = len( [ f for f in os.listdir( valid ) if cond_ext( f ) ] )
    ts  = len( [ f for f in os.listdir( test ) if cond_ext( f ) ] )

    return tr, vl, ts



def gen_dataset( dir_dset, seq_folders=('f0','f1','f2') ):
    """ -----------------------------------------------------------------------------------------------------
    Iterate over a dataset, yielding batches of images

    dir_dset:       [str] folder of dataset (it must contain subfolders for train/valid)
    seq_folders:    [list of str] with names of subfolders of each frame in the sequence (RGBSEQ only)

    return:         [keras_preprocessing.image.DirectoryIterator or generator] (python built-in type)
    ----------------------------------------------------------------------------------------------------- """
    if cnfg[ 'data_class' ] == 'RGB':
        return dset_same( dir_dset )

    if cnfg[ 'data_class' ] in ( 'LANE', 'CAR' ):
        return dset_class( dir_dset, cnfg[ 'data_class' ].lower() )

    if cnfg[ 'data_class' ] == 'MULTI':
        return dset_multiple( dir_dset )

    if cnfg[ 'data_class' ] == 'RGBSEQ':
        return dset_sequence( dir_dset, seq_folders=seq_folders )

    if cnfg[ 'data_class' ] == 'MULTISEQ':
        return dset_multi_seq( dir_dset, seq_folders=seq_folders )

    ms.print_err( "Data class {} not valid".format( cnfg[ 'data_class' ] ) )
