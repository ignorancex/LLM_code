"""
#############################################################################################################

Losses that do not need model compilation

all loss functions are computed using numpy, trying to match exactly the equivalent
Keras/TensorFlow functions

the losses are not averaged over the dimensions of the arrays, so that it is possible to evaluate
(and to visualize as images) the losses

    Alice   2019

#############################################################################################################
"""

import  os
import  numpy       as np

from    PIL         import Image
from    math        import sqrt, ceil, inf

cnfg            = []                    # NOTE: should be initialized by the calling module

p_ones_car      = 0
p_zeros_car     = 0
p_ones_lane     = 0
p_zeros_lane    = 0

_EPSILON        = 1e-7                  # as in Keras common.py

# ===========================================================================================================
# loss functions
#
# ===========================================================================================================

def get_unb_par( img_dir ):
    """ -----------------------------------------------------------------------------------------------------
    Validate the global parameters required in the unbalanced losse of a class

    img_dir:        [str] directory with the images used for the statistics of True/False pixels

    ----------------------------------------------------------------------------------------------------- """

    imgs    = [ f for f in os.listdir( img_dir ) if f.lower().endswith( ( '.png', '.jpg', '.jpeg' ) ) ]
    pix_img = cnfg[ 'img_size' ][ 0 ] * cnfg[ 'img_size' ][ 1 ]
    n_tot   = 0
    n_ones  = 0

    # compute percentage of white pixels over total number of pixels
    for i in imgs:
        f       = os.path.join( img_dir, i )
        im      = Image.open( f )
        ia      = np.array( im )
        n_ones  += ( ia == 255 ).sum()
        im.close()
        n_tot   +=  pix_img

    p_ones  = ( float( n_ones ) / n_tot ) ** ( 1/4 )        # percentage of white pixels
    p_zeros = 1 - p_ones                                    # percentage of black pixels

    return p_ones, p_zeros


def get_unb_pars():
    """ -----------------------------------------------------------------------------------------------------
    Validate the global parameters required in the unbalanced losses

    ----------------------------------------------------------------------------------------------------- """
    global p_ones_car, p_zeros_car, p_ones_lane, p_zeros_lane

    dr                          = os.path.join( cnfg[ 'dir_dset' ], 'valid', 'car', 'img' )
    p_ones_car, p_zeros_car     = get_unb_par( dr )
    dr                          = os.path.join( cnfg[ 'dir_dset' ], 'valid', 'lane', 'img' )
    p_ones_lane, p_zeros_lane   = get_unb_par( dr )


def mse( y_true, y_pred ):
    """ -----------------------------------------------------------------------------------------------------
    compute mean squared error
    y_true:         [np.array] target
    y_pred:         [np.array] model output

    return:         [np.array] loss
    ----------------------------------------------------------------------------------------------------- """

    assert y_true.shape == y_pred.shape

    return np.square( y_pred - y_true )


def xentropy( y_true, y_pred ):
    """ -----------------------------------------------------------------------------------------------------
    compute sigmoid cross entropy
    y_true:         [np.array] target
    y_pred:         [np.array] model output

    equivalent to sigmoid_cross_entropy_with_logits() in tensorflow/python/ops/nn_impl.py
    with preliminary conversion from probability to logits as in K.binary_crossentropy()

    return:         [np.array] loss
    ----------------------------------------------------------------------------------------------------- """

    assert y_true.shape == y_pred.shape

    y_pred  = np.clip( y_pred, _EPSILON, 1 - _EPSILON )
    y_pred  = np.log( y_pred / ( 1 - y_pred ) )
    zeros   = np.zeros_like( y_pred )
    cond    = ( y_pred >= zeros )
    rl_pred = np.where( cond, y_pred, zeros )
    na_pred = np.where( cond, -y_pred, y_pred )
    loss    = rl_pred - y_pred * y_true
    loss    += np.log1p( np.exp( na_pred ) )

    return loss


def unb_car( y_true, y_pred ):
    """ -----------------------------------------------------------------------------------------------------
    compute unbalanced loss for car
    y_true:         [np.array] target
    y_pred:         [np.array] model output

    return:         [np.array] loss
    ----------------------------------------------------------------------------------------------------- """

    assert y_true.shape == y_pred.shape

    if not p_zeros_car:
        get_unb_pars()
    w   = p_zeros_car * y_true + p_ones_car * ( 1. - y_true )
    l   = xentropy( y_true, y_pred )
    return w * l


def unb_lane( y_true, y_pred ):
    """ -----------------------------------------------------------------------------------------------------
    compute unbalanced loss for lane
    y_true:         [np.array] target
    y_pred:         [np.array] model output

    return:         [np.array] loss
    ----------------------------------------------------------------------------------------------------- """

    assert y_true.shape == y_pred.shape

    if not p_zeros_lane:
        get_unb_pars()
    w   = p_zeros_lane * y_true + p_ones_lane * ( 1. - y_true )
    l   = xentropy( y_true, y_pred )
    return w * l


# ===========================================================================================================
# auxiliary functions
#
# ===========================================================================================================


def array_to_image( array, rgb=False, normalize=True ):
    """ -----------------------------------------------------------------------------------------------------
    Convert numpy.ndarray to PIL.Image, optionally reduce from RGB to grayscale

    array:          [numpy.ndarray] pixel values (between 0..1)
    rgb:            [bool] True if RGB, false if grayscale
    normalize:      map the full range of array values to 0..255

    return:         [PIL.Image.Image]
    ----------------------------------------------------------------------------------------------------- """
    if len( array.shape ) == 4:
        array   = array[ 0, :, :, : ]                               # remove batch axis

    if len( array.shape ) == 2:
        rgb     = False
        pixels  = array
    else:
        colors  = array.shape[ -1 ]
        if not rgb:
            if colors == 3:
                pixels  = array.mean( axis=-1 )
            else:
                pixels  = array[ :, :, 0 ]
        else:
            pixels  = array

    if normalize:
        ptp     = pixels.ptp()
        if ptp:
            pixels  = ( pixels - pixels.min() ) / ptp
    pixels  = 255. * pixels
    pixels  = np.uint8( pixels )

    if rgb:
        img     = Image.fromarray( pixels, 'RGB' )

    else:
        img     = Image.fromarray( pixels )
        img     = img.convert( 'RGB' )

    return img



def image_to_array( img ):
    """ -----------------------------------------------------------------------------------------------------
    Convert PIL.Image to numpy.ndarray with the shape required by tensorflow/Keras

    img:            [str] path to image file

    return:         [numpy.ndarray] pixel values (between 0..1) with an extra (batch) dimensions
    ----------------------------------------------------------------------------------------------------- """
    if not os.path.isfile( img ):
        raise ValueError( "image {} not found".format( img ) )

    im      = Image.open( img )
    ia      = np.array( im )
    ia      = ia / 255.
    if len( ia.shape ) == 2:
        ia      = np.expand_dims( ia, axis=-1 )
    ia      = np.expand_dims( ia, axis=0 )
    im.close()

    return ia


def save_loss( array, fname="loss.jpg" ):
    """ -----------------------------------------------------------------------------------------------------
    Save an image, resulting from loss computation, to file

    array:          [numpy.ndarray]
    fname:          [str] path of output file
    ----------------------------------------------------------------------------------------------------- """
    if not isinstance( array, np.ndarray ):
        raise ValueError( "input to save_loss should be a numpy ndarray" )

    norm    = array.max() > 1.0 or array.min() < 0.0
    img     = array_to_image( array, rgb=False, normalize=norm )
    img.save( fname )

