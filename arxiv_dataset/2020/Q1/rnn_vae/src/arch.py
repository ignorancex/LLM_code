"""
#############################################################################################################

Definition of the neural network architecture

    Alice   2019

#############################################################################################################
"""

import  os
import  re
import  numpy               as np
import  tensorflow          as tf

from    abc                 import ABC, abstractmethod
from    distutils.version   import LooseVersion
from    keras               import models, layers, utils, initializers, regularizers, optimizers
from    keras               import metrics, losses, preprocessing
from    keras               import backend      as K

import  mesg                as ms
import  h5lib               as hl


PLOT            = True
TRAIN           = True
DEBUG           = False

# Range of values (absolute) for latents.
# The following constants are computed with inspection.m on RMVAE_19-10-01_13-20-08, and are the
# double of the max variance on the components of the latent vectors car/RGB/lane
# NOTE that if latent has split!=2 RBGRANGE is applied to the entire vector, while in the case split=2
# RBGRANGE is applied only to the inner part of the vector, car/lane excluded
RBGRANGE        = 2.4
CARRANGE        = 2.8
LANRANGE        = 3.6

k_unbalance     = 1. / 4                # exponential used in computing the unbalanced loss
perc_ones       = {                     # precomputed percentage of white pixels in images
#   the commented values are computed on subset 'valid', the others on 'train'
#   "dataset/synthia/link/mseq12_L" :   { 'CAR' : 0.04919, 'LANE' : 0.03956 }
    "dataset/synthia/link/mseq12_L" :   { 'CAR' : 0.05000, 'LANE' : 0.03955 },
    "dataset/synthia/link/mseq12_S" :   { 'CAR' : 0.05000, 'LANE' : 0.03955 },
    "dataset/synthia/link/multi_L"  :   { 'CAR' : 0.05000, 'LANE' : 0.03955 },
    "dataset/synthia/link/multi_S"  :   { 'CAR' : 0.05000, 'LANE' : 0.03955 }
}

nn_wght         = 'nn_wght.h5'          # filename of model weights
nn_arch         = 'nn_arch.json'        # filename of model architecture
dir_model       = 'models'              # folder containing pre-trained models
dir_plot        = 'plot'                # folder containing the architecture plots
plot_ext        = '.png'                # file format of the plot (pdf/png/jpg)

cnfg            = []                    # NOTE initialized by 'nn_main.py'

layer_code      = {                     # codes specifying the type of layer, passed to 'cnfg.py'
        'conv'  : 'C',
        'dcnv'  : 'T',
        'dnse'  : 'D',
        'flat'  : 'F',
        'pool'  : 'P',
        'rshp'  : 'R',
        'stop'  : '-'
}


# ===========================================================================================================
#
#   - stanh
#
#   - get_loss
#   - get_unb_loss
#   - jaccard_loss
#
#   - get_optim
#   - model_summary
#
#   - load_model
#   - save_model
#   - create_model
#
# ===========================================================================================================

def stanh( x ):
    """ -------------------------------------------------------------------------------------------------
    Scaled tanh

    x:              [tf.Tensor]

    return:         [tf.Tensor]
    ------------------------------------------------------------------------------------------------- """
    return RBGRANGE * K.tanh( x )



def get_loss( l ):
    """ -------------------------------------------------------------------------------------------------
    Return a Loss function according to the passed code

    l:              [str] code of loss function

    return:         [function]
    ------------------------------------------------------------------------------------------------- """
    if l == 'MSE':
        return losses.mean_squared_error
    if l == 'BXE':
        return losses.binary_crossentropy
    if l == 'CXE':
        return losses.categorical_crossentropy
    if l == 'UXE_CAR':
        return get_unb_loss( 'CAR' )
    if l == 'UXE_LANE':
        return get_unb_loss( 'LANE' )
    if l == 'JAC':
        return jaccard_loss
    
    ms.print_err( "Loss {} not valid".format( l ) )



def pre_unb_ones( data_class=None, sub='valid' ):
    """ -----------------------------------------------------------------------------------------------------
    Precompute the percentage of "one" values (white pixels) in a dataset
    The argument 'data_class' is used for evaluation how unbalanced a specific dataset is.
    If None is passed, it considers the data_class specified in the config file.

    data_class:     [str] 'CAR' or 'LANE', None otherwise
    sub:            [str] subfolder in the dataset on which the percentage is computed

    return:         [float] percentage of "one" values
    ----------------------------------------------------------------------------------------------------- """
    if cnfg[ 'arch_class' ] == 'RMVAE':
        sub     += '/f1'
    if data_class is None:
        dr      = os.path.join( cnfg[ 'dir_dset' ], sub, cnfg[ 'data_class' ].lower(), 'img' )
    else:
        dr      = os.path.join( cnfg[ 'dir_dset' ], sub, data_class.lower(), 'img' )

    if not os.path.isdir( dr ):
        raise ValueError( "{} directory does not exist".format( dr ) )

    imgs    = [ f for f in os.listdir( dr ) if f.lower().endswith( ( '.png', '.jpg', '.jpeg' ) ) ]
    pix_img = cnfg[ 'img_size' ][ 0 ] * cnfg[ 'img_size' ][ 1 ]
    n_tot   = 0
    n_ones  = 0

    # compute percentage of white pixels over total number of pixels
    for i in imgs:
        f       = os.path.join( dr, i )
        im      = preprocessing.image.load_img( f, color_mode = 'grayscale', target_size=cnfg[ 'img_size' ][ :-1 ] )
        im      = preprocessing.image.img_to_array( im )
        n_ones  += ( im == 255 ).sum()
        n_tot   +=  pix_img

    return  float( n_ones ) / n_tot



def get_unb_loss( data_class=None ):
    """ -----------------------------------------------------------------------------------------------------
    Setup a binary crossentropy loss function for evaluating highly unbalanced binary images
    (very few white pixels on black background)

    The argument 'data_class' is used for evaluation how unbalanced a specific dataset is.
    If None is passed, it considers the data_class specified in the config file.

    The function first check if a precomputed value of the percentage of "one" values in the datast
    is available, otherwise it is computed (and typically takes a while)

    data_class:     [str] 'CAR' or 'LANE', None otherwise

    return:         [fuction] loss function
    ----------------------------------------------------------------------------------------------------- """
    if data_class is None:
        data_class  = cnfg[ 'data_class' ]

    if cnfg[ 'dir_dset' ] in perc_ones:
        p_ones      = perc_ones[ cnfg[ 'dir_dset' ] ][ data_class ]
    else:
        p_ones      = pre_unb_ones( data_class=data_class )

    p_ones  = p_ones ** k_unbalance                         # normalized percentage of white pixels
    p_zeros = 1 - p_ones                                    # normalized percentage of black pixels

    def unbalanced_loss( y_true, y_pred ):
        y_true  = tf.reshape( y_true, [ -1 ], name="y_true_flat" )
        y_pred  = tf.reshape( y_pred, [ -1 ], name="y_pred_flat" )

        w   = p_zeros * y_true + p_ones * ( 1. - y_true )   # compute weights from inverse probability
        # l   = losses.binary_crossentropy( y_true, y_pred )  # ordinary binary crossentropy
        l   = K.binary_crossentropy( y_true, y_pred )       # ordinary binary crossentropy
        u   = w * l                                         # unbalanced loss
        return K.mean( u )

    return unbalanced_loss



def jaccard_loss( y_true, y_pred, smooth=100 ):
    """ -----------------------------------------------------------------------------------------------------
    The jaccard distance is useful for unbalanced datasets. This has been shifted so it converges
    on 0 and is smoothed to avoid exploding or disapearing gradient.
    
    Code taken from: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96

    y_true:         [tf.Tensor] target image
    y_pred:         [tf.Tensor] output image

    return:         [tf.Tensor] loss
    ----------------------------------------------------------------------------------------------------- """
    i   = K.sum( K.abs( y_true * y_pred ), axis=-1 )
    s   = K.sum( K.abs( y_true ) + K.abs( y_pred ), axis=-1 )
    j   = ( i + smooth ) / ( s - i + smooth )
    return ( 1 - j ) * smooth



def get_optim():
    """ -------------------------------------------------------------------------------------------------
    Return an Optimizer according to the object attribute

    return:         [keras.optimizers.Optimizer]
    ------------------------------------------------------------------------------------------------- """
    if cnfg[ 'optimizer' ] == 'ADAGRAD':
        return optimizers.Adagrad( lr=cnfg[ 'lrate' ] )
    if cnfg[ 'optimizer' ] == 'SDG':
        return optimizers.SGD( lr=cnfg[ 'lrate' ] )
    if cnfg[ 'optimizer' ] == 'RMS':
        return optimizers.RMSprop( lr=cnfg[ 'lrate' ] )
    if cnfg[ 'optimizer' ] == 'ADAM':
        return optimizers.Adam( lr=cnfg[ 'lrate' ] )
    
    ms.print_err( "Optimizer {} not valid".format( cnfg[ 'optimizer' ] ) )
        

    
def model_summary( model, fname='model' ):
    """ -------------------------------------------------------------------------------------------------
    Print a summary of the model, and plot a graph of the model and save it to a file

    model:          [keras.engine.training.Model]
    fname:          [str] name of the output image without extension
    ------------------------------------------------------------------------------------------------- """
    if PLOT:
        utils.print_summary( model )

        d   = os.path.join( cnfg[ 'dir_current' ], dir_plot )
        if not os.path.exists( d ):
            os.makedirs( d )

        f   = os.path.join( d, fname + plot_ext )

        #utils.plot_model( model, to_file=f, show_shapes=True, show_layer_names=True, expand_nested=True )
        utils.plot_model( model, to_file=f, show_shapes=True, show_layer_names=True )



def load_model( model, ref_model ):
    """ -----------------------------------------------------------------------------------------------------
    Initialize the weigths of given model usign the weigths of another reference model.
    The two models should have compatible architectures.

    If a folder is passed, it should contain the two files HDF5 and JSON.
    If a single file is passed, it is considered as model+weights HDF5 file

    model:          [keras.engine.training.Model] current model
    ref_model:      [str] folder or filename of the reference model

    return:         [bool] False if the loading process fails
    ----------------------------------------------------------------------------------------------------- """
    if ref_model.endswith( '.h5' ):         # single file
        h5  = ref_model
    else:                                   # folder
        h5  = os.path.join( ref_model, nn_wght )

    return hl.load_h5( model, h5 )



def load_vgg16( model ):
    """ -----------------------------------------------------------------------------------------------------
    Load weights of the VGG16 model trained on ImageNet, and put them in the corresponding encoder
    part of the given model.

    The code is extracted from
        /opt/local/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages
            /keras_applications/vgg16.py

    model:          [keras.engine.training.Model] current model

    return:         [bool] False if the loading process fails
    ----------------------------------------------------------------------------------------------------- """
    if not os.path.exists( os.path.join( os.getcwd(), dir_model ) ):
        os.makedirs( os.path.join( os.getcwd(), dir_model ) )

    file_orig       = ( 'https://github.com/fchollet/deep-learning-models/'
                        'releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5' )
    file_dest       = os.path.join( os.getcwd(), dir_model, 'vgg16_notop.h5' )
    cache_dir       = 'models'
    file_hash       = '6d6bbae143d832006294945121d1f1fc'

    h5              = utils.get_file(
            file_dest,
            file_orig,
            cache_subdir    = cache_dir,
            file_hash       = file_hash
    )
    return hl.load_h5_vgg16( model, h5 )

    
    
def save_model( model ):
    """ -----------------------------------------------------------------------------------------------------
    Save a trained model in two files: one file for the architecture (JSON) and one for the weights (HDF5)

    model:          [keras.engine.training.Model]
    ----------------------------------------------------------------------------------------------------- """
    model.save_weights( os.path.join( cnfg[ 'dir_current' ], nn_wght ) )

    with open( os.path.join( cnfg[ 'dir_current' ], nn_arch ), 'w' ) as f:
        f.write( model.to_json() )



def create_model():
    """ -----------------------------------------------------------------------------------------------------
    Create the model of the neural network

    return:         [keras.engine.training.Model]
    ----------------------------------------------------------------------------------------------------- """
    if cnfg[ 'arch_class' ] == 'AE':
        nn  = AE()
    elif cnfg[ 'arch_class' ] == 'VAE':
        nn  = VAE()
    elif cnfg[ 'arch_class' ] == 'MVAE':
        nn  = MultipleVAE()
    elif cnfg[ 'arch_class' ] == 'MAE':
        nn  = MultipleAE()
    elif cnfg[ 'arch_class' ] == 'RAE':
        nn  = RecAE()
    elif cnfg[ 'arch_class' ] == 'RVAE':
        nn  = RecVAE()
    elif cnfg[ 'arch_class' ] == 'RMVAE':
        nn  = RecMultiVAE()
    elif cnfg[ 'arch_class' ] == 'RMAE':
        nn  = RecMultiAE()
    elif cnfg[ 'arch_class' ] == 'TIME':
        nn  = Timepred()
    elif cnfg[ 'arch_class' ] == 'RTIME':
        nn  = RecTimepred( lname='pure_rnn_1' )     # FIXME DO NOT pass here different name of layer, but in R*AE instead
    elif cnfg[ 'arch_class' ] == 'R2TIME':
        nn  = Rec2Timepred( lname=('rnn_layer_1', 'rnn_layer_2') )
    elif cnfg[ 'arch_class' ] == 'RMTIME':
        nn  = RecMTimepred()
    else:
        ms.print_err( "Architecture {} not valid".format( cnfg[ 'arch_class' ] ) )

    nn.model.compile(
            optimizer       = get_optim(),
            loss            = nn.loss_func,
            loss_weights    = nn.loss_wght if hasattr( nn, 'loss_wght' ) else None
    )

    return nn



# ===========================================================================================================
#
#   Classes
#
#   Superclasses:
#   - Autoencoder
#   - Multiple
#   - Recursive
#
#   Basic autoencoder classes:
#   - AE            (AE)
#   - VAE           (VAE)
#
#   Classes with multiple decoders:
#   - MultipleAE    (MAE)
#   - MultipleVAE   (MVAE)
#
#   Classes for time prediction in latent space:
#   - Timepred      (TIME)
#   - RecTimepred   (RTIME)
#   - Rec2Timepred  (R2TIME)
#   - RecMTimepred  (RMTIME)
#
#   Classes for autoencoders that force predictibility:
#   - RecAE         (RAE)
#   - RecVAE        (RVAE)
#   - RecMultiVAE   (RMVAE)
#   - RecMultiAE    (RMAE)
#
# ===========================================================================================================

class Autoencoder( ABC ):
    """ -----------------------------------------------------------------------------------------------------
    (almost) Abstract class for the generation of a neural network with encoding/decoding structure
    ----------------------------------------------------------------------------------------------------- """

    @abstractmethod
    def define_model( self ):
        pass


    def _get_init( self ):
        """ -------------------------------------------------------------------------------------------------
        Return an Initializer according to the object attribute

        return:         [keras.initializers.Initializer]
        ------------------------------------------------------------------------------------------------- """
        if self.k_initializer == 'RUNIF':
            return initializers.RandomUniform( minval=-0.05, maxval=0.05, seed=cnfg[ 'seed' ] )
        
        if self.k_initializer == 'GLOROT':
            return initializers.glorot_normal( seed=cnfg[ 'seed' ] )

        if self.k_initializer == 'HE':
            return initializers.he_normal( seed=cnfg[ 'seed' ] )

        ms.print_err( "Initializer {} not valid".format( self.k_initializer ) )


    def _get_config( self ):
        """ -------------------------------------------------------------------------------------------------
        get class attributes from the configuration, allowing backward compatibility for specific attributes
        introduced in newer versions:
            - dnse_dropout

        ------------------------------------------------------------------------------------------------- """
        for k in self.__dict__:
            if k in cnfg:
                exec( "self.{} = cnfg[ '{}' ]".format( k, k ) )                              
            elif k == "dnse_dropout":
# it would be sensible to have a list with self.dnse_size 0s, but we cannot be sure that
# self.dnse_size has been initialized, so just use a large enough list of 0s
                self.dnse_dropout   = 10 * [ 0 ]
            else:
                ms.print_err( "Attribute '{}' of class '{}' not indicated".format( k, self.__class__ ) )



    def _get_regul( self ):
        """ -------------------------------------------------------------------------------------------------
        Return a Regularizer according to the object attribute

        return:         [keras.regularizers.Regularizer]
        ------------------------------------------------------------------------------------------------- """
        if self.k_regularizer == 'L2':
            return regularizers.l2( 0 )     # FIXME REGUL WITH FACTOR=0 MAKES NO SENSE!

        if self.k_regularizer == 'NONE':
            return None

        ms.print_err( "Regularizer {} not valid".format( self.k_regularizer ) )



    def __init__( self ):
        """ -------------------------------------------------------------------------------------------------
        Initialize class attributes usign the config dicitonary, and create the network model.

        When using multi_gpu_model, Keras recommends to instantiate the model under a CPU device scope,
        so that the model's weights are hosted on CPU memory.
        Otherwise they may end up hosted on a GPU, which would complicate weight sharing.

        https://keras.io/utils/#multi_gpu_model
        ------------------------------------------------------------------------------------------------- """
        self.arch_layout        = None          # [str] code describing the order of layers in the model

        self.loss               = None          # [str] code specifying the type of loss function
        self.img_size           = None          # [list] use 'channels_last' convention
        self.ref_model          = None          # [str] optional reference model from which to load weights
        self.k_initializer      = None          # [str] code specifying the type of convolution initializer
        self.k_regularizer      = None          # [str] code specifying the type of convolution regularizer

        self.conv_filters       = None          # [list of int] number of kernels for each convolution
        self.conv_kernel_size   = None          # [list of int] (square) size of kernels for each convolution
        self.conv_strides       = None          # [list of int] stride for each convolution
        self.conv_padding       = None          # [list of str] padding (same/valid) for each convolution
        self.conv_activation    = None          # [list of str] activation function for each convolution
        self.conv_train         = None          # [list of bool] False to lock training of each convolution

        self.pool_size          = None          # [list of int] pooling size for each MaxPooling

        self.dnse_size          = None          # [list of int] size of each dense layer
        self.dnse_activation    = None          # [list of str] activation function for each dense layer
        self.dnse_train         = None          # [list of bool] False to lock training of each dense layer

        self.dcnv_filters       = None          # [list of int] number of kernels for each deconvolution
        self.dcnv_kernel_size   = None          # [list of int] (square) size of kernels for each deconvolution
        self.dcnv_strides       = None          # [list of int] stride for each deconvolution
        self.dcnv_padding       = None          # [list of str] padding (same/valid) for each deconvolution
        self.dcnv_activation    = None          # [list of str] activation function for each deconvolution
        self.dcnv_train         = None          # [list of bool] False to lock training of each deconvolution

        if DEBUG: print ( "\nstart Autoencoder.__init__()" )

        super( Autoencoder, self ).__init__()   # to keep the super() chain
        # initialize class attributes with values from cnfg dict
        self._get_config()

        # check if the string defining the architecture layout contains incorrect chars
        s0      = set( layer_code.values() )    # accepted chars
        s1      = set( self.arch_layout )       # chars passed as config
        if s1 - s0:
            ms.print_err( "Incorrect code {} for architecture layout".format( self.arch_layout ) )

        # check if the architecture layout is well defined
        if self.arch_layout.count( layer_code[ 'flat' ] ) > 1:
            ms.print_wrn( "Multiple flatten layer found. The architecture may be ill-defined" )
        if self.arch_layout.count( layer_code[ 'rshp' ] ) > 1:
            ms.print_wrn( "Multiple reshape layer found. The architecture may be ill-defined" )
 
        # keep a global count of layers per kind, to ensure different names for layers that are at
        # the same level in the architecture, but in different branches
        self.i_conv             = 1
        self.i_pool             = 1
        self.i_dnse             = 1
        self.i_dcnv             = 1

        # create model
        if cnfg[ 'n_gpus' ] > 1:
            with tf.device( '/cpu:0' ):
                self.model  = self.define_model( mname=cnfg[ 'arch_class' ] )
        else:
            self.model  = self.define_model( mname=cnfg[ 'arch_class' ] )

        assert cnfg[ 'n_conv' ] == len( self.conv_filters ) == len( self.conv_kernel_size ) == \
                len( self.conv_strides ) == len( self.conv_padding ) == len( self.conv_activation ) == \
                len( self.conv_train )

        assert cnfg[ 'n_dcnv' ] == len( self.dcnv_filters ) == len( self.dcnv_kernel_size ) == \
                len( self.dcnv_strides ) == len( self.dcnv_padding ) == len( self.dcnv_activation ) == \
                len( self.dcnv_train )

        assert cnfg[ 'n_dnse' ] == len( self.dnse_size ) == len( self.dnse_activation ) == len( self.dnse_train )
        assert cnfg[ 'n_pool' ] == len( self.pool_size )

        # load from a possible reference model
        if self.ref_model == 'VGG16':
            if not load_vgg16( self.model ):
                ms.print_err( "Failed to load weights from VGG16" )
        elif self.ref_model is not None:
            if not load_model( self.model, self.ref_model ):
                ms.print_err( "Failed to load weights from {}".format( self.ref_model ) )

        model_summary( self.model, fname=cnfg[ 'arch_class' ] )

        if DEBUG: print ( "\nend Autoencoder.__init__()" )



    def define_basic_encoder( self, mname='encoder' ):
        """ -------------------------------------------------------------------------------------------------
        Define the structure of the encoder part of the network
        Note: the input tensor is pointed by the new variable self.encoder_input

        mname:          [str] name of the model

        return:         [tf.Tensor] output of the encoder
        ------------------------------------------------------------------------------------------------- """
        i_conv, i_pool, i_dnse  = 3 * [ 0 ]                         # for keeping count
        enc_arch_layout         = self.arch_layout.split( layer_code[ 'stop' ] )[ 0 ]

        init        = self._get_init() 
        kreg        = self._get_regul()

        # INPUT LAYER of encoder part
        self.encoder_input  = layers.Input( shape=self.img_size )   # height, width, channels
        x                   = self.encoder_input

        for i, layer in enumerate( enc_arch_layout ):

            # CONVOLUTIONAL LAYER
            if layer == layer_code[ 'conv' ]:
                x       = layers.Conv2D(
                    self.conv_filters[ i_conv ],                            # number of filters
                    kernel_size         = self.conv_kernel_size[ i_conv ],  # size of window
                    strides             = self.conv_strides[ i_conv ],      # stride (window shift)
                    padding             = self.conv_padding[ i_conv ],      # zero-padding around the image
                    activation          = self.conv_activation[ i_conv ],   # activation function
                    kernel_initializer  = init,
                    kernel_regularizer  = kreg,                             # TODO check also activity_regularizer
                    use_bias            = True,                             # TODO watch out for the biases!
                    trainable           = self.conv_train[ i_conv ],
                    name                = 'conv{}'.format( self.i_conv )
                )( x )
                i_conv      += 1
                self.i_conv += 1

            # MAX POOLING LAYER
            elif layer == layer_code[ 'pool' ]:
                x       = layers.MaxPooling2D(                          
                    pool_size       = self.pool_size[ i_pool ],             # pooling size
                    padding         = self.conv_padding[ i_pool ],          # zero-padding around the image
                    name            = 'pool{}'.format( self.i_pool )
                )( x )
                i_pool      += 1
                self.i_pool += 1

            # DENSE LAYERs
            elif layer == layer_code[ 'dnse' ]:
                if self.dnse_dropout[ i_dnse ]:
                    x           = layers.Dropout( self.dnse_dropout[ i_dnse ] )( x )
                x           = layers.Dense(                             
                    self.dnse_size[ i_dnse ],                              # dimensionality of the output
                    activation      = self.dnse_activation[ i_dnse ],       # activation function
                    trainable       = self.dnse_train[ i_dnse ],
                    name            = 'dnse{}'.format( self.i_dnse )
                )( x )
                i_dnse      += 1
                self.i_dnse += 1

            # FLATTEN LAYER
            # NOTE it supposes a single flatten layer in the architecture
            elif layer == layer_code[ 'flat' ]:

                if self.dnse_size[ -1 ] is None:
                    # save the shape after last convolution and match it with the shape of last dense layer
                    self.last_shape         = K.int_shape( x )[ 1: ]
                    self.dnse_size[ -1 ]    = np.prod( self.last_shape )

                x       = layers.Flatten( name='flat' )( x )

            else:
                ms.print_err( "Layer code '{}' not valid for {} architecture".format( layer, cnfg[ 'arch_class' ] ) )

        return x



    def define_basic_decoder( self, input_size, mname='decoder' ):
        """ -------------------------------------------------------------------------------------------------
        Define the structure of the decoder part of the network
        Note: the input tensor is pointed by the new variable self.decoder_input

        input_size:     [tuple] size of the input of the decoder
        mname:          [str] name of the model

        return:         [tf.Tensor] output of the decoder
        ------------------------------------------------------------------------------------------------- """
        enc_arch_layout, dec_arch_layout    = self.arch_layout.split( layer_code[ 'stop' ] )
        i_dcnv              = 0
        i_dnse              = enc_arch_layout.count( layer_code[ 'dnse' ] )     # resume count of dense from encoder part

        # INPUT LAYER of decoder part
        self.decoder_input  = layers.Input( input_size )
        x                   = self.decoder_input

        for layer in dec_arch_layout:

            # DECONVOLUTIONAL LAYERs
            if layer == layer_code[ 'dcnv' ]:
                x       = layers.Conv2DTranspose(                           # TODO consider to use init & regul
                    self.dcnv_filters[ i_dcnv ],                            # number of filters
                    kernel_size     = self.dcnv_kernel_size[ i_dcnv ],      # size of window
                    strides         = self.dcnv_strides[ i_dcnv ],          # stride
                    padding         = self.dcnv_padding[ i_dcnv ],          # zero-padding
                    activation      = self.dcnv_activation[ i_dcnv ],       # activation function
                    use_bias        = False,                                # TODO watch out for the biases!
                    trainable       = self.dcnv_train[ i_dcnv ],
                    name            = 'dcnv{}'.format( self.i_dcnv )
                )( x )
                i_dcnv      += 1
                self.i_dcnv += 1

            # DENSE LAYERs
            elif layer == layer_code[ 'dnse' ]:
                x           = layers.Dense(                             
                    self.dnse_size[ i_dnse ],                               # dimensionality of the output
                    activation      = self.dnse_activation[ i_dnse ],       # activation function
                    trainable       = self.dnse_train[ i_dnse ],
                    name            = 'dnse{}'.format( self.i_dnse )
                )( x )
                i_dnse      += 1
                self.i_dnse += 1

            # RESHAPE LAYER
            # NOTE it supposes a single reshape layer in the architecture
            elif layer == layer_code[ 'rshp' ]:

                if hasattr( self, 'last_shape' ):
                    ts  = self.last_shape
                else:
                    # NOTE it supposes an aspect ratio of 2:1 for the input images of the dataset
                    assert self.img_size[ 1 ] / self.img_size[ 0 ] == 2

                    dn  = self.dnse_size[ -1 ]                              # current flat size
                    ch  = self.dcnv_filters[ 0 ]                            # desired num of channels
                    hg  = int( ( dn / ( 2 * ch ) ) ** ( 1/2 ) )             # desired height 
                    wd  = int( 2 * hg )                                     # desided width
                    ts  = [ hg, wd, ch ]

                x       = layers.Reshape(
                    target_shape    = ts,                                   # new shape (height, width, channels)
                    name            = 'rshp'
                )( x )

            else:
                ms.print_err( "Layer code '{}' not valid for {} architecture".format( layer, cnfg[ 'arch_class' ] ) )
                
        return x



# ===========================================================================================================



class Multiple( ABC ):
    """ -----------------------------------------------------------------------------------------------------
    Abstract class for multiple decoding branches
    Each decoding branch takes as input a slice of the latent space.

    This class should be inherited by a class that subclasses also Autoencoder or any subclass of it
    ----------------------------------------------------------------------------------------------------- """

    def __init__( self ):
        """ -------------------------------------------------------------------------------------------------
        Initialize class attributes usign the config dicitonary, and create the network model
        ------------------------------------------------------------------------------------------------- """
        self.latent_subsize             = None      # [int] size of the overlap in the latent space
        self.split                      = None      # [int] code describing the type of split of the latent space
        if DEBUG: print ( "\nstart Multiple.__init__()" )
        super( Multiple, self ).__init__()          # to keep the super() chain
        if DEBUG: print ( "\nend Multiple.__init__()" )


    def latent_parts( self, split ):
        """ -------------------------------------------------------------------------------------------------
        compute the proportion of the latent space for the rgb and the segmentation components
        perform a consistency check of latent_size for the different splitting configurations

        split:              [int] index for choosing different ways of splitting the tensor
                            see split_latent for the meaning of the code
        return:             [tuple] size of rgb component, size of segm component
        ------------------------------------------------------------------------------------------------- """

        if split == 0:
            assert not self.latent_size % 3
            s_rgb       = self.latent_size // 3
            s_segm      = self.latent_size // 3
            return s_rgb, s_segm

        if split == 1:
            assert not self.latent_size % 2
            assert not self.latent_subsize % 2
            assert self.latent_size >= 2 * self.latent_subsize
            s_rgb       = self.latent_subsize
            s_segm      = ( self.latent_size + self.latent_subsize ) // 2
            return s_rgb, s_segm

        if split == 2:
            s_rgb       = self.latent_size
            s_segm      = self.latent_subsize
            return s_rgb, s_segm

        if split == 3:
            s_rgb       = self.latent_size
            s_segm      = self.latent_size
            return s_rgb, s_segm

        ms.print_err( "Split code {} not valid".format( split ) )
        return False


    def split_latent( self, encoder_output, split ):
        """ -------------------------------------------------------------------------------------------------
        Split the latent space in 3 components, to be used as input by the 3 decoding branches.
        The 3 components do NOT always have the same size.

        encoder_output:     [tf.Tensor] latent space

        split:              [int] index for choosing different ways of splitting the tensor:

                            0:  | car | rgb | lane|
                                | ... | ... | ... |

                            1:  |    car    |
                                      |    lane   |
                                      | rgb |
                                | ... | ... | ... |

                            2:  |       rgb       |
                                |    car    |
                                      |    lane   |
                                | ... | ... | ... |

                            3:  |       rgb       |
                                |       car       |
                                |       lane      |
                                | ............... |

        return:             [tuple] split latent space, and size of the rgb/segm components
        ------------------------------------------------------------------------------------------------- """

        # in this case RGB, CAR and LANE are equally split and not overlapping
        # here 'self.latent_subsize' is ignored
        if split == 0:
            sL                  = self.latent_size // 3
            sR                  = 2 * self.latent_size // 3
            encoder_output_car  = layers.Lambda( lambda x: x[ :,    : sL ] )( encoder_output )
            encoder_output_rgb  = layers.Lambda( lambda x: x[ :, sL : sR ] )( encoder_output )
            encoder_output_lane = layers.Lambda( lambda x: x[ :, sR :    ] )( encoder_output )

        # in this case, RGB is entirely contained in both CAR and LANE
        elif split == 1:
            sL                  = ( self.latent_size // 2 ) - ( self.latent_subsize // 2 )
            sR                  = ( self.latent_size // 2 ) + ( self.latent_subsize // 2 )
            encoder_output_car  = layers.Lambda( lambda x: x[ :,    : sR ] )( encoder_output )
            encoder_output_rgb  = layers.Lambda( lambda x: x[ :, sL : sR ] )( encoder_output )
            encoder_output_lane = layers.Lambda( lambda x: x[ :, sL :    ] )( encoder_output )

        # in this case, RGB contains entirely both CAR and LANE
        elif split == 2:
            sL                  = self.latent_subsize
            sR                  = self.latent_size - self.latent_subsize
            encoder_output_car  = layers.Lambda( lambda x: x[ :,    : sL ] )( encoder_output )
            encoder_output_rgb  = encoder_output
            encoder_output_lane = layers.Lambda( lambda x: x[ :, sR :    ] )( encoder_output )

        # in this case there is actually no split: RGB, CAR and LANE share the same latent vector
        elif split == 3:
            return encoder_output, encoder_output, encoder_output

        else:
            ms.print_err( "Split code {} not valid".format( split ) )
            
        return encoder_output_rgb, encoder_output_car, encoder_output_lane



# ===========================================================================================================



class Recursive( ABC ):
    """ -----------------------------------------------------------------------------------------------------
    Minimum common support for recursive networks
    ----------------------------------------------------------------------------------------------------- """

    def _get_recurr( self ):
        """ -------------------------------------------------------------------------------------------------
        Return a recurrent layer according to the object attribute

        return:         [keras.regularizers.Regularizer]
        ------------------------------------------------------------------------------------------------- """
        if self.recurr == 'RNN':
            return layers.SimpleRNN

        if self.recurr == 'GRU':
            return layers.GRU

        if self.recurr == 'LSTM':
            return layers.LSTM

        ms.print_err( "Recurrent layer {} not valid".format( self.recurr ) )



    def _init_denorm( self ):
        """ -------------------------------------------------------------------------------------------------
        compute an array for de-normalizing the output of the recurrent network, when using tanh as activation
        by multiplying the output with the array here computed, values will span the entire range of the latent
        A composite de-normalizing array makes sense for split code 2 only, otherwise a uniform array, tuned to
        RGB range is returned
        ------------------------------------------------------------------------------------------------- """
        if self.split == 2:
            s_rgb       = self.latent_size - 2 * self.latent_subsize
            s_segm      = self.latent_subsize
            denorm      = s_segm * [ CARRANGE ] + s_rgb * [ RBGRANGE ] + s_segm * [ LANRANGE ]
            return      K.variable( value=np.array( denorm ) )

        return      K.variable( value=np.array( self.latent_size * [ RBGRANGE ] ) )



    def __init__( self ):
        """ -------------------------------------------------------------------------------------------------
        class attributes
        ------------------------------------------------------------------------------------------------- """
        self.recurr                 = None           # [str] kind of recurrent layer (RNN/GRU/LSTM)
        if DEBUG: print ( "\nstart Recursive.__init__()" )
        super( Recursive, self ).__init__()          # to keep the super() chain
        if DEBUG: print ( "\nend Recursive.__init__()" )


    def define_recurrent( self, lname=None, mname='RTP' ):
        """ -------------------------------------------------------------------------------------------------
        Define the structure of the recurrent network

        lname:          [str] optional name of the layer
        # FIXME quick and brutal way to solve the problem of using this class inside RecMultiVAE,
        # because when testing an RTIME it may be necessary to load an RMVAE model,
        # and both have a layer with the same name (big problem for HDF5 loading)
        mname:          [str] name of the model
        
        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        # INPUT LAYER
        inp             = layers.Input( shape=( self.n_input, self.latent_size, ) )

        recurr_layer    = self._get_recurr()

        # RECURRENT LAYER
        x               = recurr_layer(
                self.latent_size,
                activation              = stanh,
                return_sequences        = False,
                return_state            = False,
                stateful                = False,
                unroll                  = False,
                name                    = lname         # note that in the case lname=None name will be
                                                        # the default one of Keras
        )( inp )

        return models.Model( inputs=inp, outputs=x, name=mname )





# ===========================================================================================================



class AE( Autoencoder ):
    """ -----------------------------------------------------------------------------------------------------
    Create a standard convolutional autoencoder

    The encoder is composed of a stack of convolution.
    The decoder is composed of a symmetric stack of deconvolutions.
    The innermost layers are flat Dense layers.
    ----------------------------------------------------------------------------------------------------- """

    def __init__( self ):
        """ -------------------------------------------------------------------------------------------------
        Initialize class attributes usign the config dicitonary, and create the network model
        ------------------------------------------------------------------------------------------------- """
        self.latent_size        = None                      # [int] size of the latent space
        self.dnse_dropout       = None                      # [list] dropout for the dense layers

        if DEBUG: print ( "\nstart AE.__init__()" )
        super( AE, self ).__init__()                        # init attributes and create model
        self.loss_func          = get_loss( self.loss )     # assign loss function
        if DEBUG: print ( "\nend AE.__init__()" )


    def define_encoder( self, mname='encoder' ):
        """ -------------------------------------------------------------------------------------------------
        Define the structure of the encoder part of the network

        mname:          [str] name of the model

        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        x               = self.define_basic_encoder( mname=mname )
        z               = layers.Dense( self.latent_size, name='z' )( x )

        # ENCODER MODEL
        encoder         = models.Model( 
                    inputs      = self.encoder_input,
                    outputs     = z,
                    name        = mname
        )
        model_summary( encoder, fname=mname )
        return encoder



    def define_decoder( self, input_size, mname='decoder' ):
        """ -------------------------------------------------------------------------------------------------
        Define the structure of the decoder part of the network

        input_size:     [tuple] size of the input of the decoder
        mname:          [str] name of the model
        
        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        x               = self.define_basic_decoder( input_size=input_size, mname=mname )
                
        # DECODER MODEL
        decoder         = models.Model(
                    inputs      = self.decoder_input,
                    outputs     = x,
                    name        = mname
        )
        model_summary( decoder, fname=mname )
        return decoder


    def define_model( self, mname='AE' ):
        """ -------------------------------------------------------------------------------------------------
        Define the entire structure of the network

        mname:          [str] name of the model
        
        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        enc                 = self.define_encoder()
        dec                 = self.define_decoder( ( self.latent_size, ) )

        encoder_output      = enc( self.encoder_input )
        model_output        = dec( encoder_output )

        return models.Model(
                    inputs      = self.encoder_input,
                    outputs     = model_output,
                    name        = mname
        )




# ===========================================================================================================



class VAE( Autoencoder ):
    """ -----------------------------------------------------------------------------------------------------
    Create a variational autoencoder

    The encoder is composed of a stack of convolutions.
    The decoder is composed of a symmetric stack of deconvolutions.
    The innermost layers contain two layers for the means and variances, from which is sampled
    the latent layer 'z', which is the input of the decoder.
    ----------------------------------------------------------------------------------------------------- """

    def __init__( self ):
        """ -------------------------------------------------------------------------------------------------
        Initialize class attributes usign the config dicitonary, and create the network model
        ------------------------------------------------------------------------------------------------- """
        self.latent_size        = None          # [int] size of the latent space
        self.kl_wght            = None          # [float] weight of KL component in loss function
        self.kl_incr            = None          # [float] increase of KL component in loss function
        self.dnse_dropout       = None          # [list] dropout for the dense layers
        
        if DEBUG: print ( "\nstart VAE.__init__()" )
        super( VAE, self ).__init__()           # init attributes and create model
        self.loss_func          = self.get_vae_loss( i_loss=self.loss )     # assign loss function
        self.kl_weight_0        = 1e-04         # starting weight of KL component

        # the current weight of KL component should be a K.variable
        # if kl_incr is defined greater than 0, then KL annealing is used, otherwise the value
        # remain fixed to kl_wght
        if self.kl_incr > 0.0:
            self.kl_weight          = K.variable( self.kl_weight_0, name='kl_weight' )
        else:
            self.kl_weight          = K.variable( self.kl_wght, name='kl_weight' )

        if DEBUG: print ( "\nend VAE.__init__()" )


    def latent_sampling( self, args ):
        """ -------------------------------------------------------------------------------------------------
        Draws a latent point from a distribution defined by the arguments passed as input,
        using a small random epsilon

        args:           [list of tf.Tensor] mean and log of variance

        return:         [tf.Tensor]
        ------------------------------------------------------------------------------------------------- """
        z_mean, z_log_var   = args
        epsilon             = K.random_normal(
                shape   = ( K.shape( z_mean )[ 0 ], self.latent_size ),
                mean    = 0.0,
                stddev  = 1.0
        )

        # 0.5 is used to take square root of the variance, to obtain standard deviation
        return z_mean + epsilon * K.exp( 0.5 * z_log_var )



    def get_vae_loss( self, i_loss=None ):
        """ -------------------------------------------------------------------------------------------------
        Compute the loss function for the variational autoencoder.
        The function is composed of two parts
            1. meausure of the difference between the output and the target images;
            2. kl_loss is the Kullback–Leibler divergence measuring how good is the approximated
               distribution computer by the encoder.

        This function uses direct 'tf' calls to name elements, instead of 'K' calls, so that are visible
        inside TensorBoard

        i_loss:         [function] meausure of the difference between the output and the target images

        return:         [function] loss function
        ------------------------------------------------------------------------------------------------- """
        lf  = get_loss( i_loss )
        dim = np.prod( self.img_size )

        # version of the model to be used for inference
        if not TRAIN:
            return lf

        def vae_loss( y_true, y_pred ):
            y_true          = tf.reshape( y_true, [ -1 ], name="y_true_flat" )
            y_pred          = tf.reshape( y_pred, [ -1 ], name="y_pred_flat" )
            y_dim           = y_true.shape[ -1 ]

            # loss meausuring the difference between the images
            img_loss        = lf( y_true, y_pred )
            img_loss        *= dim
            tf.summary.scalar( "img_loss", img_loss )

            # Kullback–Leibler divergence
            z_var           = tf.exp( self.z_log_var, name="z_var" )
            z_mean_2        = tf.square( self.z_mean, name="z_mean_2" )
            kl_loss         = - 0.5 * tf.reduce_sum(
                    1 + self.z_log_var - z_mean_2 - z_var,
                    axis        = -1,
                    name        = "kl_loss_sum"
            )

            if LooseVersion( tf.VERSION ) > LooseVersion( '1.5' ):
                kl_loss         = tf.reduce_mean( kl_loss, keepdims=False, name="kl_loss_mean" )
            else:
                kl_loss         = tf.reduce_mean( kl_loss, keep_dims=False, name="kl_loss_mean" )

            # DEBUG
            # img_loss   = tf.Print( img_loss, [ img_loss, kl_loss ] )

            return img_loss + self.kl_weight * kl_loss

        return vae_loss



    def define_encoder( self, mname='encoder' ):
        """ -------------------------------------------------------------------------------------------------
        Define the structure of the encoder part of the network

        mname:          [str] name of the model

        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """

        x               = self.define_basic_encoder( mname=mname )

        if TRAIN:
            # mean and variance of the probability distribution over the latent space
            self.z_mean     = layers.Dense( self.latent_size, name='z_mean' )( x )
            self.z_log_var  = layers.Dense( self.latent_size, name='z_log_var' )( x )

            # LAMBDA LAYER
            # sample a latent point from the distribution
            z               = layers.Lambda( self.latent_sampling, name='zeta' )( [ self.z_mean, self.z_log_var ] )

        # version of the model to be used for inference
        else:
            z               = layers.Dense( self.latent_size, name='z_mean' )( x )

        # ENCODER MODEL
        encoder         = models.Model( 
                    inputs      = self.encoder_input,
                    outputs     = z,
                    name        = mname
        )
        model_summary( encoder, fname=mname )
        return encoder



    def define_decoder( self, input_size, mname='decoder' ):
        """ -------------------------------------------------------------------------------------------------
        Define the structure of the decoder part of the network

        input_size:     [tuple] size of the input of the decoder
        mname:          [str] name of the model
        
        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        x               = self.define_basic_decoder( input_size=input_size, mname=mname )
                
        # DECODER MODEL
        decoder         = models.Model(
                    inputs      = self.decoder_input,
                    outputs     = x,
                    name        = mname
        )
        model_summary( decoder, fname=mname )
        return decoder



    def define_model( self, mname='VAE' ):
        """ -------------------------------------------------------------------------------------------------
        Define the entire structure of the network

        mname:          [str] name of the model
        
        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        enc                 = self.define_encoder()
        dec                 = self.define_decoder( ( self.latent_size, ) )

        encoder_output      = enc( self.encoder_input )
        model_output        = dec( encoder_output )

        return models.Model(
                    inputs      = self.encoder_input,
                    outputs     = model_output,
                    name        = mname
        )



# ===========================================================================================================



class MultipleAE( Multiple, AE ):
    """ -----------------------------------------------------------------------------------------------------
    Create an autoencoder with multiple decoding branches

    The encoder is composed of a stack of convolutions.
    The decoders are composed of stacks of deconvolutions.
    Each decoding branch takes as input a slice of the latent space.
    ----------------------------------------------------------------------------------------------------- """

    def __init__( self ):
        """ -------------------------------------------------------------------------------------------------
        Initialize class attributes usign the config dicitonary, and create the network model
        ------------------------------------------------------------------------------------------------- """
        self.loss_wght                  = None      # [tuple] weights for multiple loss computation

        if DEBUG: print ( "\nstart MultipleAE.__init__()" )
        super( MultipleAE, self ).__init__()        # init attributes and create model

        self.loss_func                  = [ get_loss( self.loss ), get_loss( 'UXE_CAR' ), get_loss( 'UXE_LANE' ) ]
        if DEBUG: print ( "\nend MultipleAE.__init__()" )


    def define_decoder_segm( self, input_size, mname='decoder_segm' ):
        """ -------------------------------------------------------------------------------------------------
        Define the structure of the decoder branches of the network producing segmented (B/W) output.

        It supposes that all the decoding branches of the network have same structure, except for
        the number of features of the last deconvolution (in the segmented case is equal to 1).

        input_size:     [tuple] size of the input of the decoder
        mname:          [str] name of the model

        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """

        # TODO consider the possibility of having different architectures for the 3 deconv branches
        # for now the are all the same

        v                       = self.dcnv_filters[ -1 ]
        self.dcnv_filters[ -1 ] = 1                             # set graylevel output
        decoder                 = super( MultipleAE, self ).define_decoder( input_size, mname=mname )
        self.dcnv_filters[ -1 ] = v                             # restore initial value
        return decoder


    def define_model( self, mname='MultipleAE' ):
        """ -------------------------------------------------------------------------------------------------
        Define the entire structure of the network

        mname:          [str] name of the model

        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        sz                  = self.latent_parts( self.split )
        enc                 = self.define_encoder()
        enc_output          = enc( self.encoder_input )                                 # latent space
        enc_split           = self.split_latent( enc_output, split=self.split )         # split latent space

        dec_rgb             = self.define_decoder( ( sz[ 0 ], ), mname='decoder_rgb' )
        dec_car             = self.define_decoder_segm( ( sz[ 1 ], ), mname='decoder_car' )
        dec_lane            = self.define_decoder_segm( ( sz[ 1 ], ), mname='decoder_lane' )

        # the model output is a tuple of 3
        model_outputs       = ( 
                dec_rgb(  enc_split[ 0 ] ),
                dec_car(  enc_split[ 1 ] ),
                dec_lane( enc_split[ 2 ] )
        )

        return models.Model(
                    inputs      = self.encoder_input,
                    outputs     = model_outputs,
                    name        = mname
        )



# ===========================================================================================================



class MultipleVAE( VAE, Multiple ):
    """ -----------------------------------------------------------------------------------------------------
    Create a variational autoencoder with multiple decoding branches

    The encoder is composed of a stack of convolutions.
    The decoders are composed of stacks of deconvolutions.
    Each decoding branch takes as input a slice of the latent space.
    ----------------------------------------------------------------------------------------------------- """

    def __init__( self ):
        """ -------------------------------------------------------------------------------------------------
        Initialize class attributes usign the config dicitonary, and create the network model
        ------------------------------------------------------------------------------------------------- """
        self.loss_wght                  = None      # [tuple] weights for multiple loss computation

        if DEBUG: print ( "\nstart MultipleVAE.__init__()" )
        super( MultipleVAE, self ).__init__()       # init attributes and create model

        # NOTE: separated kl_wght for RGB/car/lane is not in use any more, for incompatibly with
        # KL-annealing, and for its dubious theoretical significance

        self.loss_func                  = [
            self.get_vae_loss( i_loss=self.loss ),
            self.get_vae_loss( i_loss='UXE_CAR' ),
            self.get_vae_loss( i_loss='UXE_LANE' )
        ]
        if DEBUG: print ( "\nend MultipleVAE.__init__()" )


    def define_decoder_segm( self, input_size, mname='decoder_segm' ):
        """ -------------------------------------------------------------------------------------------------
        Define the structure of the decoder branches of the network producing segmented (B/W) output.

        It supposes that all the decoding branches of the network have same structure, except for
        the number of features of the last deconvolution (in the segmented case is equal to 1).

        input_size:     [tuple] size of the input of the decoder
        mname:          [str] name of the model

        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """

        # TODO consider the possibility of having different architectures for the 3 deconv branches
        # for now the are all the same

        v                       = self.dcnv_filters[ -1 ]
        self.dcnv_filters[ -1 ] = 1                             # set graylevel output
        decoder                 = super( MultipleVAE, self ).define_decoder( input_size, mname=mname )
        self.dcnv_filters[ -1 ] = v                             # restore initial value
        return decoder


    def define_model( self, mname='MultipleVAE' ):
        """ -------------------------------------------------------------------------------------------------
        Define the entire structure of the network

        mname:          [str] name of the model

        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        sz                  = self.latent_parts( self.split )
        enc                 = self.define_encoder()
        enc_output          = enc( self.encoder_input )                                 # latent space
        enc_split           = self.split_latent( enc_output, split=self.split )         # split latent space

        dec_rgb             = self.define_decoder( ( sz[ 0 ], ), mname='decoder_rgb' )
        dec_car             = self.define_decoder_segm( ( sz[ 1 ], ), mname='decoder_car' )
        dec_lane            = self.define_decoder_segm( ( sz[ 1 ], ), mname='decoder_lane' )

        # the model output is a tuple of 3
        model_outputs       = [ 
                dec_rgb(  enc_split[ 0 ] ),
                dec_car(  enc_split[ 1 ] ),
                dec_lane( enc_split[ 2 ] )
        ]

        return models.Model(
                    inputs      = self.encoder_input,
                    outputs     = model_outputs,
                    name        = mname
        )



# ===========================================================================================================



class Timepred:
    """ -----------------------------------------------------------------------------------------------------
    Create a feed-forward network predicting future frame in the feature space, using odometry data
    ----------------------------------------------------------------------------------------------------- """

    def __init__( self ):
        """ -------------------------------------------------------------------------------------------------
        Initialize class attributes usign the config dicitonary, and create the network model
        ------------------------------------------------------------------------------------------------- """
        self.n_input                = None          # [int] number of input frames
        self.n_output               = None          # [int] number of output frames
        self.loss                   = None          # [str] code specifying the type of loss function
        self.latent_size            = None          # [int] size of the latent space

        self.dnse_size              = None          # [list of int] size of each dense layer
        self.dnse_activation        = None          # [list of str] activation function for each dense layer
        self.dnse_wght_regul  	    = None          # [list of float] regularization factors for weights
        self.dnse_bias_regul  	    = None          # [list of float] regularization factors for biases
        self.dnse_actv_regul  	    = None          # [list of float] regularization factors for activations
        self.dnse_train             = None          # [list of bool] False to lock training of each dense layer
        
        # initialize class attributes with values from cnfg dict
        for k in self.__dict__:
            if k not in cnfg:
                ms.print_err( "Attribute '{}' of class '{}' not indicated".format( k, self.__class__ ) )
            exec( "self.{} = cnfg[ '{}' ]".format( k, k ) )                              

        self.regul_func     = regularizers.l2

        # create model
        if cnfg[ 'n_gpus' ] > 1:
            with tf.device( '/cpu:0' ):
                self.model  = self.define_model( mname=cnfg[ 'arch_class' ] )
        else:
            self.model      = self.define_model( mname=cnfg[ 'arch_class' ] )

        assert cnfg[ 'n_dnse' ] == len( self.dnse_size ) == len( self.dnse_activation ) == \
                len( self.dnse_wght_regul ) == len( self.dnse_bias_regul ) == len( self.dnse_actv_regul ) == \
                len( self.dnse_train )
                
        # assign loss function
        self.loss_func              = get_loss( self.loss )

        model_summary( self.model, fname=cnfg[ 'arch_class' ] )



    def define_model( self, mname='TP' ):
        """ -------------------------------------------------------------------------------------------------
        Define the structure of the network

        mname:          [str] name of the model
        
        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        i_comp          = []
        o_frames        = []

        # INPUT LAYERs
        i_odoms         = [ layers.Input( shape=( 2, ) ) for i in range( self.n_input - 1 ) ]
        i_frames        = [ layers.Input( shape=( self.latent_size, ) ) for i in range( self.n_input ) ]

        # DENSE LAYERs
        for i in range( self.n_input - 1 ):
            # expand odometry
            xo          = layers.Dense(                             
                    self.dnse_size[ 0 ],
                    activation              = self.dnse_activation[ 0 ],
                    kernel_regularizer      = self.regul_func( self.dnse_wght_regul[ 0 ] ),
                    bias_regularizer        = self.regul_func( self.dnse_bias_regul[ 0 ] ),
                    activity_regularizer    = self.regul_func( self.dnse_actv_regul[ 0 ] ),
                    trainable               = self.dnse_train[ 0 ],
                    name                    = 'dnse0_{}'.format( i )
            )( i_odoms[ i ] )

            # concatenate with corresponding frame
            xo          = layers.Concatenate( axis=1 )( [ xo, i_frames[ i+1 ] ] )

            # compress frame with odometry
            xo          = layers.Dense(                             
                    self.dnse_size[ 1 ],
                    activation              = self.dnse_activation[ 1 ],
                    kernel_regularizer      = self.regul_func( self.dnse_wght_regul[ 1 ] ),
                    bias_regularizer        = self.regul_func( self.dnse_bias_regul[ 1 ] ),
                    activity_regularizer    = self.regul_func( self.dnse_actv_regul[ 1 ] ),
                    trainable               = self.dnse_train[ 1 ],
                    name                    = 'dnse1_{}'.format( i )
            )( xo )

            i_comp.append( xo )

        # CONCATENATE LAYER
        x           = layers.Concatenate( axis=1 )( [ i_frames[ 0 ], *i_comp ] )

        nd          = len( self.dnse_size )
    
        # DENSE LAYERs
        for i_dnse in range( 2, nd-1 ):
            x           = layers.Dense(                             
                    self.dnse_size[ i_dnse ],
                    activation              = self.dnse_activation[ i_dnse ],
                    # regularizations
                    kernel_regularizer      = self.regul_func( self.dnse_wght_regul[ i_dnse ] ),
                    bias_regularizer        = self.regul_func( self.dnse_bias_regul[ i_dnse ] ),
                    activity_regularizer    = self.regul_func( self.dnse_actv_regul[ i_dnse ] ),
                    trainable               = self.dnse_train[ i_dnse ],
                    name                    = 'dnse2_{}'.format( i_dnse )
            )( x )

        # OUTPUT LAYERs
        for i in range( self.n_output ):
            o_frames.append( layers.Dense(                             
                    self.latent_size,
                    activation              = self.dnse_activation[ -1 ],
                    # regularizations
                    kernel_regularizer      = self.regul_func( self.dnse_wght_regul[ -1 ] ),
                    bias_regularizer        = self.regul_func( self.dnse_bias_regul[ -1 ] ),
                    activity_regularizer    = self.regul_func( self.dnse_actv_regul[ -1 ] ),
                    trainable               = self.dnse_train[ -1 ],
                    name                    = 'output_{}'.format( i + 1 )
            )( x ) )

        return models.Model( inputs=[ *i_odoms, *i_frames ], outputs=o_frames, name=mname )




# ===========================================================================================================



class RecTimepred( Recursive ):
    """ -----------------------------------------------------------------------------------------------------
    Create a recursive network predicting future frame in the feature space
    ----------------------------------------------------------------------------------------------------- """


    def __init__( self, lname=None ):
        """ -------------------------------------------------------------------------------------------------
        Initialize class attributes usign the config dicitonary, and create the network model

        lname:          [str] optional name of the layer (TO BE IMPROVED, see comments in define_recurrent)
        ------------------------------------------------------------------------------------------------- """
        self.n_input                = None          # [int] number of input frames
        self.n_output               = None          # [int] number of output frames
        self.loss                   = None          # [str] code specifying the type of loss function
        self.latent_size            = None          # [int] size of the latent space
        self.split                  = None          # [int] code of latent space configuration,
                                                    # should be 3 for non-multi
        self.latent_subsize         = None          # [int] size of lane/car components in latent space

        if DEBUG: print ( "\nstart RecTimepred.__init__()" )
        super( RecTimepred, self ).__init__()

        # initialize class attributes with values from cnfg dict
        for k in self.__dict__:
            if k not in cnfg:
                ms.print_err( "Attribute '{}' of class '{}' not indicated".format( k, self.__class__ ) )
            exec( "self.{} = cnfg[ '{}' ]".format( k, k ) )                              

        # create model
        if cnfg[ 'n_gpus' ] > 1:
            with tf.device( '/cpu:0' ):
                self.model  = self.define_model( lname=lname, mname=cnfg[ 'arch_class' ] )
        else:
            self.model      = self.define_model( lname=lname, mname=cnfg[ 'arch_class' ] )

        """
        assert cnfg[ 'n_dnse' ] == len( self.dnse_size ) == len( self.dnse_activation ) == \
                len( self.dnse_wght_regul ) == len( self.dnse_bias_regul ) == len( self.dnse_actv_regul ) == \
                len( self.dnse_train )
        """
                
        # assign loss function
        self.loss_func              = get_loss( self.loss )

        model_summary( self.model, fname=cnfg[ 'arch_class' ] )
        if DEBUG: print ( "\nend RecTimepred.__init__()" )


    def define_model( self, lname=None, mname='RTP' ):
        """ -------------------------------------------------------------------------------------------------
        Define the structure of the network

        lname:          [str] optional name of the layer
        mname:          [str] name of the model
        
        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        return self.define_recurrent( lname=lname, mname=mname )



# ===========================================================================================================



class Rec2Timepred( Recursive ):
    """ -----------------------------------------------------------------------------------------------------
    Create a network with two recursive layers predicting future frame in the feature space
    Accept as layer any available Keras recurrent layer, with possibility of bidirectionality
    ----------------------------------------------------------------------------------------------------- """

    def _get_recurr( self ):
        """ -------------------------------------------------------------------------------------------------
        Return a recurrent layer according to the object attribute

        Take into account that the code parsed in self.recurr may specify bidirectional recurrente network,
        by the initial character 'B'

        return:         [keras.regularizers.Regularizer]
        ------------------------------------------------------------------------------------------------- """
        self.bidirectional  = False
        if self.recurr[ 0 ] == 'B':
            self.bidirectional  = True
            self.recurr         = self.recurr[ 1 : ]

        if self.recurr == 'RNN':
            return layers.SimpleRNN

        if self.recurr == 'GRU':
            return layers.GRU

        if self.recurr == 'LSTM':
            return layers.LSTM

        ms.print_err( "Recurrent layer {} not valid".format( self.recurr ) )



    def __init__( self, lname=None ):
        """ -------------------------------------------------------------------------------------------------
        Initialize class attributes usign the config dicitonary, and create the network model

        lname:          [str] optional name of the layer (TO BE IMPROVED, see comments in define_recurrent)
        ------------------------------------------------------------------------------------------------- """
        self.n_input                = None          # [int] number of input frames
        self.n_output               = None          # [int] number of output frames
        self.loss                   = None          # [str] code specifying the type of loss function
        self.latent_size            = None          # [int] size of the latent space
        self.split                  = None          # [int] code of latent space configuration,
                                                    # should be 3 for non-multi
        self.latent_subsize         = None          # [int] size of lane/car components in latent space

        if DEBUG: print ( "\nstart Rec2Timepred.__init__()" )
        super( Rec2Timepred, self ).__init__()

        # initialize class attributes with values from cnfg dict
        for k in self.__dict__:
            if k not in cnfg:
                ms.print_err( "Attribute '{}' of class '{}' not indicated".format( k, self.__class__ ) )
            exec( "self.{} = cnfg[ '{}' ]".format( k, k ) )                              

        self.denorm         = self._init_denorm()

        # create model
        if cnfg[ 'n_gpus' ] > 1:
            with tf.device( '/cpu:0' ):
                self.model  = self.define_model( lname=lname, mname=cnfg[ 'arch_class' ] )
        else:
            self.model      = self.define_model( lname=lname, mname=cnfg[ 'arch_class' ] )

        # assign loss function
        self.loss_func              = get_loss( self.loss )

        model_summary( self.model, fname=cnfg[ 'arch_class' ] )
        if DEBUG: print ( "\nend Rec2Timepred.__init__()" )



    def define_recurrent( self, lname=None, mname='R2TP' ):
        """ -------------------------------------------------------------------------------------------------
        Define the structure of the recurrent network

        lname:          [tuple] optional tuple with names name of the two layers
        mname:          [str] name of the model
        
        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        if lname is None:
            lname   = None, None

        # INPUT LAYER
        inp             = layers.Input( shape=( self.n_input, self.latent_size, ) )

        recurr_layer    = self._get_recurr()

        # FIRST RECURRENT LAYER
        x               = recurr_layer(
                self.latent_size,
                activation              = "tanh",
                return_sequences        = True,
                return_state            = False,
                stateful                = False,
                unroll                  = False,
                name                    = lname[ 0 ]
        )( inp )

        # SECOND RECURRENT LAYER
        if self.bidirectional:
            x               = layers.Bidirectional(
                        recurr_layer(
                        self.latent_size,
                        activation              = "tanh",
                        return_sequences        = False,
                        return_state            = False,
                        stateful                = False,
                        unroll                  = False,
                        name                    = lname[ 1 ]
                    ),
                    merge_mode              = 'ave'
            )( x )
        else:
            x               = recurr_layer(
                    self.latent_size,
                    activation              = "tanh",
                    return_sequences        = False,
                    return_state            = False,
                    stateful                = False,
                    unroll                  = False,
                    name                    = lname[ 1 ]
            )( x )

        x               = layers.Lambda( lambda x: self.denorm * x, name="de_normalize" )( x )

        return models.Model( inputs=inp, outputs=x, name=mname )



    def define_model( self, lname=None, mname='R2TP' ):
        """ -------------------------------------------------------------------------------------------------
        Define the structure of the network

        lname:          [str] optional name of the layer
        mname:          [str] name of the model
        
        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        return self.define_recurrent( lname=lname, mname=mname )



# ===========================================================================================================



class RecMTimepred( Recursive ):
    """ -----------------------------------------------------------------------------------------------------
    Create a network with arbitrary number of staked recursive layers, with multiple top layers
    predicting future frames at various time steps ahead
    Accept as layer any available Keras recurrent layer
    ----------------------------------------------------------------------------------------------------- """

    def _init_wloss( self ):
        """ -------------------------------------------------------------------------------------------------
        compute an array for weighting the loss on the components of latent vector
        ------------------------------------------------------------------------------------------------- """
        if self.split == 2:
            s_rgb       = self.latent_size - 2 * self.latent_subsize
            s_segm      = self.latent_subsize
            car, lane   = self.loss_w_segm
            weight      = s_segm * [ car ] + s_rgb * [ 1.0 ] + s_segm * [ lane ]
            return      K.variable( value=np.array( weight ) )

        return      K.variable( value=np.array( self.latent_size * [ 1.0 ] ) )



    def get_weighted_loss( self ):
        """ -------------------------------------------------------------------------------------------------
        Compute the loss function as mean square error, weighting differently the segmentation 
        components and the RGB component

        return:         [function] loss function
        ------------------------------------------------------------------------------------------------- """
        def weighted_loss( y_true, y_pred ):
            loss    = K.square( y_pred - y_true )
            return K.mean( self.wloss * loss, axis=-1 )

        return weighted_loss



    def __init__( self, lname=None ):
        """ -------------------------------------------------------------------------------------------------
        Initialize class attributes usign the config dicitonary, and create the network model

        lname:          [str] optional name of the layer (TO BE IMPROVED, see comments in define_recurrent)
        ------------------------------------------------------------------------------------------------- """
        self.n_input                = None          # [int] number of input frames
        self.n_output               = None          # [int] number of output frames
        self.n_stack                = None          # [int] number of stacked recurrent layers
        self.loss_wght              = None          # [tuple] weights for multiple loss computation
        self.loss_w_segm            = None          # [tuple] extra loss weight for (car, lane) components
        self.latent_size            = None          # [int] size of the latent space
        self.split                  = None          # [int] code of latent space configuration,
                                                    # should be 3 for non-multi
        self.latent_subsize         = None          # [int] size of lane/car components in latent space
        self.recurr                 = None          # [str] kind of recurrent layer (RNN/GRU/LSTM)
        self.dropout                = None          # [tuple] (dropout, recurrent_dropout) for recurrent layer

        if DEBUG: print ( "\nstart RecMTimepred.__init__()" )
        super( RecMTimepred, self ).__init__()
        # initialize class attributes with values from cnfg dict
        for k in self.__dict__:
            if k not in cnfg:
                ms.print_err( "Attribute '{}' of class '{}' not indicated".format( k, self.__class__ ) )
            exec( "self.{} = cnfg[ '{}' ]".format( k, k ) )                              

        assert  len( self.loss_w_segm ) == 2
        assert  len( self.dropout )     == 2
        assert  len( self.loss_wght )   == self.n_output

        self.denorm         = self._init_denorm()
        self.wloss          = self._init_wloss()

        # create model
        if cnfg[ 'n_gpus' ] > 1:
            with tf.device( '/cpu:0' ):
                self.model  = self.define_model( mname=cnfg[ 'arch_class' ] )
        else:
            self.model      = self.define_model( mname=cnfg[ 'arch_class' ] )

        # assign loss function
        self.loss_func              = self.n_output * [ self.get_weighted_loss() ]

        model_summary( self.model, fname=cnfg[ 'arch_class' ] )

        if DEBUG: print ( "\nend RecMTimepred.__init__()" )


    def define_recurrent( self, mname='RMTP' ):
        """ -------------------------------------------------------------------------------------------------
        Define the structure of the recurrent network

        mname:          [str] name of the model
        
        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """

        # INPUT LAYER
        inp             = layers.Input( shape=( self.n_input, self.latent_size, ) )

        recurr_layer    = self._get_recurr()

        x               = inp
        # STACKED RECURRENT LAYERS
        for i in range( self.n_stack ):
            x               = recurr_layer(
                    self.latent_size,
                    activation              = "tanh",
                    return_sequences        = True,
                    return_state            = False,
                    stateful                = False,
                    unroll                  = False,
                    dropout                 = self.dropout[ 0 ],
                    recurrent_dropout       = self.dropout[ 1 ],
                    name                    = "stacked_rnn_{}".format( i+1 )
            )( x )

        # TOP PREDICTING LAYERS
        outputs         = []
        for i in range( self.n_output ):
            y   = recurr_layer(
                    self.latent_size,
                    activation              = "tanh",
                    return_sequences        = False,
                    return_state            = False,
                    stateful                = False,
                    unroll                  = False,
                    dropout                 = self.dropout[ 0 ],
                    recurrent_dropout       = self.dropout[ 1 ],
                    name                    = "top_rnn_{}".format( i+1 )
            )( x )
            outputs.append(
                    layers.Lambda( lambda x: self.denorm * x, name="de_norm_{}".format( i+1 ) )( y )
            )

        return models.Model( inputs=inp, outputs=outputs, name=mname )



    def define_model( self, mname='RMTP' ):
        """ -------------------------------------------------------------------------------------------------
        Define the structure of the network

        mname:          [str] name of the model
        
        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        return self.define_recurrent( mname=mname )



# ===========================================================================================================



class RecAE( AE, Recursive ):
    """ -----------------------------------------------------------------------------------------------------
    Create a network composed by a common encoder, a latent space, a common decoder
    for the self-image, and a two-input recurrent network using the common decoder for predicting
    the next frame
    ----------------------------------------------------------------------------------------------------- """

    def __init__( self ):
        """ -------------------------------------------------------------------------------------------------
        Initialize class attributes usign the config dicitonary, and create the network model
        ------------------------------------------------------------------------------------------------- """
        self.n_input                = None          # [int] number of input frames
        self.n_output               = None          # [int] number of output frames
        self.loss_wght              = None          # [tuple] weights for multiple loss computation
        self.recurr                 = None          # [str] kind of recurrent layer (RNN/GRU/LSTM)

        if DEBUG: print ( "\nstart RecAE.__init__()" )
        super( RecAE, self ).__init__()
        self.ae_loss_func           = self.loss_func
        self.rec_loss_func          = get_loss( self.loss )
        self.loss_func              = [ self.ae_loss_func, self.ae_loss_func, self.rec_loss_func ]
        if DEBUG: print ( "\nend RecAE.__init__()" )



    def define_model( self, mname='RecAE' ):
        """ -------------------------------------------------------------------------------------------------
        Define the entire structure of the network

        mname:          [str] name of the model

        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        enc                 = self.define_encoder()
        dec                 = self.define_decoder( ( self.latent_size, ) )
        rec                 = self.define_recurrent()               # FIXME pass here different name of layer
        frame1              = layers.Input( shape=self.img_size, name="input_1" )
        frame2              = layers.Input( shape=self.img_size, name="input_2" )
        lat_frame1          = enc( frame1 )
        lat_frame2          = enc( frame2 )
        dec_frame1          = dec( lat_frame1 )
        dec_frame2          = dec( lat_frame2 )
        stack_frames        = layers.Lambda( lambda x: K.stack( x, axis=1 ), name="frame_sequence" )
        sequence            = stack_frames( [ lat_frame1, lat_frame2 ] )
        lat_frame3          = rec( sequence )
        dec_frame3          = dec( lat_frame3 )

        # the model output is a tuple of 3
        model_outputs       = [ dec_frame1, dec_frame2, dec_frame3 ]

        return models.Model(
                    inputs      = [ frame1, frame2 ],
                    outputs     = model_outputs,
                    name        = mname
        )



# ===========================================================================================================



class RecVAE( VAE, Recursive ):
    """ -----------------------------------------------------------------------------------------------------
    Create a network composed by a common encoder, a variational latent space, a common decoder
    for the self-image, and a two-input recurrent network using the common decoder for predicting
    the next frame

    Few notes:
    - variational loss is computed only once for the first frame, since the KL divergence is independent
      of the frame loss, there is no reason to compute it more than once
    - all layers with weights of encoder and decoder are shared, and it is a source of troubles for the naming
      of placeholders (especially Tensorflow internal placeholders) and their reuse, when computing gradients
    - this is the reason for a lot of naming, including in define_model() Lambda layers for naming only
    ----------------------------------------------------------------------------------------------------- """

    def __init__( self ):
        """ -------------------------------------------------------------------------------------------------
        Initialize class attributes usign the config dictionary, and create the network model
        ------------------------------------------------------------------------------------------------- """
        self.n_input            = None          # [int] number of input frames
        self.n_output           = None          # [int] number of output frames
        self.recurr             = None          # [str] kind of recurrent layer (RNN/GRU/LSTM)
        
        if DEBUG: print ( "\nstart RecVAE.__init__()" )
        super( RecVAE, self ).__init__()

        self.loss_func              = {         # use variational loss for the first frame
            'out_frame0'  : self.get_loss_frame( i_loss=self.loss ),
            'out_frame1'  : get_loss( self.loss ),
            'out_frame2'  : get_loss( self.loss )
        }
        if DEBUG: print ( "\nend RecVAE.__init__()" )



    def latent_sampling( self, args ):
        """ -------------------------------------------------------------------------------------------------
        Draws a latent point from a distribution defined by the arguments passed as input,
        using a small random epsilon

        args:           [list of tf.Tensor] mean and log of variance

        return:         [tf.Tensor]
        ------------------------------------------------------------------------------------------------- """
        z_mean, z_log_var   = args
        epsilon             = K.random_normal(
                shape   = ( K.shape( z_mean )[ 0 ], self.latent_size ),
                mean    = 0.0,
                stddev  = 1.0
        )

        # 0.5 is used to take square root of the variance, to obtain standard deviation
        return z_mean + epsilon * K.exp( 0.5 * z_log_var )



    def latent_frame( self, args ):
        """ -------------------------------------------------------------------------------------------------
        Latent layer with inputs for the first frame

        return:         [tf.Tensor]
        ------------------------------------------------------------------------------------------------- """
        z_mean, z_log_var       = args
        self.frame1_z_mean      = z_mean
        self.frame1_z_log_var   = z_log_var
        z                       = layers.Lambda( self.latent_sampling, name="frame1_z" )( [ z_mean, z_log_var ] )
        return z



    def get_loss_frame( self, i_loss=None ):
        """ -------------------------------------------------------------------------------------------------
        Compute the loss function for the variational autoencoder when executed on the first frame
        The function is composed of two parts
            1. meausure of the difference between the output and the target images;
            2. kl_loss is the Kullback–Leibler divergence measuring how good is the approximated
               distribution computer by the encoder.

        This function uses direct 'tf' calls to name elements, instead of 'K' calls, so that are visible
        inside TensorBoard

        i_loss:         [function] meausure of the difference between the output and the target images

        return:         [function] loss function
        ------------------------------------------------------------------------------------------------- """
        lf  = get_loss( i_loss )
        dim = np.prod( self.img_size )

        # version of the model to be used for inference
        if not TRAIN:
            return lf

        def vae_loss( y_true, y_pred ):
            y_true          = tf.reshape( y_true, [ -1 ], name="frame1_y_true_flat" )
            y_pred          = tf.reshape( y_pred, [ -1 ], name="frame1_y_pred_flat" )
            y_dim           = y_true.shape[ -1 ]

            # loss meausuring the difference between the images
            img_loss        = lf( y_true, y_pred )
            img_loss        *= dim
            tf.summary.scalar( "frame1_img_loss", img_loss )

            # Kullback–Leibler divergence
            z_var           = tf.exp( self.frame1_z_log_var, name="frame1_z_var" )
            z_mean_2        = tf.square( self.frame1_z_mean, name="frame1_z_mean_2" )
            kl_loss         = - 0.5 * tf.reduce_sum(
                    1 + self.frame1_z_log_var - z_mean_2 - z_var,
                    axis        = -1,
                    name        = "frame1_kl_loss_sum"
            )

            if LooseVersion( tf.VERSION ) > LooseVersion( '1.5' ):
                kl_loss         = tf.reduce_mean( kl_loss, keepdims=False, name="frame1_kl_loss_mean" )
            else:
                kl_loss         = tf.reduce_mean( kl_loss, keep_dims=False, name="frame1_kl_loss_mean" )

            # DEBUG
            # img_loss   = tf.Print( img_loss, [ img_loss, kl_loss ] )

            return img_loss + self.kl_weight * kl_loss

        return vae_loss



    def define_encoder( self, mname='encoder' ):
        """ -------------------------------------------------------------------------------------------------
        Define the structure of the encoder part of the network
        NOTE: this model is derived from VAE.define_encoder() with the difference that
        it does not include the latent sampling layer, which should be included only
        when evaluating the first frame of the sequence

        mname:          [str] name of the model

        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """

        x               = self.define_basic_encoder( mname=mname )

        if TRAIN:
            # mean and variance of the probability distribution over the latent space
            self.z_mean     = layers.Dense( self.latent_size, name='z_mean' )( x )
            self.z_log_var  = layers.Dense( self.latent_size, name='z_log_var' )( x )

            z               = [ self.z_mean, self.z_log_var ]

        # version of the model to be used for inference
        else:
            z               = layers.Dense( self.latent_size, name='z_mean' )( x )

        # ENCODER MODEL
        encoder         = models.Model( 
                    inputs      = self.encoder_input,
                    outputs     = z,
                    name        = mname
        )
        model_summary( encoder, fname=mname )
        return encoder



    def define_model( self, mname='RecVAE' ):
        """ -------------------------------------------------------------------------------------------------
        Define the entire structure of the network

        mname:          [str] name of the model

        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        # load the main submodels
        enc                 = self.define_encoder()
        dec                 = self.define_decoder( ( self.latent_size, ) )
        rec                 = self.define_recurrent()               # FIXME pass here different name of layer

        # placeholders for the two input frames
        frame0              = layers.Input( shape=self.img_size, name="frame0" )
        frame1              = layers.Input( shape=self.img_size, name="frame1" )

        # include the random sampling of the latent, but only when training and for the first frame
        if TRAIN:
            enc_frame0          = enc( frame0 )
            enc_frame1          = enc( frame1 )
            lat_frame0          = self.latent_frame( enc_frame0 )
            lat_frame1          = enc_frame1[ 0 ]

        # otherwise take the output of the encoder without random sampling
        else:
            lat_frame0          = enc( frame0 )
            lat_frame1          = enc( frame1 )
        dec_frame0          = dec( lat_frame0 )
        dec_frame1          = dec( lat_frame1 )
        stack_frames        = layers.Lambda( lambda x: K.stack( x, axis=1 ), name="frame_sequence" )
        sequence            = stack_frames( [ lat_frame0, lat_frame1 ] )
        lat_frame2          = rec( sequence )
        dec_frame2          = dec( lat_frame2 )

        # dummy layers necessary for the multiple outputs
        # otherwise Keras uses as name of the output the name of the last layer, which is shared
        # NOTE that the names here used are somehow special, because are the keys for the
        # loss and loss_weights dictionaries, arguments of model.compile
        # TODO: define the names globally elsewhere to ensure consistency
        out_frame0          = layers.Lambda( lambda x: x, name="out_frame0" )( dec_frame0 )
        out_frame1          = layers.Lambda( lambda x: x, name="out_frame1" )( dec_frame1 )
        out_frame2          = layers.Lambda( lambda x: x, name="out_frame2" )( dec_frame2 )
        model_outputs       = [ out_frame0, out_frame1, out_frame2 ]

        return models.Model(
                    inputs      = [ frame0, frame1 ],
                    outputs     = model_outputs,
                    name        = mname
        )


# ===========================================================================================================



class RecMultiVAE( MultipleVAE, Recursive ):
    """ -----------------------------------------------------------------------------------------------------
    Create a network composed by a common encoder, a variational latent space, a common decoder
    for the self-image, and a two-input recurrent network using the common decoder for predicting
    the next frame

    Few notes:
    - variational loss is computed only once for the first frame, since the KL divergence is independent
      of the frame loss, there is no reason to compute it more than once
    - all layers with weights of encoder and decoder are shared, and it is a source of troubles for the naming
      of placeholders (especially Tensorflow internal placeholders) and their reuse, when computing gradients
    - this is the reason for a lot of naming, including in define_model() Lambda layers for naming only
    ----------------------------------------------------------------------------------------------------- """

    def __init__( self ):
        """ -------------------------------------------------------------------------------------------------
        Initialize class attributes usign the config dictionary, and create the network model
        ------------------------------------------------------------------------------------------------- """
        self.n_input            = None          # [int] number of input frames
        self.n_output           = None          # [int] number of output frames
        self.recurr             = None          # [str] kind of recurrent layer (RNN/GRU/LSTM)
        
        if DEBUG: print ( "\nstart RecMultiVAE.__init__()" )
        super( RecMultiVAE, self ).__init__()

        self.loss_func              = {         # use variational loss for the first frame
            "out_f0_rgb"  : self.get_loss_frame( i_loss=self.loss ),
            "out_f0_car"  : self.get_loss_frame( i_loss='UXE_CAR' ),
            "out_f0_lane" : self.get_loss_frame( i_loss='UXE_LANE' ),
            "out_f1_rgb"  : get_loss( self.loss ),
            "out_f1_car"  : get_loss( 'UXE_CAR' ),
            "out_f1_lane" : get_loss( 'UXE_LANE' ),
            "out_f2_rgb"  : get_loss( self.loss ),
            "out_f2_car"  : get_loss( 'UXE_CAR' ),
            "out_f2_lane" : get_loss( 'UXE_LANE' )
        }
        if DEBUG: print ( "\nend RecMultiVAE.__init__()" )



    def latent_sampling( self, args ):
        """ -------------------------------------------------------------------------------------------------
        Draws a latent point from a distribution defined by the arguments passed as input,
        using a small random epsilon

        args:           [list of tf.Tensor] mean and log of variance

        return:         [tf.Tensor]
        ------------------------------------------------------------------------------------------------- """
        z_mean, z_log_var   = args
        epsilon             = K.random_normal(
                shape   = ( K.shape( z_mean )[ 0 ], self.latent_size ),
                mean    = 0.0,
                stddev  = 1.0
        )

        # 0.5 is used to take square root of the variance, to obtain standard deviation
        return z_mean + epsilon * K.exp( 0.5 * z_log_var )



    def latent_frame( self, args ):
        """ -------------------------------------------------------------------------------------------------
        latent layer with inputs for the first frame

        return:         [tf.Tensor]
        ------------------------------------------------------------------------------------------------- """
        z_mean, z_log_var       = args
        self.frame1_z_mean      = z_mean
        self.frame1_z_log_var   = z_log_var
        z                       = layers.Lambda( self.latent_sampling, name="frame1_z" )( [ z_mean, z_log_var ] )
        return z



    def get_loss_frame( self, i_loss=None ):
        """ -------------------------------------------------------------------------------------------------
        Compute the loss function for the variational autoencoder when executed on the first frame
        The function is composed of two parts
            1. meausure of the difference between the output and the target images;
            2. kl_loss is the Kullback–Leibler divergence measuring how good is the approximated
               distribution computer by the encoder.

        This function uses direct 'tf' calls to name elements, instead of 'K' calls, so that are visible
        inside TensorBoard

        i_loss:         [function] meausure of the difference between the output and the target images

        return:         [function] loss function
        ------------------------------------------------------------------------------------------------- """
        lf  = get_loss( i_loss )
        dim = np.prod( self.img_size )

        # version of the model to be used for inference
        if not TRAIN:
            return lf

        def vae_loss( y_true, y_pred ):
            y_true          = tf.reshape( y_true, [ -1 ], name="frame1_y_true_flat" )
            y_pred          = tf.reshape( y_pred, [ -1 ], name="frame1_y_pred_flat" )
            y_dim           = y_true.shape[ -1 ]

            # loss meausuring the difference between the images
            img_loss        = lf( y_true, y_pred )
            img_loss        *= dim
            tf.summary.scalar( "frame1_img_loss", img_loss )

            # Kullback–Leibler divergence
            z_var           = tf.exp( self.frame1_z_log_var, name="frame1_z_var" )
            z_mean_2        = tf.square( self.frame1_z_mean, name="frame1_z_mean_2" )
            kl_loss         = - 0.5 * tf.reduce_sum(
                    1 + self.frame1_z_log_var - z_mean_2 - z_var,
                    axis        = -1,
                    name        = "frame1_kl_loss_sum"
            )

            if LooseVersion( tf.VERSION ) > LooseVersion( '1.5' ):
                kl_loss         = tf.reduce_mean( kl_loss, keepdims=False, name="frame1_kl_loss_mean" )
            else:
                kl_loss         = tf.reduce_mean( kl_loss, keep_dims=False, name="frame1_kl_loss_mean" )

            # DEBUG
            # img_loss   = tf.Print( img_loss, [ img_loss, kl_loss ] )

            return img_loss + self.kl_weight * kl_loss

        return vae_loss


    def define_encoder( self, mname='encoder' ):
        """ -------------------------------------------------------------------------------------------------
        Define the structure of the encoder part of the network
        NOTE: this model is derived from VAE.define_encoder() with the difference that
        it does not include the latent sampling layer, which should be included only
        when evaluating the first frame of the sequence

        mname:          [str] name of the model

        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        x               = self.define_basic_encoder( mname=mname )

        if TRAIN:
            # mean and variance of the probability distribution over the latent space
            self.z_mean     = layers.Dense( self.latent_size, name='z_mean' )( x )
            self.z_log_var  = layers.Dense( self.latent_size, name='z_log_var' )( x )

            z               = [ self.z_mean, self.z_log_var ]

        # version of the model to be used for inference
        else:
            z               = layers.Dense( self.latent_size, name='z_mean' )( x )

        # ENCODER MODEL
        encoder         = models.Model( 
                    inputs      = self.encoder_input,
                    outputs     = z,
                    name        = mname
        )
        model_summary( encoder, fname=mname )
        return encoder


    def define_model( self, mname='RecMultiVAE' ):
        """ -------------------------------------------------------------------------------------------------
        Define the entire structure of the network

        mname:          [str] name of the model

        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        # load the main submodels
        enc                 = self.define_encoder()
        sz                  = self.latent_parts( self.split )
        dec_rgb             = self.define_decoder( ( sz[ 0 ], ), mname='decoder_rgb' )
        dec_car             = self.define_decoder_segm( ( sz[ 1 ], ), mname='decoder_car' )
        dec_lane            = self.define_decoder_segm( ( sz[ 1 ], ), mname='decoder_lane' )
        rec                 = self.define_recurrent()               # FIXME pass here different name of layer

        # placeholders for the two input frames
        frame0              = layers.Input( shape=self.img_size, name="frame0" )
        frame1              = layers.Input( shape=self.img_size, name="frame1" )

        # include the random sampling of the latent, but only when training and for the first frame
        if TRAIN:
            enc_frame0          = enc( frame0 )
            enc_frame1          = enc( frame1 )
            lat_frame0          = self.latent_frame( enc_frame0 )
            lat_frame1          = enc_frame1[ 0 ]

        # otherwise take the output of the encoder without random sampling
        else:
            lat_frame0          = enc( frame0 )
            lat_frame1          = enc( frame1 )

        # compute the third latent with the recursive submodel
        stack_frames        = layers.Lambda( lambda x: K.stack( x, axis=1 ), name="frame_sequence" )
        sequence            = stack_frames( [ lat_frame0, lat_frame1 ] )
        lat_frame2          = rec( sequence )

        # now split latents to feed the separated decoders
        lat_f0_split        = self.split_latent( lat_frame0, split=self.split )
        lat_f1_split        = self.split_latent( lat_frame1, split=self.split )
        lat_f2_split        = self.split_latent( lat_frame2, split=self.split )

        # decoders for all frames and components
        dec_f0_rgb          = dec_rgb( lat_f0_split[ 0 ] )
        dec_f0_car          = dec_car( lat_f0_split[ 1 ] )
        dec_f0_lane         = dec_lane( lat_f0_split[ 2 ] )
        dec_f1_rgb          = dec_rgb( lat_f1_split[ 0 ] )
        dec_f1_car          = dec_car( lat_f1_split[ 1 ] )
        dec_f1_lane         = dec_lane( lat_f1_split[ 2 ] )
        dec_f2_rgb          = dec_rgb( lat_f2_split[ 0 ] )
        dec_f2_car          = dec_car( lat_f2_split[ 1 ] )
        dec_f2_lane         = dec_lane( lat_f2_split[ 2 ] )

        # dummy layers necessary for the multiple outputs
        # otherwise Keras uses as name of the output the name of the last layer, which is shared
        # NOTE that the names here used are somehow special, because are the keys for the
        # loss and loss_weights dictionaries, arguments of model.compile
        # TODO: define the names globally elsewhere to ensure consistency
        out_f0_rgb          = layers.Lambda( lambda x: x, name="out_f0_rgb"  )( dec_f0_rgb )
        out_f0_car          = layers.Lambda( lambda x: x, name="out_f0_car"  )( dec_f0_car )
        out_f0_lane         = layers.Lambda( lambda x: x, name="out_f0_lane" )( dec_f0_lane )
        out_f1_rgb          = layers.Lambda( lambda x: x, name="out_f1_rgb"  )( dec_f1_rgb )
        out_f1_car          = layers.Lambda( lambda x: x, name="out_f1_car"  )( dec_f1_car )
        out_f1_lane         = layers.Lambda( lambda x: x, name="out_f1_lane" )( dec_f1_lane )
        out_f2_rgb          = layers.Lambda( lambda x: x, name="out_f2_rgb"  )( dec_f2_rgb )
        out_f2_car          = layers.Lambda( lambda x: x, name="out_f2_car"  )( dec_f2_car )
        out_f2_lane         = layers.Lambda( lambda x: x, name="out_f2_lane" )( dec_f2_lane )

        model_outputs       = [
            out_f0_rgb,
            out_f0_car,
            out_f0_lane,
            out_f1_rgb,
            out_f1_car,
            out_f1_lane,
            out_f2_rgb,
            out_f2_car,
            out_f2_lane
        ]

        return models.Model(
                    inputs      = [ frame0, frame1 ],
                    outputs     = model_outputs,
                    name        = mname
        )



# ===========================================================================================================


class RecMultiAE( MultipleAE, RecAE ):
    """ -----------------------------------------------------------------------------------------------------
    Create a network composed by a common encoder, a latent space, a common decoder
    for the self-image, and a two-input recurrent network using the common decoder for predicting
    the next frame

    Few notes:
    - all layers with weights of encoder and decoder are shared, and it is a source of troubles for the naming
      of placeholders (especially Tensorflow internal placeholders) and their reuse, when computing gradients
    - this is the reason for a lot of naming, including in define_model() Lambda layers for naming only
    ----------------------------------------------------------------------------------------------------- """

    def __init__( self ):
        """ -------------------------------------------------------------------------------------------------
        Initialize class attributes usign the config dictionary, and create the network model
        ------------------------------------------------------------------------------------------------- """
        self.n_input            = None          # [int] number of input frames
        self.n_output           = None          # [int] number of output frames
        self.recurr             = None          # [str] kind of recurrent layer (RNN/GRU/LSTM)
        
        if DEBUG: print ( "\nstart RecMultiAE.__init__()" )
        super( RecMultiAE, self ).__init__()

        self.loss_func              = {
            "out_f0_rgb"  : get_loss( self.loss ),
            "out_f0_car"  : get_loss( 'UXE_CAR' ),
            "out_f0_lane" : get_loss( 'UXE_LANE' ),
            "out_f1_rgb"  : get_loss( self.loss ),
            "out_f1_car"  : get_loss( 'UXE_CAR' ),
            "out_f1_lane" : get_loss( 'UXE_LANE' ),
            "out_f2_rgb"  : get_loss( self.loss ),
            "out_f2_car"  : get_loss( 'UXE_CAR' ),
            "out_f2_lane" : get_loss( 'UXE_LANE' )
        }
        if DEBUG: print ( "\nend RecMultiAE.__init__()" )


    def define_model( self, mname='RecMultiAE' ):
        """ -------------------------------------------------------------------------------------------------
        Define the entire structure of the network

        mname:          [str] name of the model

        return:         [keras.engine.training.Model]
        ------------------------------------------------------------------------------------------------- """
        # load the main submodels
        enc                 = self.define_encoder()
        sz                  = self.latent_parts( self.split )
        dec_rgb             = self.define_decoder( ( sz[ 0 ], ), mname='decoder_rgb' )
        dec_car             = self.define_decoder_segm( ( sz[ 1 ], ), mname='decoder_car' )
        dec_lane            = self.define_decoder_segm( ( sz[ 1 ], ), mname='decoder_lane' )
        rec                 = self.define_recurrent()               # FIXME pass here different name of layer

        # placeholders for the two input frames
        frame0              = layers.Input( shape=self.img_size, name="frame0" )
        frame1              = layers.Input( shape=self.img_size, name="frame1" )

        # encode the two input frames
        lat_frame0          = enc( frame0 )
        lat_frame1          = enc( frame1 )

        # compute the third latent with the recursive submodel
        stack_frames        = layers.Lambda( lambda x: K.stack( x, axis=1 ), name="frame_sequence" )
        sequence            = stack_frames( [ lat_frame0, lat_frame1 ] )
        lat_frame2          = rec( sequence )

        # now split latents to feed the separated decoders
        lat_f0_split        = self.split_latent( lat_frame0, split=self.split )
        lat_f1_split        = self.split_latent( lat_frame1, split=self.split )
        lat_f2_split        = self.split_latent( lat_frame2, split=self.split )

        # decoders for all frames and components
        dec_f0_rgb          = dec_rgb( lat_f0_split[ 0 ] )
        dec_f0_car          = dec_car( lat_f0_split[ 1 ] )
        dec_f0_lane         = dec_lane( lat_f0_split[ 2 ] )
        dec_f1_rgb          = dec_rgb( lat_f1_split[ 0 ] )
        dec_f1_car          = dec_car( lat_f1_split[ 1 ] )
        dec_f1_lane         = dec_lane( lat_f1_split[ 2 ] )
        dec_f2_rgb          = dec_rgb( lat_f2_split[ 0 ] )
        dec_f2_car          = dec_car( lat_f2_split[ 1 ] )
        dec_f2_lane         = dec_lane( lat_f2_split[ 2 ] )

        # dummy layers necessary for the multiple outputs
        # otherwise Keras uses as name of the output the name of the last layer, which is shared
        # NOTE that the names here used are somehow special, because are the keys for the
        # loss and loss_weights dictionaries, arguments of model.compile
        # TODO: define the names globally elsewhere to ensure consistency
        out_f0_rgb          = layers.Lambda( lambda x: x, name="out_f0_rgb"  )( dec_f0_rgb )
        out_f0_car          = layers.Lambda( lambda x: x, name="out_f0_car"  )( dec_f0_car )
        out_f0_lane         = layers.Lambda( lambda x: x, name="out_f0_lane" )( dec_f0_lane )
        out_f1_rgb          = layers.Lambda( lambda x: x, name="out_f1_rgb"  )( dec_f1_rgb )
        out_f1_car          = layers.Lambda( lambda x: x, name="out_f1_car"  )( dec_f1_car )
        out_f1_lane         = layers.Lambda( lambda x: x, name="out_f1_lane" )( dec_f1_lane )
        out_f2_rgb          = layers.Lambda( lambda x: x, name="out_f2_rgb"  )( dec_f2_rgb )
        out_f2_car          = layers.Lambda( lambda x: x, name="out_f2_car"  )( dec_f2_car )
        out_f2_lane         = layers.Lambda( lambda x: x, name="out_f2_lane" )( dec_f2_lane )

        model_outputs       = [
            out_f0_rgb,
            out_f0_car,
            out_f0_lane,
            out_f1_rgb,
            out_f1_car,
            out_f1_lane,
            out_f2_rgb,
            out_f2_car,
            out_f2_lane
        ]

        return models.Model(
                    inputs      = [ frame0, frame1 ],
                    outputs     = model_outputs,
                    name        = mname
        )


