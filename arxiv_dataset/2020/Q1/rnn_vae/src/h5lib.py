"""
#############################################################################################################

Read HDF5 binary files of trained models

    Alice   2019

#############################################################################################################
"""


import  h5py
import  re
import  numpy   as np

import  mesg    as ms

w_pttrn = re.compile( '_W_[0-9]+:[0-9]+|kernel:0' ) # pattern name of the weight key in HDF5 saved models
b_pttrn = re.compile( '_b_[0-9]+:[0-9]+|bias:0' )   # pattern name of the bias key in HDF5 saved models
w_key   = "kernel:0"                # standard name of the weight key in H5 saved models
b_key   = "bias:0"                  # standard name of the bias key in H5 saved models
r_key   = "recurrent_kernel:0"      # standard name of the recurrent weight key in H5 saved models

valid_prfx   = (                                    # prefixes of layer names with weights to load
    'conv',
    'dnse',
    'dcnv',
    'z_mean',
    'simple_rnn',
    'pure_rnn',                      # FIXME temporary fix!
    'rnn_'
)

invalid_prfx = (                                    # prefixes of layer names that should be excluded
    'frame',
    'out_',
    'input',
    'pool',
    'lambda'
)



def valid_layer( name ):
    """ -----------------------------------------------------------------------------------------------------
    Check whether a layer has weights to load

    name:           [str] name of the layer

    return:         [bool] True if the layer is valid
    ----------------------------------------------------------------------------------------------------- """
    for valid in valid_prfx:
        if valid in name:
            return True

    return False



def invalid_layer( name ):
    """ -----------------------------------------------------------------------------------------------------
    Check whether a layer should be skipped during loading of weights

    name:           [str] name of the layer

    return:         [bool] True if the layer is NOT valid
    ----------------------------------------------------------------------------------------------------- """
    for invalid in invalid_prfx:
        if invalid in name:
            return True

    return False



def load_h5_layer( layer, h5 ):
    """ -----------------------------------------------------------------------------------------------------
    Load weights (and biases if existing) of a layer inside the HDF5 group, and put them in the given
    layer of the model.

    The HDF5 group should not have sub-groups, except for a single sub-group with exactly the same name.

    layer:          one of [keras.layers] layer to be filled with the weights and biases
    h5:             [h5py._hl.group.Group] single H5 group (without sub-groups)

    return:         [bool] False if the loading process fails
    ----------------------------------------------------------------------------------------------------- """
    wb          = layer.get_weights()
    has_bias    = False
    has_recur   = False

    if len( wb ) == 1:
        w0          = wb[ 0 ]               # weights
    elif len( wb ) == 2:
        w0, b0      = wb                    # weights and biases
        has_bias    = True
    elif len( wb ) == 3:
        w0, r0, b0  = wb                    # weights, recurrent weights and biases
        has_bias    = True
        has_recur   = True
    else:
        ms.print_wrn( "Found too many values ({}) when extracting weights of layer {}".format( len( wb ), layer.name ) )
        return False

    # if there is no weight key in h5, it may be a group with a single sub-group with exactly the same name
    if w_key not in h5.keys():
        h5      = h5[ list( h5.keys() )[ 0 ] ]

        """
        if len( list( h5.keys() ) ) > 1:
            ms.print_wrn( "Expected one sub-group in HDF5 file for layer {}, found {} instead".format(
                        layer.name, len( list( h5.keys() ) ) )
            )
        """

        # if still there is no weight key in h5
        if w_key not in h5.keys():
            ms.print_wrn( "No weights found in HDF5 file for layer {}".format( layer.name ) )
            return False

    w   = np.array( h5[ w_key ] )           # weights in the HDF5 file

    if w.shape != w0.shape:
        ms.print_wrn( "Mismatched weights for layer {}, {} vs {}".format( layer.name, w.shape, w0.shape ) )

    if has_bias:
        if b_key not in h5.keys():
            ms.print_wrn( "No biases found in HDF5 file for layer {}".format( layer.name ) )
            return False

        b   = np.array( h5[ b_key ] )       # biases in the HDF5 file

        if has_recur:
            if r_key not in h5.keys():
                ms.print_wrn( "No recurrent weights found in HDF5 file for layer {}".format( layer.name ) )
                return False

            r   = np.array( h5[ r_key ] )   # recurrent weights in the HDF5 file

            layer.set_weights( [ w, r, b ] )
            return True


        layer.set_weights( [ w, b ] )
        return True

    layer.set_weights( [ w ] )
    return True



def load_h5_group( model, h5, single=False ):
    """ -----------------------------------------------------------------------------------------------------
    Load weights (and biases if existing) contained in the layers inside the HDF5 group, and put them
    in the corresponding layers of the given model.

    The HDF5 group can have sub-groups, in that case use 'single' = False.

    model:          [keras.engine.training.Model] model to be filled with the weights and biases
    h5:             [h5py._hl.group.Group] single H5 group (but can have sub-groups)
    single:         [bool] True if h5 has no sub-groups

    return:         [bool] False if the loading process fails
    ----------------------------------------------------------------------------------------------------- """
    names   = [ l.name for l in model.layers ]

    for k in h5.keys():
        if not valid_layer( k ):
            continue

        if k not in names:
            if single:
                continue
            ms.print_wrn( "The layer '{}' of the HDF5 file was not found inside the model".format( k ) )
            return False

        l       = model.get_layer( k )

        if not load_h5_layer( l, h5[ k ] ):
            return False

    return True



def h5_single_group( model, h5 ):
    """ -----------------------------------------------------------------------------------------------------
    Load weights (and biases if existing) from a single HDF5 group, and put them in the given model.

    model:          [keras.engine.training.Model] model to be filled with the weights and biases
    h5:             [h5py._hl.group.Group] single HDF5 group

    return:         [bool] False if the loading process fails
    ----------------------------------------------------------------------------------------------------- """
    names   = [ l.name for l in model.layers ]

    for i, n in enumerate( names ):
        if 'input' in n:
            continue

        if not load_h5_group( model.layers[ i ], h5, single=True ):
            return False

    return True



def h5_multi_groups( model, h5 ):
    """ -----------------------------------------------------------------------------------------------------
    Load weights (and biases if existing) from multiple HDF5 groups, and put them in the given model.

    An architecture has multiple HDF5 groups when it is defined by multiple sub-models
    (e.g. and encoder model and a decoder model together).

    model:          [keras.engine.training.Model] model to be filled with the weights and biases
    h5:             [h5py._hl.files.File] HDF5 main structure

    return:         [bool] False if the loading process fails
    ----------------------------------------------------------------------------------------------------- """
    names   = [ l.name for l in model.layers ]

    for i, n in enumerate( names ):
        if invalid_layer( n ):
            continue

        if n not in h5.keys():
            ms.print_wrn( "The layer '{}' of the model was not found inside the HDF5 file".format( n ) )
            return False

        if not load_h5_group( model.layers[ i ], h5[ n ], single=False ):
            return False

    return True



def load_h5( model, h5_file ):
    """ -----------------------------------------------------------------------------------------------------
    Load weights (and biases if existing) from a HDF5, and put them in the given model.

    model:          [keras.engine.training.Model] model to be filled with the weights and biases
    h5_file:        [str] pathname to the HDF5 file

    return:         [bool] False if the loading process fails
    ----------------------------------------------------------------------------------------------------- """
    try:
        h5  = h5py.File( h5_file, 'r' )
    except:
        ms.print_wrn( "Error opening HDF5 file " + h5_file )
        return False
        
    # this is the case where each HDF5 group corresponds to a layer of the network
    # e.g. any model trained on a single GPU, without internal submodels like encoder/decoder (verified with AE)
    for k in h5.keys():
        if valid_layer( k ):
            return load_h5_group( model, h5 )
        
    # this is the case where only the HDF5 group with key equal to the model's name contains all the significant layers
    # e.g. any model trained on multiple GPUs (verified with AE/VAE/MVAE)
    if model.name in h5.keys():
        return h5_single_group( model, h5[ model.name ] )

    # this is the case where there are several HDF5 groups containing significant layers
    # e.g. any model trained on a single GPU, with internal submodels like encoder/decoder (verified with VAE/MVAE)
    return h5_multi_groups( model, h5 )



def load_h5_vgg16( model, h5_file ):
    """ -----------------------------------------------------------------------------------------------------
    Load weights and biases from the HDF5 file of the VGG16 model trained on ImageNet.
    Only the part of the model corresponding to the VGG16 encoder is filled with the weights.

    model:          [keras.engine.training.Model] entire model to be partially filled with the weights
    h5_file:        [str] pathname to the HDF5 file

    return:         [bool] False if the loading process fails
    ----------------------------------------------------------------------------------------------------- """
    if model.name != 'AE':
        enc_model   = model.layers[ 1 ]
        assert 'enc' in enc_model.name                                  # the encoder part of the model
    else:
        enc_model   = model

    try:
        h5  = h5py.File( h5_file, 'r' )
    except:
        ms.print_wrn( "Error opening HDF5 file " + h5_file )
        
    for i, k in enumerate( h5.keys() ):
        if valid_layer( k ):
            # the numeration in the model is shifted by one position w.r.t. the h5, because of the input layer
            layer   = enc_model.layers[ i+1 ]
            w0, b0  = layer.get_weights()

            for j in h5[ k ].keys():
                if w_pttrn.search( j ):                 # match the pattern corresponding to the weights key
                    w   = np.array( h5[ k ][ j ] )
                elif b_pttrn.search( j ):               # match the pattern corresponding to the biases key
                    b   = np.array( h5[ k ][ j ] )
                else:
                    ms.print_wrn( "Unexpected strcture '{}' found in HDF5 file".format( j ) )
                    return False

            if w.shape != w0.shape:
                ms.print_wrn( "Mismatched weights for layer '{}', {} vs {}".format( layer.name, w.shape, w0.shape ) )
                return False

            layer.set_weights( [ w, b ] )

    return True
