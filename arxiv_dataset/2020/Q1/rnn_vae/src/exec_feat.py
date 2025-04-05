"""
#############################################################################################################

Generate a dataset of feature (latent space) codifications computed by an already trained VAE model

    Alice   2019

#############################################################################################################
"""

import  os
import  h5py

import  exec_eval   as ev
import  arch        as ar
import  tester      as ts
import  cnfg        as cf


DEBUG       = True
dir_h5      = 'dataset/synthia/latent'


def gen_latent( model, dset, data_class ):
    """ -----------------------------------------------------------------------------------------------------
    Generate a dictionary of latent-space encodings computed by the model, over a given dataset of images

    model:          [keras.engine.training.Model] encoder model ending with latent space 'z_mean'
    dset:           [str] path to plain folder of frames (like 'scaled_256_128')
    data_class:     [str] code describing the class of frames to consider (RGB_FR,RGB_FL)

    return:         [dict]
    ----------------------------------------------------------------------------------------------------- """
    dd  = {}

    for dpath, dnames, fnames in os.walk( dset ):
        for f in fnames:
            if f.endswith( '.jpg' ) and data_class in f:
                n           = f.split( '.' )[ 0 ]                               # code name of frame
                l           = ts.pred_image( model, os.path.join( dpath, f ) )  # output of encoder
                dd[ n ]     = l
                if DEBUG:   print( n )

    return dd



def dset_latent( model, model_name, dset ):
    """ -----------------------------------------------------------------------------------------------------
    Generate a new dataset of latent-space encodings of all images in a given dataset.

    model:          [keras.engine.training.Model] full model
    model_name:     [str] name of folder containing model results
    dset:           [str] path to plain folder of frames (like 'scaled_256_128')
    ----------------------------------------------------------------------------------------------------- """
    if not os.path.exists( dir_h5 ):
        os.makedirs( dir_h5 )

    hn      = os.path.join( dir_h5, model_name ) + '.h5'
    
    print( ">>>>>>>", hn )

    hf      = h5py.File( hn, 'w' )
    if DEBUG:   print( "Creating file {}".format( hn ) )

    e       = ev.get_encoder( model )
    d       = gen_latent( e, dset, 'RGB' )

    for k in d.keys():
        hf.create_dataset( k, data=d[ k ] )
    hf.close()



# ===========================================================================================================

if __name__ == '__main__':
    ar.TRAIN    = False
    ar.PLOT     = False
    args        = cf.get_args_eval()

    if args[ 'MODEL' ] is not None:
        nn      = ev.recreate_model( args[ 'MODEL' ] )
        mn      = args[ 'MODEL' ].split( '/' )[ -1 ]
        if mn == '':
            mn      = args[ 'MODEL' ].split( '/' )[ -2 ]

        dt      = "dataset/synthia/scaled_256_128"
        dset_latent( nn.model, mn, dt )

# ===========================================================================================================
