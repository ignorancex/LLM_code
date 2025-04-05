"""
#############################################################################################################

Evaluation utilities for trained neural models

    Alice   2019

#############################################################################################################
"""

import  os
import  sys
import  numpy                           as np

from    keras       import losses, optimizers
from    keras       import backend      as K

import  cnfg                            as cf
import  arch                            as ar
import  tester                          as ts
import  gener                           as gn
from    arch        import nn_wght
from    trainer     import nn_best


DEBUG               = False
EVAL                = False
PRED                = True
# TODO: currently implementation of inspect, for MVAE models, works with split=0 only!
INSPECT             = False

dset_root           = 'dataset'
dir_dataraw         = os.path.join( dset_root, 'synthia/scaled_256_128' )
dir_log             = 'log'
dir_eval            = 'eval'
dir_eval_b          = 'eval_best'
dir_eval_l          = 'eval_last'
dir_inspect_b       = 'inspect_best'
dir_inspect_l       = 'inspect_last'
dir_cnfg            = 'config'
probe_img           = 'dataset/synthia/link/rgb_S/test/img/S06WR_000273_RGB_FR.jpg'

cnfg                = None
dt_list             = ( 'train', 'test', 'valid' )       # sub-datasets to evaluate (can also include 'valid' or 'train')
test_set            = None              # used only for 'arch_class' == TIME
test_str            = None              # used only for 'arch_class' == TIME
train_set           = None              # used only for 'arch_class' == TIME
train_str           = None              # used only for 'arch_class' == TIME

default_loss        = losses.mean_squared_error


# ===========================================================================================================
#
#   Utilities for writing to file
#
#   - write_header
#   - frmt_result
#
# ===========================================================================================================

def write_header( fd ):
    """ -----------------------------------------------------------------------------------------------------
    Write the file header

    fd:           [_io.TextIOWrapper] file descriptor for results
    ----------------------------------------------------------------------------------------------------- """
    h   = 'model'.center( 17 )

    for dt in dt_list:
        h   += '\t' + dt.center( 50 )
    fd.write( h + '\n' )

    h   = 17 * ' '
    sh  = '\t{:^10}{:^10}{:^30}'.format( 'mean', 'std', 'worst' )
    for dt in dt_list:
        h   += sh
    fd.write( h + '\n' )

    sep = ( 17 + len( dt_list ) * ( 50 + len( '\t'.expandtabs() ) ) ) * '-'
    fd.write( sep + '\n' )



def frmt_result( d ):
    """ -----------------------------------------------------------------------------------------------------
    Format a string with the results of a model evaluation

    d:              [dict] dictionary with image file name as key and error as value

    return:         [str] formatted result
    ----------------------------------------------------------------------------------------------------- """
    vals    = np.array( list( d.values() ) )
    mean    = vals.mean()
    std     = vals.std()
    worst   = max( d, key = lambda x: d[ x ] )
    return  '{:^10.6f}{:^10.6f}{:^10.6f}({:^20})'.format( mean, std, d[ worst ], worst )



# ===========================================================================================================
#
#   Creation of the model from the saved data
#
#   - recreate_model
#   - get_encoder
#   - get_decoder
#   - get_multi_decoders
#
# ===========================================================================================================

def recreate_model( dr, f_model=None ):
    """ -----------------------------------------------------------------------------------------------------
    From a folder of results, recreate the model architecture and load its weights

    dr:             [str] path to folder with model results
    f_model:        [str] path to the model weights to load

    return:         [AE/VAE/MultiVAE/Timepred object]
    ----------------------------------------------------------------------------------------------------- """
    global cnfg, test_set, test_str, train_set, train_str, valid_set, valid_str

    if DEBUG:   print( "Loading model from {}".format( dr ) )

    # if config already validated, keep it
    restore = False
    if cnfg is not None:
        restore = True
        cnfg_copy   = {}
        for k in cnfg.keys():
            cnfg_copy[ k ]  = cnfg[ k ]

    # load config from saved file
    if not os.path.isdir( dr ):
        print( "Folder {} not found".format( dr ) )
        sys.exit( 1 )

    dr_cnfg     = os.path.join( dr, dir_cnfg )
    if not os.path.isdir( dr_cnfg ):
        print( "Folder {} not found".format( dr_cnfg ) )
        sys.exit( 1 )

    f_cnfg              = os.path.join( dr_cnfg, os.listdir( dr_cnfg )[ 0 ] )
    cnfg                = cf.get_config( f_cnfg )
    cnfg[ 'n_gpus' ]    = 0     # forces CPU usage, maybe not necessary


    ar.cnfg             = cnfg
    ts.cnfg             = cnfg
    gn.cnfg             = cnfg

    # create model from config file and HDF5 file
    if f_model is None:
        f_model         = os.path.join( dr, nn_best )
        if not os.path.isfile( f_model ):
            print( "Warning: file {} not found".format( f_model ) )
            f_model     = os.path.join( dr, nn_wght )
            if not os.path.isfile( f_model ):
                print( "Error: file {} not found".format( f_model ) )
                sys.exit( 1 )

    ar.PLOT     = False
    nn  = ar.create_model()
    if not ar.load_model( nn.model, f_model ):
        print( "Failed to load weigths into the model" )
        sys.exit( 1 )

    # generate dataset for prediction in feature space - maybe not necessary, but easier to
    # mirror the code in exec_main
    if cnfg[ 'arch_class' ] == 'TIME':
        x, y, xs, ys        = gn.dset_time(
                cnfg[ 'dset_latent' ],
                cnfg[ 'dset_odom' ],
                cnfg[ 'n_input' ],
                cnfg[ 'n_output' ],
                step    = cnfg[ 'step' ]
        )
        train_set, valid_set, test_set, train_str, valid_str, test_str  = gn.dset_time_neat(
                x,
                y,
                xs,
                ys,
                size    = cnfg[ 'dset_size' ]
        )

    if not hasattr( nn.model, 'loss' ):
        nn.model.compile( optimizer=optimizers.SGD(), loss=default_loss )
    else:
        nn.model.loss = default_loss

    if restore:
        for k in cnfg_copy.keys():
            cnfg[ k ]       = cnfg_copy[ k ]
        ar.cnfg             = cnfg
        ts.cnfg             = cnfg
        gn.cnfg             = cnfg

    return nn

    

def get_encoder( model ):
    """ -----------------------------------------------------------------------------------------------------
    Extract the encoder model from an entire *AE architecture

    model:          [keras.engine.training.Model]

    return:         [keras.engine.training.Model]
    ----------------------------------------------------------------------------------------------------- """
    enc     = [ l for l in model.layers if l.name == 'encoder' ][ 0 ]

    if enc.name != 'encoder': 
        print( "Structure '{}' not valid as encoder architecture".format( enc.name ) )
        sys.exit( 1 )

    if enc.layers[ -1 ].name != 'z_mean': 
        print( "Last layer '{}' is not the expected 'z_mean'".format( enc.layers[ -1 ].name ) )
        sys.exit( 1 )

    return enc

    

def get_decoder( model ):
    """ -----------------------------------------------------------------------------------------------------
    Extract the decoder model from an entire (M)*AE architecture

    model:          [keras.engine.training.Model]

    return:         [keras.engine.training.Model or list of them]
    ----------------------------------------------------------------------------------------------------- """
    decod   = [ l for l in model.layers if 'decoder' in l.name ]

    for dec in decod:
        if 'decoder' not in dec.name: 
            print( "Structure '{}' not valid as decoder architecture".format( dec.name ) )
            sys.exit( 1 )

        if dec.layers[ 0 ].input_shape[ -1 ] != cnfg[ 'latent_size' ]: 
            if 'latent_subsize' not in cnfg: 
                print( "Input size '{:d}' is not the expected '{}'".format(
                            dec.layers[ 0 ].input_shape[ -1 ], cnfg[ 'latent_size' ] ) )
                sys.exit( 1 )

            elif dec.layers[ 0 ].input_shape[ -1 ] != cnfg[ 'latent_subsize' ]: 
                print( "Input size '{:d}' is not the expected '{}'".format(
                            dec.layers[ 0 ].input_shape[ -1 ], cnfg[ 'latent_subsize' ] ) )
                sys.exit( 1 )

    # just to try a random latent space
    # ts.save_image( dec.predict( rnd.uniform(-3, 3) * np.random.rand(1,128) ), True, "i.jpg" )

    if len( decod ) == 1:
        return decod[ 0 ]

    return decod



def get_multi_decoders( model ):
    """ -----------------------------------------------------------------------------------------------------
    Extract the decoder model from a MultiVae architecture

    model:          [keras.engine.training.Model]

    return:         3 * [keras.engine.training.Model]
    ----------------------------------------------------------------------------------------------------- """

    dec = [ None, None, None ]
    for l in model.layers:
        if l.name == 'decoder_rgb': 
            dec[ 0 ]    = l
        if l.name == 'decoder_car': 
            dec[ 1 ]    = l
        if l.name == 'decoder_lane': 
            dec[ 2 ]    = l
            

    for d in dec:
        if d is None:
            print( "Structure '{}' lacks a decoder".format( model.name ) )
            sys.exit( 1 )

    return dec



# ===========================================================================================================
#
#   Evaluation and prediction
#
#   - eval_model
#   - eval_models
#
#   - pred_model
#   - pred_models
#
# ===========================================================================================================

def eval_model_set( nn, dir_tset, fname=None ):
    """ -----------------------------------------------------------------------------------------------------
    Evaluate a single given model on a single dataset

    nn:             [keras.engine.training.Model]
    dir_tset:       [str] full path to the dataset
    fname:          [str] optional name of the file for writing results of a model

    return:         [dict] keys are image file names, values are the evaluation errors
    ----------------------------------------------------------------------------------------------------- """

    # AE/VAE
    if cnfg[ 'data_class' ] in ( 'FRAME', 'RGB' ):
        tset_rgb    = os.path.join( dir_tset, 'img' )                                       # input/target test set
        d           = ts.eval_tset( nn, tset_rgb, tset_rgb, metric='MSE', fname=fname )     # loss evaluations

    # AE/VAE segm
    if cnfg[ 'data_class' ] in ( 'CAR', 'LANE' ):
        tset_rgb    = os.path.join( dir_tset, 'rgb', 'img' )                                # input test set
        tset_class  = os.path.join( dir_tset, cnfg[ 'data_class' ].lower(), 'img' )         # target test set
        d           = ts.eval_tset( nn, tset_rgb, tset_class, metric='MSE', fname=fname )   # loss evaluations

    # TODO MVAE
    if cnfg[ 'data_class' ] == 'MULTI':
        tset_rgb    = os.path.join( dir_tset, 'rgb', 'img' )                                # input/target test set
        tset_car    = os.path.join( dir_tset, 'car', 'img' )                                # target test set
        tset_lane   = os.path.join( dir_tset, 'lane', 'img' )                               # target test set
        d           = ts.eval_multitset( nn, tset_rgb, tset_car, tset_lane, fname=fname )   # loss evaluations
        if fname is None:                                                                   # if more than one models
            d   = d[ 0 ]

    if isinstance( d, np.ndarray ):     # only if 'ts.eval_tset()' uses 'eval_raw()'
        d   = d.mean()

    return d



def eval_model( dr, fname='evals' ):
    """ -----------------------------------------------------------------------------------------------------
    Evaluate a single models over all datasets, writing separate files for each dataset

    dr:         [str] full path to the folder with model results
    fname:      [str] base pathname of output files
    ----------------------------------------------------------------------------------------------------- """
    nn  = recreate_model( dr )

    for dt in dt_list:
        dir_tset    = os.path.join( cnfg[ 'dir_dset' ], dt )
        f_eval      = fname + '_' + dt + '.txt'
        d           = eval_model_set( nn, dir_tset, fname=f_eval )


def eval_models( dr_list, fname='results.txt' ):
    """ -----------------------------------------------------------------------------------------------------
    Evaluate the list of models passed as argument, and write a text file with the evaluation errors

    dr_list:    [list of str] full path to folders with model results
    fname:      [str] pathname of output file
    ----------------------------------------------------------------------------------------------------- """
    f       = open( results, 'w' )
    write_header( f )

    for dr in dr_list:
        nn  = recreate_model( dr )
        r   = dr

        for dt in dt_list:
            dir_tset    = os.path.join( cnfg[ 'dir_dset' ], dt )
            d           = eval_model_set( nn, dir_tset )
            r           += '\t' + frmt_result( d )

        f.write( r + '\n' )
        K.clear_session()

    f.close()



def inspect_model( nn, dest ):
    """ -----------------------------------------------------------------------------------------------------
    Plots of internal weights and responses to a probe image for a model

    nn:             [keras.engine.training.Model] the model to evaluate
    dest:           [str] path to the destination folder
    ----------------------------------------------------------------------------------------------------- """

    if cnfg[ 'arch_class' ] == 'VAE':
        nt  = ts.create_test_model( nn )                # create models with outputs at all layers
        ts.model_weights( nt, dest )
        ts.model_outputs( nt, probe_img, dest )
        return True

    if cnfg[ 'arch_class' ] == 'MVAE':
        nt  = ts.create_test_model( nn, branch='RGB' )  # create models with outputs at all layers, RGB branch
        ts.model_weights( nt, dest )
        ts.model_outputs( nt, probe_img, dest )
        return True

    return False



def do_inspect_model( dr ):
    """ -----------------------------------------------------------------------------------------------------
    wrapper to inspect_model, managing both last and best model weights

    dr:             [str] path to model folder
    ----------------------------------------------------------------------------------------------------- """
    best        = False
    last        = False
    f_best      = os.path.join( dr, nn_best )
    f_last      = os.path.join( dr, nn_wght )
    if os.path.isfile( f_last ):
        last        = True
        f_model     = f_last
    if os.path.isfile( f_best ):
        best        = True
        f_model     = f_best

    both        = best and last

    if both:
        dest        = os.path.join( dr, dir_inspect_b )
        if not os.path.exists( dest ):
            os.makedirs( dest )
        dest_l      = os.path.join( dr, dir_inspect_l )
        if not os.path.exists( dest_l ):
            os.makedirs( dest_l )
    else:
        dest        = os.path.join( dr, dir_inspect )
        if not os.path.exists( dest ):
            os.makedirs( dest )

    nn  = recreate_model( dr, f_model=f_model )
    inspect_model( nn, dest )

    if both:
        K.clear_session()
        nn  = recreate_model( dr, f_model=f_last )
        inspect_model( nn, dest_l )



def pred_time( nn, dr ):
    """ -----------------------------------------------------------------------------------------------------
    Plot the prediction of a time model over a dataset of images

    nn:             [keras.engine.training.Model] the model to evaluate
    NOTE:           needs update!
    ----------------------------------------------------------------------------------------------------- """
    dec         = recreate_model( cnfg[ 'decod' ] )
    dec         = get_decoder( dec )
   #ts.pred_time( nn, dec, test_set, test_str, de, dir_dataraw, samples=6 )
    ts.pred_time( nn, dec, train_set, train_str, de, dir_dataraw, samples=6 )

    return False
        

def pred_set( nn, dest, dir_tset, prefix ):
    """ -----------------------------------------------------------------------------------------------------
    Plot the prediction of a model over a dataset of images

    nn:             [keras.engine.training.Model] the model to evaluate
    dest:           [str] path where to write results
    dir_tset:       [str] path of the dataset to use
    prefix:         [str] prefix for the output filenames
    ----------------------------------------------------------------------------------------------------- """
    eval_file   = os.path.join( dest, "eval_" + prefix + "set.txt" )

    # AE/VAE
    if cnfg[ 'data_class' ] in ( 'FRAME', 'RGB' ):
        tset_rgb    = os.path.join( dir_tset, 'img' )                                   # target test set
        ts.pred_tset( nn, tset_rgb, tset_rgb, dest, prfx=prefix )                         # predict images
        return True

    # AE/VAE segm
    if cnfg[ 'data_class' ] in ( 'CAR', 'LANE' ):
        tset_rgb    = os.path.join( dir_tset, 'rgb', 'img' )                            # input test set
        tset_class  = os.path.join( dir_tset, cnfg[ 'data_class' ].lower(), 'img' )     # target test set
        ts.pred_tset( nn, tset_rgb, tset_class, dest, prfx=prefix )                       # predict images
        return True

    # MVAE
    if cnfg[ 'data_class' ] == 'MULTI':
        tset_rgb    = os.path.join( dir_tset, 'rgb', 'img' )
        tset_car    = os.path.join( dir_tset, 'car', 'img' )
        tset_lane   = os.path.join( dir_tset, 'lane', 'img' )
        ts.pred_tset( nn, tset_rgb, tset_rgb, dest, multi_class='RGB', prfx=prefix )
        ts.pred_tset( nn, tset_rgb, tset_car, dest, multi_class='CAR', prfx=prefix ) 
        ts.pred_tset( nn, tset_rgb, tset_lane, dest, multi_class='LANE', prfx=prefix )
        return True

    return False
        


def pred_model( dr ):
    """ -----------------------------------------------------------------------------------------------------
    Plot the prediction of a model over a dataset of images

    dr:             [str] path to model folder
    ----------------------------------------------------------------------------------------------------- """
    best        = False
    last        = False
    f_best      = os.path.join( dr, nn_best )
    f_last      = os.path.join( dr, nn_wght )
    if os.path.isfile( f_last ):
        last        = True
        f_model     = f_last
    if os.path.isfile( f_best ):
        best        = True
        f_model     = f_best

    both        = best and last

    if both:
        dest        = os.path.join( dr, dir_eval_b )
        if not os.path.exists( dest ):
            os.makedirs( dest )
        dest_l      = os.path.join( dr, dir_eval_l )
        if not os.path.exists( dest_l ):
            os.makedirs( dest_l )
    else:
        dest        = os.path.join( dr, dir_eval )
        if not os.path.exists( dest ):
            os.makedirs( dest )

    nn  = recreate_model( dr, f_model=f_model )
    for d in dt_list:
        dir_tset    = os.path.join( cnfg[ 'dir_dset' ], d )
        prefix      = d[ : 2 ] + '_'
        pred_set( nn, dest, dir_tset, prefix )

    if both:
        K.clear_session()
        nn  = recreate_model( dr, f_model=f_last )
        for d in dt_list:
            dir_tset    = os.path.join( cnfg[ 'dir_dset' ], d )
            prefix      = d[ : 2 ] + '_'
            pred_set( nn, dest_l, dir_tset, prefix )

    # TIME PRED
    if cnfg[ 'arch_class' ] == 'TIME':
        pred_time( nn, dr )
        return True

    return True
        


def pred_models( dr_list ):
    """ -----------------------------------------------------------------------------------------------------
    Plot the predictions of a list of models, over the same dataset of images

    dr_list:    [list of str] full path to folders with model results
    ----------------------------------------------------------------------------------------------------- """
    for dr in dr_list:
        nn          = recreate_model( dr )
        dir_tset    = os.path.join( cnfg[ 'dir_dset' ], 'test' )
        pred_model( nn, dr )

        K.clear_session()
        

        
def recall_img( dr, img ):
    """ -----------------------------------------------------------------------------------------------------
    display the image(s) predicted by a model

    dr:             [str] path to model folder
    img:            [str] path to the image
    ----------------------------------------------------------------------------------------------------- """
    if not os.path.isfile( img ):
        print( "Error: image file {} not found".format( img ) )
        sys.exit( 1 )

    best        = False
    last        = False
    f_best      = os.path.join( dr, nn_best )
    f_last      = os.path.join( dr, nn_wght )
    if os.path.isfile( f_last ):
        last        = True
        f_model     = f_last
    if os.path.isfile( f_best ):
        best        = True
        f_model     = f_best

    both    = best and last


    if both:
        prefix      = "best_"
        pref_last   = "last_"
    else:
        prefix      = ""

    nn      = recreate_model( dr, f_model=f_model )
    if cnfg[ 'arch_class' ] == 'MVAE':
        i_rgb, i_car, i_lane    = ts.pred_image( nn, img )
        ts.save_image( i_rgb, True, prefix + "rgb_pred.jpg" )
        ts.save_image( i_car, False, prefix + "car_pred.jpg" )
        ts.save_image( i_lane, False, prefix + "lane_pred.jpg" )
    else:
        out     = ts.pred_image( nn, img )
        ts.save_image( out, ts.out_rgb( nn ), prefix + "pred.jpg" )

    if both:
        K.clear_session()
        nn      = recreate_model( dr, f_model=f_last )
        prefix  = pref_last
        if cnfg[ 'arch_class' ] == 'MVAE':
            i_rgb, i_car, i_lane    = ts.pred_image( nn, img )
            ts.save_image( i_rgb, True, prefix + "rgb_pred.jpg" )
            ts.save_image( i_car, False, prefix + "car_pred.jpg" )
            ts.save_image( i_lane, False, prefix + "lane_pred.jpg" )
        else:
            out     = ts.pred_image( nn, img )
            ts.save_image( out, ts.out_rgb( nn ), prefix + "pred.jpg" )




# ===========================================================================================================
#
#   usage example
#   python src/exec_eval.py -m log/best_car
#               -i dataset/synthia/scaled_256_128/S05_winternight/S05WN_000286_RGB_FL.jpg 
#
# ===========================================================================================================
if __name__ == '__main__':
    ar.TRAIN    = False
    ar.PLOT     = False
    args        = cf.get_args_eval()

    # if a single model is passed
    if args[ 'MODEL' ] is not None:
        if EVAL:
            eval_model( args[ 'MODEL' ] )

        if PRED:
            pred_model( args[ 'MODEL' ] )

        if INSPECT:
            do_inspect_model( args[ 'MODEL' ] )


        # if an image path is passed
        if args[ 'IMAGE' ] is not None:
            recall_img( args[ 'MODEL' ], args[ 'IMAGE' ] )

                
        # if a list of image folders is passed
        if args[ 'IMAGES' ] is not None:
            nn  = recreate_model( args[ 'MODEL' ] )
            s   = args[ 'IMAGES' ]
            if not s[ 0 ].isalpha():    s = s[ 1: ]
            if not s[ -1 ].isalpha():   s = s[ :-1 ]
            dr_seqs = s.split( ',' )
                
            nm      = args[ 'MODEL' ].split( '/' )[ -1 ]
            for seq in dr_seqs:
                ts.pred_seq( nn, seq, nm )


    # if a list of models is passed
    if args[ 'MODELS' ] is not None:
        s   = args[ 'MODELS' ]
        if not s[ 0 ].isalpha():    s = s[ 1: ]
        if not s[ -1 ].isalpha():   s = s[ :-1 ]
        dr_list = s.split( ',' )

        if EVAL:
            eval_models( dr_list )

        if PRED:
            pred_models( dr_list )
