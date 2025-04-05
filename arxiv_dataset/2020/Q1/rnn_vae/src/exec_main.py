"""
#############################################################################################################

AutoEncoder neural network models

    Alice   2019

#############################################################################################################
"""

import  numpy                       as np
import  random                      as rn
import  cnfg                        as cf

# NOTE seed must be set before importing anything from Keras or TF
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
if __name__ == '__main__':
    args                = cf.get_args()
    cnfg                = cf.get_config( args[ 'CONFIG' ] )
    np.random.seed( cnfg[ 'seed' ] )
    rn.seed( cnfg[ 'seed' ] )

import  os
import  sys
import  time
import  datetime

import  tensorflow                  as tf
from    keras       import backend  as K

import  mesg                        as ms
import  arch                        as ar
import  trainer                     as tr
import  tester                      as ts
import  gener                       as gn
import  exec_eval                   as ev
import  pred                        as pr


SAVE                = False

FRMT                = "%y-%m-%d_%H-%M-%S"

dset_root           = 'dataset'
dir_res             = 'res'
dir_log             = 'log'
dir_eval            = 'eval'
dir_cnfg            = 'config'
dir_src             = 'src'

dir_dataraw         = os.path.join( dset_root, 'synthia/scaled_256_128' )
dir_city            = os.path.join( dset_root, 'synthia/samples/city' )
dir_freeway         = os.path.join( dset_root, 'synthia/samples/freeway' )

log_err             = os.path.join( dir_log, "log.err" )
log_out             = os.path.join( dir_log, "log.out" )
log_msg             = os.path.join( dir_log, "log.msg" )
log_time            = os.path.join( dir_log, "log.time" )

nn                  = None
dir_current         = None
log_duration        = None

test_set            = None      # used only for 'arch_class' == TIME
test_str            = None      # used only for 'arch_class' == TIME

time_models         = ( 'TIME', 'RTIME', 'R2TIME', 'RMTIME' ) # models predicting in time

# -----------------------------------------------------------------------------------------------------------
# some sample images used for quick testing

list_tset           = ( 'S01_summer', 'S02_summer', 'S04_summer', 'S05_summer', 'S06_summer',
        'S04_dawn', 'S04_fall', 'S04_fog', 'S04_night', 'S04_rainnight', 'S04_softrain', 'S04_spring',
        'S04_sunset', 'S04_winter', 'S04_winternight'
)
list_timepred       = (
        ( os.path.join( dir_dataraw, 'S01_summer' ),      'S01SM_000071_RGB_FR' ),
        ( os.path.join( dir_dataraw, 'S02_night' ),       'S02NT_000167_RGB_FR' ),
        ( os.path.join( dir_dataraw, 'S04_softrain' ),    'S04SR_000310_RGB_FR' ),
        ( os.path.join( dir_dataraw, 'S05_fog' ),         'S05FG_000073_RGB_FR' ),
        ( os.path.join( dir_dataraw, 'S06_sunset' ),      'S06ST_000254_RGB_FR' )
)
list_hallux         = (
        ( os.path.join( dir_dataraw, 'S01_summer' ),      'S01SM_000066_RGB_FR' ),
        ( os.path.join( dir_dataraw, 'S02_night' ),       'S02NT_000001_RGB_FR' ),
        ( os.path.join( dir_dataraw, 'S04_softrain' ),    'S04SR_000114_RGB_FR' ),
        ( os.path.join( dir_dataraw, 'S05_fog' ),         'S05FG_000237_RGB_FR' ),
        ( os.path.join( dir_dataraw, 'S06_sunset' ),      'S06ST_000092_RGB_FR' )
)
list_interp         = [
        os.path.join( dir_dataraw, "S06_spring/S06SP_000394_RGB_FL.jpg" ),
        os.path.join( dir_dataraw, "S01_dawn/S01DW_000155_RGB_FL.jpg" ),
        os.path.join( dir_dataraw, "S05_night/S05NT_000168_RGB_FL.jpg" ),
        os.path.join( dir_dataraw, "S04_summer/S04SM_000136_RGB_FL.jpg" )
]
list_interp.append( list_interp[ 0 ] )



def init_config():
    """ -----------------------------------------------------------------------------------------------------
    Initialization
    ----------------------------------------------------------------------------------------------------- """
    global dir_current, log_duration

    dir_current     = os.path.join( dir_res, time.strftime( FRMT ) )
    os.makedirs( dir_current )
    os.makedirs( os.path.join( dir_current, dir_log ) )
    log_duration    = os.path.join( dir_current, log_time )

    # redirect stderr and stdout in log files
    if args[ 'REDIRECT' ]:
        le          = os.path.join( dir_current, log_err )
        lo          = os.path.join( dir_current, log_out )
        sys.stderr  = open( le, 'w' )
        sys.stdout  = open( lo, 'w' )

        # how to restore
        # sys.stdout = sys.__stdout__

    #exec( "from {} import cnfg".format( args[ 'CONFIG' ] ) )

    cnfg[ 'dir_current' ]   = dir_current
    cnfg[ 'log_msg' ]       = os.path.join( dir_current, log_msg )

    # visible GPUs - must be here, before all the keras stuff
    n_gpus  = eval( args[ 'GPU' ] )
    if isinstance( n_gpus, int ):
        os.environ[ "CUDA_VISIBLE_DEVICES" ]    = str( list( range( n_gpus ) ) )[ 1 : -1 ]
    elif isinstance( n_gpus, ( tuple, list ) ):
        os.environ[ "CUDA_VISIBLE_DEVICES" ]    = str( n_gpus )[ 1 : -1 ]
        n_gpus                                  = len( n_gpus )
    else:
       ms.print_err( "GPUs specification {} not valid".format( n_gpus ) )
    cnfg[ 'n_gpus' ]    = n_gpus

    # GPU memory fraction
    if n_gpus > 0:
        tf.set_random_seed( cnfg[ 'seed' ] )
        tf_cnfg                                             = tf.ConfigProto()
        tf_cnfg.gpu_options.per_process_gpu_memory_fraction = args[ 'FGPU' ]
        tf_session                                          = tf.Session( config=tf_cnfg )
        K.set_session( tf_session )

    # TODO quite brute way to share globals...
    ar.cnfg     = cnfg
    tr.cnfg     = cnfg
    ts.cnfg     = cnfg
    gn.cnfg     = cnfg
    ev.cnfg     = cnfg
    pr.cnfg     = cnfg



def create_model():
    """ -----------------------------------------------------------------------------------------------------
    Model architecture and weight training
    ----------------------------------------------------------------------------------------------------- """
    global nn, test_set, test_str

    ar.TRAIN    = args[ 'TRAIN' ]
    if cnfg[ 'arch_class' ] != 'ITIME':
        nn          = ar.create_model()

    if args[ 'LOAD' ] is not None:
        if not ar.load_model( nn.model, args[ 'LOAD' ] ):
           ms.print_err( "Failed to load weights from {}".format( args[ 'LOAD' ] ) )

    if args[ 'TRAIN' ]:

        if cnfg[ 'arch_class' ] in time_models:
            history, train_duration = tr.train_time_model( nn )
        else:
            history, train_duration = tr.train_ae_model( nn )

        tr.plot_history( history, os.path.join( dir_current, 'loss' ) )

        # save duration of training
        with open( log_duration, 'a' ) as f:
            f.write( "Training duration:\t{}\n".format( str( train_duration ) ) )

        if SAVE:
            ar.save_model( nn.model )



def test_model():
    """ -----------------------------------------------------------------------------------------------------
    Test routines
    ----------------------------------------------------------------------------------------------------- """

    # reload model in non-trainable version, with the best weights obtained
    K.clear_session()
    ar.TRAIN    = False
    ar.PLOT     = False

    print( '\nLoading model for testing. Wait and hope...\n' )

    if cnfg[ 'arch_class' ] != 'ITIME':
        nn          = ar.create_model()
        if args[ 'LOAD' ] is not None:
            if not ar.load_model( nn.model, args[ 'LOAD' ] ):
               ms.print_wrn( "Failed to load weights from {}".format( args[ 'LOAD' ] ) )

            # if the model is loaded, the dataset has not been created because it is supposed to be done in trainer.py
            if cnfg[ 'arch_class' ] in time_models:
                tr.generate_dset_time()

        else:
            if not ar.load_model( nn.model, os.path.join( dir_current, tr.nn_best ) ):
               ms.print_wrn( "Failed to load weights from {}".format( tr.nn_best ) )

    # folder where to save predict/evaluation
    de          = os.path.join( dir_current, dir_eval )
    dei         = os.path.join( de, 'interp' )
    deh         = os.path.join( de, "hallux" )
    det         = os.path.join( de, "timepred" )

    if not os.path.exists( de ):
        os.makedirs( de )
    eval_file   = os.path.join( de, "eval_tset.txt" )
    accur_file  = os.path.join( de, "accur_tset.txt" )


    # -------- TIME PRED --------
    if cnfg[ 'arch_class' ] in ( 'ITIME', ) + time_models:
        dec_obj     = ev.recreate_model( cnfg[ 'decod' ] )
        dec         = ev.get_decoder( dec_obj.model )
        enc         = ev.get_encoder( dec_obj.model )

        if cnfg[ 'arch_class' ] in ( 'RTIME', 'R2TIME', 'RMTIME' ):
            if isinstance( dec, ( list, tuple ) ):

                # hallucination
                if args[ 'HALLX' ]:
                    for n, ( d, f ) in enumerate( list_hallux ): 
                        ts.hallucinate( nn.model, dec, dec_obj, d, f, deh, suffx='_{}'.format( n+1 ), n_iter=50 )

                # interpolation between two frames
                if args[ 'INTRP' ]:
                    for i in range( len( list_interp ) - 1 ) :
                        ts.interpolate_pred( enc, dec, list_interp[ i ], list_interp[ i+1 ], model_obj=dec_obj,
                                save=dei, suffx='_{}'.format( i+1 ) )

                # time prediction
                for n, ( d, f ) in enumerate( list_timepred ):
                    ts.pred_time_sample( nn.model, dec, dec_obj, d, f, det, suffx='_{}'.format( n+1 ) )
                ts.pred_time( nn.model, dec, tr.time_test_set, tr.time_test_str, de, dir_dataraw, odom=False, dec_obj=dec_obj )

                # accuracy scores
                if args[ 'ACCUR' ] == 1:
                    ts.accur_sel_time( nn.model, dec, dec_obj, fname=os.path.join( de, "accur_sel.txt" ) )
                if args[ 'ACCUR' ] == 2:
                    ts.accur_tset_time( nn.model, dec, dec_obj, tr.time_test_str, fname=os.path.join( de, "accur_tset.txt" ) )

            else:
                ts.pred_time( nn.model, dec, tr.time_test_set, tr.time_test_str, de, dir_dataraw, odom=False )

        if cnfg[ 'arch_class' ] == 'TIME':
            ts.pred_time( nn.model, dec, tr.time_test_set, tr.time_test_str, de, dir_dataraw )

        elif cnfg[ 'arch_class' ] == 'ITIME':
            model       = pr.TimeInterPred()
            ts.pred_time( model, dec, tr.time_test_set, tr.time_test_str, de, dir_dataraw )

    else:
        dir_tset    = os.path.join( cnfg[ 'dir_dset' ], 'test' )

        # -------- AE/VAE --------
        if cnfg[ 'data_class' ] == 'RGB':
            tset_rgb    = os.path.join( dir_tset, 'img' )                                       # input/target test set

            ts.pred_tset( nn.model, tset_rgb, tset_rgb, de )                                    # plot of predictions

            # interpolation between two frames
            if args[ 'INTRP' ]:
                enc         = ev.get_encoder( nn.model )
                dec         = ev.get_decoder( nn.model )
                ts.interpolate_pred( enc, dec, i01, i02, save=de )

            # loss evaluations
            if args[ 'EVAL' ]:
                ts.eval_tset( nn.model, tset_rgb, tset_rgb, metric='MSE', fname=eval_file )


        # -------- AE/VAE segm --------
        elif cnfg[ 'data_class' ] in ( 'CAR', 'LANE' ):
            tset_rgb    = os.path.join( dir_tset, 'rgb', 'img' )                                # input test set
            tset_class  = os.path.join( dir_tset, cnfg[ 'data_class' ].lower(), 'img' )         # target test set

            ts.pred_tset( nn.model, tset_rgb, tset_class, de )                                  # plot of predictions

            # accuracy scores
            if args[ 'ACCUR' ] == 2:
                ts.accur_tset( nn.model, tset_rgb, tset_class, fname=accur_file )
                for n in list_tset:
                    ts.accur_tset(
                            nn.model,
                            "{}/{}".format( dir_dataraw, n ),
                            "{}/{}".format( dir_dataraw, n ),
                            input_cond  = lambda x: '_RGB_' in x,
                            target_cond = lambda x: '_SEGM_{}_'.format( cnfg[ 'data_class' ].upper() ) in x,
                            fname       = os.path.join( de, "accur_{}.txt".format( n ) ),
                            epsilon     = 1000
                    )

            # loss evaluations
            if args[ 'EVAL' ]:
                ts.eval_tset( nn.model, tset_rgb, tset_class, metric='MSE', fname=eval_file )


        # -------- MVAE --------
        elif cnfg[ 'data_class' ] == 'MULTI':
            dec         = ev.get_decoder( nn.model )
            enc         = ev.get_encoder( nn.model )
            tset_rgb    = os.path.join( dir_tset, 'rgb/img' )                                   # input/target test set
            tset_car    = os.path.join( dir_tset, 'car/img' )                                   # target test set
            tset_lane   = os.path.join( dir_tset, 'lane/img' )                                  # target test set

            ts.pred_multi_tset( nn.model, tset_rgb, ( tset_rgb, tset_car, tset_lane ), de, raw=False )

            # predictions over a selected set of images
            if args[ 'PRED' ]:
                ts.pred_folder( nn.model, dir_city, os.path.join( de, 'city' ), multi_output=True )
                ts.pred_folder( nn.model, dir_freeway, os.path.join( de, 'freeway' ), multi_output=True )

            # interpolation between two frames
            if args[ 'INTRP' ]:
                for i in range( len( list_interp ) - 1 ) :
                    ts.interpolate_pred( enc, dec, list_interp[ i ], list_interp[ i+1 ], model_obj=nn,
                            save=dei, suffx='_{}'.format( i+1 ) )

            # loss evaluations
            if args[ 'EVAL' ]:
                ts.eval_multitset( nn.model, tset_rgb, tset_car, tset_lane, fname=eval_file )

            # prediction from latent dataset
            ts.eval_latent_folder( dir_city, os.path.join( de, 'city' ), dec, dec_obj=nn )
            ts.eval_latent_folder( dir_freeway, os.path.join( de, 'freeway' ), dec, dec_obj=nn )

            # accuracy from latent dataset
            ts.accur_class_latent( dec, nn, os.path.join( de, "accur_lat.txt" ) )



        # -------- RGBSEQ (RVAE) --------
        elif cnfg[ 'data_class' ] == 'RGBSEQ':
            ts.pred_rec_tset( nn.model, dir_tset, de )


        # -------- MUTLISEQ (RMVAE,RMAE) --------
        elif cnfg[ 'data_class' ] == 'MULTISEQ':

            if args[ 'PRED' ]:
                ts.pred_multirec_tset( nn.model, dir_tset, de )

            # accuracy scores
            if args[ 'ACCUR' ] == 1:
                ts.accur_class_multiseq( nn.model, fname=os.path.join( de, "accur_sel.txt" ) )
            if args[ 'ACCUR' ] == 2:
                # full test accuracy yet to be implemented...
                ts.accur_class_multiseq( nn.model, fname=os.path.join( de, "accur_sel.txt" ) )

            # interpolation between two frames
            if args[ 'INTRP' ]:
                dec         = ev.get_decoder( nn.model )
                enc         = ev.get_encoder( nn.model )
                for i in range( len( list_interp ) - 1 ) :
                    ts.interpolate_pred( enc, dec, list_interp[ i ], list_interp[ i+1 ], model_obj=nn,
                            save=dei, suffx='_{}'.format( i+1 ) )



def archive():
    """ -----------------------------------------------------------------------------------------------------
    Archiving
    ----------------------------------------------------------------------------------------------------- """
    if args[ 'ARCHIVE' ] > 0:
        # save config files
        if args[ 'ARCHIVE' ] >= 1:
            d       = os.path.join( dir_current, dir_cnfg )
            cfile   =  args[ 'CONFIG' ]
            os.makedirs( d )
            os.system( "cp {} {}".format( cfile, d ) )

        # save python sources
        if args[ 'ARCHIVE' ] >= 2:
            d       = os.path.join( dir_current, dir_src )
            pfile   = "src/*.py"
            os.makedirs( d )
            os.system( "cp {} {}".format( pfile, d ) )



# ===========================================================================================================
#
#   MAIN
#
# ===========================================================================================================

if __name__ == '__main__':
    t_start     = datetime.datetime.now()
    init_config()
    archive()
    create_model()

    if args[ 'TEST' ]:
        tt_start    = datetime.datetime.now()
        test_model()
        tt_end      = datetime.datetime.now()

        # save duration of testing
        with open( log_duration, 'a' ) as f:
            f.write( "Testing duration:\t{}\n".format( str( tt_end - tt_start ) ) )

    # save total duration
    t_end   = datetime.datetime.now()
    with open( log_duration, 'a' ) as f:
        f.write( "Total duration:\t\t{}\n".format( str( t_end - t_start ) ) )

    print( '\nFinished!\n' )
