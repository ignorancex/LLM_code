"""
#############################################################################################################

Functions for training a model

    Alice   2019

#############################################################################################################
"""

import  os
import  sys
import  datetime

import  numpy                               as np
from    math                import ceil, sqrt, inf
from    keras               import utils, models, callbacks, preprocessing
from    keras               import backend  as K

import  matplotlib
matplotlib.use( 'agg' )     # to use matplotlib with unknown 'DISPLAY' var (when using remote display)
from    matplotlib          import pyplot   as plt

import  mesg                                as ms
from    gener               import gen_dataset, len_dataset
from    exec_dset           import load_dset_time


SHUFFLE         = True
DEBUG           = True

cnfg            = []        # NOTE initialized by 'nn_main.py'

dir_check       = 'chkpnt'
nn_best         = 'nn_best.h5'

# dataset for timepred
time_train_set  = None
time_valid_set  = None
time_test_set   = None
time_test_str   = None



def rearrange_rtime( train_set, valid_set, test_set ):
    """ -----------------------------------------------------------------------------------------------------
    rearrange datasets in case of recursive prediction in time
    see NOTE at trainer.train_time_model() for the difference in format
    between prediction with multiple Dense and prediction with recursion
    ----------------------------------------------------------------------------------------------------- """
    x   = np.array( train_set[ 0 ] )
    x   = x.swapaxes( 0, 1 )
    y   = train_set[ 1 ]
    train_set   = ( x, y )

    x   = np.array( valid_set[ 0 ] )
    x   = x.swapaxes( 0, 1 )
    y   = valid_set[ 1 ]
    valid_set   = ( x, y )

    x   = np.array( test_set[ 0 ] )
    x   = x.swapaxes( 0, 1 )
    y   = test_set[ 1 ]
    test_set    = ( x, y )

    return train_set, valid_set, test_set



def generate_dset_time():
    """ -----------------------------------------------------------------------------------------------------
    Load dataset
    ----------------------------------------------------------------------------------------------------- """
    global time_train_set, time_valid_set, time_test_set, time_test_str
    
    # load dataset for prediction in feature space
    dset            = load_dset_time( cnfg[ 'dset_time' ] )
    time_train_set  = dset[ 0 ]
    time_valid_set  = dset[ 1 ]
    time_test_set   = dset[ 2 ]
    time_train_str  = dset[ 3 ]
    time_valid_str  = dset[ 4 ]
    time_test_str   = dset[ 5 ]

    if cnfg[ 'arch_class' ] in ( 'RTIME', 'R2TIME', 'RMTIME' ):
        time_train_set, time_valid_set, time_test_set   = rearrange_rtime(
                time_train_set,
                time_valid_set,
                time_test_set
    )




class ChangeKlWeight( callbacks.Callback ):
    """ -----------------------------------------------------------------------------------------------------
    update the weight of the Kullback–Leibler divergence component
    in vae_loss, starting with very low values and progressively increasing
    all parameters of the increasing function are taken from the nn object

    nn:             [AE/VAE/MultipleVAE object]
    ----------------------------------------------------------------------------------------------------- """

    def __init__( self, nn, steps_per_epoch ):
        super( ChangeKlWeight, self ).__init__()
        self.kl_weight          = nn.kl_weight      # this is the symbol to a backend tensor with the actual weight
        self.kl_wght            = nn.kl_wght        # the largest weight value
        self.kl_incr            = nn.kl_incr        # the increment rate
        self.kl_weight_0        = nn.kl_weight_0    # the initial value of the weight
        self.steps_per_epoch    = steps_per_epoch
        self.iter               = 0
        self.below05            = True


    def kl_annealing( self ):
        """ ------------------------------------------------------------------------------------------------
        compute the current value of the KL weight
        the number of iterations is normalized by the number of steps per epoch
        ------------------------------------------------------------------------------------------------- """
        n_iter  = self.iter / self.steps_per_epoch
        d       = ( self.kl_wght - self.kl_weight_0 ) * self.kl_incr ** n_iter
        return self.kl_wght - d


    def on_batch_begin( self, batch, logs ):
        """ ------------------------------------------------------------------------------------------------
        this is a standard method invoked by the callback, the arguments are mandatory
        but are not used in this implementation
        ------------------------------------------------------------------------------------------------- """
        kl_updated_w    = self.kl_annealing()
        K.set_value( self.kl_weight, kl_updated_w )
        if DEBUG:
            ms.print_msg( cnfg[ 'log_msg' ], "at iteration {:^10d} kl_weight: {:^8.6f}".format( self.iter, kl_updated_w ) )
        if self.below05 and kl_updated_w > 0.5:
            ms.print_msg( cnfg[ 'log_msg' ], "at iteration {:^10d} kl_weight has reached 0.5".format( self.iter ) )
            self.below05    = False
        self.iter       += 1


    def on_epoch_end( self, epoch, logs ):
        """ ------------------------------------------------------------------------------------------------
        this is a standard method invoked by the callback, the arguments are mandatory
        used here only for messaging
        ------------------------------------------------------------------------------------------------- """
        kl_updated_w    = self.kl_annealing()
        ms.print_msg( cnfg[ 'log_msg' ], "at iteration {:^10d} kl_weight: {:^8.6f}".format( self.iter, kl_updated_w ) )



def set_callback( nn, steps_per_epoch=None ):
    """ -----------------------------------------------------------------------------------------------------
    Set of functions to call during the training procedure
        - save checkpoint of the best model so far
        - save 'n_check' checkpoints during training
        - optionally save information for TensorBoard
        - end training after 'patience' unsuccessful epochs
    nn:                 [AE/VAE/MultipleVAE object]
    steps_per_epoch:    [int]

    return:             [list] list of keras.callbacks.ModelCheckpoint
    ----------------------------------------------------------------------------------------------------- """
    calls   = []

    if cnfg[ 'n_check' ] != 0:
        period  = ceil( cnfg[ 'n_epochs' ] / cnfg[ 'n_check' ] )

        if cnfg[ 'n_check' ] > 0:
            calls.append( callbacks.ModelCheckpoint(
                    os.path.join( cnfg[ 'dir_current' ], nn_best ),
                    save_best_only          = True,
                    save_weights_only       = True,
                    period                  = 1
            ) )

        if cnfg[ 'n_check' ] > 1:
            p       = os.path.join( cnfg[ 'dir_current' ], dir_check )
            fname   = os.path.join( p, "check_{epoch:04d}.h5" )
            os.makedirs( p )
            calls.append( callbacks.ModelCheckpoint(
                        fname,
                        save_weights_only   = True,
                        period              = period
            ) )

    if cnfg[ 'tboard' ]:
        calls.append( callbacks.TensorBoard(
                    log_dir                 = cnfg[ 'dir_current' ],
                    histogram_freq          = 0,
                    batch_size              = 1,
                    write_graph             = True,
                    write_grads             = False,
                    write_images            = True
        ) )

    if cnfg[ 'patience' ] > 0:
        calls.append( callbacks.EarlyStopping( monitor='val_loss', patience=cnfg[ 'patience' ] ) )

    if hasattr( nn, "kl_incr" ) and nn.kl_incr > 0.0:
        calls.append( ChangeKlWeight( nn, steps_per_epoch ) )

    return calls



def train_ae_model( nn ):
    """ -----------------------------------------------------------------------------------------------------
    Training procedure for autoencoder models

    nn:             [AE/VAE/MultipleVAE object]

    return:         [keras.callbacks.History], [datetime.timedelta]
    ----------------------------------------------------------------------------------------------------- """

    # train and valid dataset generators
    train_feed, valid_feed                      = gen_dataset( cnfg[ 'dir_dset' ] )
    train_samples, valid_samples, test_samples  = len_dataset( cnfg[ 'dir_dset' ] )

    # train using multiple GPUs
    if cnfg[ 'n_gpus' ] > 1:
        model                   = utils.multi_gpu_model( nn.model, gpus=cnfg[ 'n_gpus' ] )
    else:
        model                   = nn.model

    t_start                     = datetime.datetime.now()       # starting time of execution
    steps_per_epoch             = ceil( train_samples / cnfg[ 'batch_size' ] )
    validation_steps            = ceil( valid_samples / cnfg[ 'batch_size' ] )

    history = model.fit_generator(
            train_feed,
            epochs              = cnfg[ 'n_epochs' ],
            validation_data     = valid_feed,
            steps_per_epoch     = steps_per_epoch,
            validation_steps    = validation_steps,
            callbacks           = set_callback( nn, steps_per_epoch ),
            verbose             = 2,
            shuffle             = SHUFFLE
    )

    t_end   = datetime.datetime.now()                           # end time of execution

    return history, ( t_end - t_start )



def train_time_model( nn ):
    """ -----------------------------------------------------------------------------------------------------
    Training procedure for model predicting in time

    model:          [keras.engine.training.Model]

    NOTE: there is a different format of the datasets for Dense-based prediction or recursive prediction
    in the Dense-based case
    x:              [list with length n_input] of arrays with shape (n_samples, latent_size)
    while in the recursive case:
    x:              [numpy array] with shape (n_samples, n_input, latent_size)

    return:         [keras.callbacks.History], [datetime.timedelta]
    ----------------------------------------------------------------------------------------------------- """
    global time_train_set, time_valid_set, time_test_set, time_test_str
    generate_dset_time()

    if cnfg[ 'arch_class' ] == 'TIME':
        train_samples   = len( time_train_set[ 0 ][ 0 ] )
        valid_samples   = len( time_valid_set[ 0 ][ 0 ] )
    else:
        train_samples   = time_train_set[ 0 ].shape[ 0 ]
        valid_samples   = time_valid_set[ 0 ].shape[ 0 ]

    # train using multiple GPUs
    if cnfg[ 'n_gpus' ] > 1:
        model                   = utils.multi_gpu_model( nn.model, gpus=cnfg[ 'n_gpus' ] )
    else:
        model                   = nn.model

    t_start                     = datetime.datetime.now()       # starting time of execution

    history = model.fit(
            x                   = time_train_set[ 0 ],
            y                   = time_train_set[ 1 ],
            epochs              = cnfg[ 'n_epochs' ],
            validation_data     = time_valid_set,
            steps_per_epoch     = ceil( train_samples // cnfg[ 'batch_size' ] ),
            validation_steps    = ceil( valid_samples // cnfg[ 'batch_size' ] ),
            callbacks           = set_callback( nn ),
            verbose             = 2,
            shuffle             = SHUFFLE
    )

    t_end   = datetime.datetime.now()                           # end time of execution

    return history, ( t_end - t_start )



def plot_history( history, fname='loss' ):
    """ -----------------------------------------------------------------------------------------------------
    Plot the loss performance during training

    history:        [keras.callbacks.History]
    fname:          [str] name of output file without extension
    ----------------------------------------------------------------------------------------------------- """
    train_loss  = history.history[ 'loss' ]
    valid_loss  = history.history[ 'val_loss' ]
    epochs      = range( 1, len( train_loss ) + 1 )

    plt.plot( epochs, train_loss, 'r--' )
    plt.plot( epochs, valid_loss, 'b-' )
    plt.legend( [ 'Training Loss', 'Validation Loss' ] )
    plt.xlabel( 'Epoch' )
    plt.ylabel( 'Loss' )
    plt.savefig( "{}.pdf".format( fname ) )

    if len( train_loss ) > 5:
        m   = np.mean( train_loss )
        s   = np.std( train_loss )
        plt.ylim( [ m - s, m + s ] )
        plt.savefig( "{}_zoom.pdf".format( fname ) )

    plt.close()
