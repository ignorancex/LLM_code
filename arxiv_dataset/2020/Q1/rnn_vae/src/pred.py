"""
#############################################################################################################

non neural time preditctions

    Alice   2019

#############################################################################################################
"""

import  os
import  re
import  numpy               as np

from    scipy.interpolate   import interp1d

cnfg            = []                    # NOTE initialized by 'nn_main.py'


# ===========================================================================================================


class TimeInterPred:
    """ -----------------------------------------------------------------------------------------------------
    Create a predictor of future frames in the feature space by interploation, without odometry data
    ----------------------------------------------------------------------------------------------------- """

    def __init__( self, kind="linear" ):
        """ -------------------------------------------------------------------------------------------------
        Initialize class attributes usign the config dicitonary

        kind:           [str] one of "linear", "zero", "quadratic", "cubic", see scipy.interpolate.interp1d
                        documentation for meaning
        ------------------------------------------------------------------------------------------------- """
        self.n_input                = None          # [int] number of input frames
        self.n_output               = None          # [int] number of output frames
        self.latent_size            = None          # [int] size of the latent space
        self.kind                   = None          # [str] kind of interpolation

        
        # initialize class attributes with values from cnfg dict
        for k in self.__dict__:
            if k not in cnfg:
                ms.print_err( "Attribute '{}' of class '{}' not indicated".format( k, self.__class__ ) )
            exec( "self.{} = cnfg[ '{}' ]".format( k, k ) )                              


    def _extrapolate( self, y ):
        """ -------------------------------------------------------------------------------------------------

        y:              [numpy.ndarray] time series for one single dimension of the latent
        
        return:         [float] predicted vale
        ------------------------------------------------------------------------------------------------- """
        x   = np.arange( self.n_input )
        f   = interp1d( x, y, kind=self.kind, fill_value="extrapolate", assume_sorted=True )
        if self.n_output == 1:
            return f( self.n_input )

        return np.array( [ f( self.n_input + i ) for i in range( self.n_output ) ] )


    def _predict( self, latents ):
        """ -------------------------------------------------------------------------------------------------

        latents:        [numpy.ndarray] input latents, with shape (n_input,latent_size)
        
        return:         [numpy.ndarray] predicted latent
        ------------------------------------------------------------------------------------------------- """
        assert latents.shape == ( self.n_input, self.latent_size )

        prediction  = [ self._extrapolate( latents[ :, i ] ) for i in range( self.latent_size ) ]

        return np.array( prediction )


    def predict( self, latents ):
        """ -------------------------------------------------------------------------------------------------

        latents:        [numpy.ndarray] array of input latents, with shape (n_input,n_samples,latent_size)
        
        return:         [numpy.ndarray] predicted latent
        ------------------------------------------------------------------------------------------------- """
        n_input, n_samples, latent_size = latents.shape
        assert latents.shape == ( self.n_input, n_samples, self.latent_size )

        prediction  = [ self._predict( latents[ :, i, : ] ) for i in range( n_samples ) ]
        prediction  = np.array( prediction )

        if self.n_output == 1:
            return prediction

        prediction  = prediction.swapaxes( 1, 2 )
        prediction  = prediction.swapaxes( 0, 1 )
        return prediction
