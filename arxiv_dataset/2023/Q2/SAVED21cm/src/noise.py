import numpy as np
np.random.seed(2)

class Noise:
    def __init__(self, nu, nLST, ant, power, deltaNu, deltaT):
        """Initialize the thermal noise model.

        Args:
            nu (array): Frequency range
            nLST (int): Number of time bins to fit
            ant (list): List of antenna designs
            power (array): Total power in (K)
            deltaNu (int): Frequency channel width
            deltaT (int): Integration time
        """
        self.nu = nu
        self.nLST = nLST
        self.ants = len(ant)
        self.power = power
        self.deltaNu = deltaNu
        self.deltaT = deltaT
    
    def thermLevel(self):
        """Thermal noise level.

        Returns:
            array: Thermal noise level from Tb/sqrt(dNu*dT)
        """
        noiselev = self.power/np.sqrt(self.deltaNu*pow(10, 6)*self.deltaT*3600)
        return noiselev

    def noiseRealz(self):
        """Random realization of the noise.

        Returns:
            array: Random realization based on a given thermal noise level
        """
        grvec = np.random.randn(*self.thermLevel().shape)
        return self.thermLevel()*grvec
    
    def covmat(self):
        """Noise covariance matrix.

        Returns:
            array: Covariance matrix for noise
        """
        covmat = np.zeros(shape=(len(self.nu)*self.nLST*self.ants,
                                 len(self.nu)*self.nLST*self.ants))
        for i in range(len(self.nu)*self.nLST*self.ants):
            covmat[i, i] = pow(self.thermLevel()[i], 2.0)
        return covmat

    def covmatInv(self):
        """Inverse of noise covariance matrix.

        Returns:
            array: Inverse of noise covariance matrix
        """
        return self.invDiag(self.covmat())
    
    def wgtTs(self, modset, opt):
        """Noise covariance weighted modelling set.

        Args:
            modset (array): Modelling set
            opt (string): Option to choose ('FG' or '21')

        Returns:
            array: Noise covariance weighted modelling set.
        """
        if opt == '21':
            exp21 = np.identity(len(self.nu))
            exp21 = np.tile(exp21, (self.nLST*self.ants, 1))
            covmat21Inv = np.matmul(exp21.T, self.mulDiag(self.covmatInv(), exp21))
            wgtTs = self.mulDiag(np.sqrt(covmat21Inv), modset)
        if opt == 'FG':
            wgtTs = self.mulDiag(np.sqrt(self.covmatInv()), modset)
        return wgtTs
        
    @staticmethod
    def invDiag(A):
        B = np.zeros(A.shape)
        np.fill_diagonal(B, 1./np.diag(A))
        return B

    @staticmethod
    def mulDiag(A, B):
        return np.multiply(np.diag(A)[:, None], B)
