import numpy as np

class Basis:
    def __init__(self, nu, nLST, ant):
        """Initialize basis for the foregrounds and 21cm signal.

        Args:
            nu (array): Frequency range
            nLST (int): Number of time bins to fit
            ant (list): List of antenna designs
        """
        self.nu = nu
        self.nLST = nLST
        self.ant = ant
    
    def wgtSVDbasis(self, modset, covmat, opt, loadPath=None, save=False, saveName=None):
        """This methods returns the noise covariance weighted modes obtained from
        the Singular Value Decomposition of the training sets.

        Args:
            modset (array): Modelling set
            covmat (array): Noise covariance matrix
            opt (string): Option to choose ('FG' or '21')
            loadPath (string, optional): File (.npy) path to load the basis. Defaults to None.
            save (bool, optional): To save the basis. Defaults to False.
            saveName (string, optional): Filename to save the basis. Defaults to None.

        Returns:
            array: Noise covariance weighted SVD modes
        """
        if loadPath is None:
            print('Basis: Performing SVD of %s modelling set...'%opt, end='', flush=True)
            if opt == 'FG':
                F, _, _ = np.linalg.svd(modset, full_matrices=False)
                F = self.mulDiag(np.sqrt(covmat), F)

            if opt == '21':
                F, _, _ = np.linalg.svd(modset, full_matrices=False)
                exp21 = np.identity(len(self.nu))
                exp21 = np.tile(exp21, (self.nLST*len(self.ant), 1))
                covmatInv = self.invDiag(covmat)
                covmat21Inv = np.matmul(exp21.T, self.mulDiag(covmatInv, exp21))
                covmat21 = self.invDiag(covmat21Inv)
                F = self.mulDiag(np.sqrt(covmat21), F)
            if save:
                import os
                if not os.path.exists('BasisOutput/'):
                    os.mkdir('BasisOutput/')
                np.save('BasisOutput/%s'%saveName, F)
            print('Done!')
        else:
            print('Basis: Loading the %s basis...'%opt, end='', flush=True)
            F = np.load(loadPath)
            print('Done!')
        return F
    
    def unwgtSVDbasis(self, modset):
        """This method returns the (unweighted) modes simply obtained from the 
        Singular Value Decomposition of the modelling set.

        Args:
            modset (array): Modelling set

        Returns:
            array: Unweighted SVD modes
        """
        F, _, _ = np.linalg.svd(modset, full_matrices=False)
        return F

    def getOverlapFg(self, basis, cmatInv):
        """Computes the overlap between foreground modes.

        Args:
            basis (array): foreground modes
            cmatInv (array): noise covariance matrix

        Returns:
            array: matrix of overlap
        """
        overlapInfo = np.matmul(basis.T, self.mulDiag(cmatInv, basis))
        return overlapInfo

    def getOverlap21(self, basis, cmatInv):
        """Computes the overlap between 21cm modes.

        Args:
            basis (array): 21cm modes
            cmatInv (array): noise covariance matrix

        Returns:
            array: matrix of overlap
        """
        exp21 = np.identity(len(self.nu))
        exp21 = np.tile(exp21, (self.nLST*len(self.ant), 1))
        overlapInfo = np.matmul(basis.T, np.matmul(exp21.T, np.matmul(cmatInv, np.matmul(exp21, basis))))
        return overlapInfo
    
    def getOverlapFg21(self, bFg, b21, cmatInv):
        """Computes the overlap between foreground and 21cm modes.

        Args:
            bFg (array): foreground modes
            b21 (array): 21cm modes
            cmatInv (array): noise covariance matrix

        Returns:
            array: matrix of overlap
        """
        exp21 = np.identity(len(self.nu))
        exp21 = np.tile(exp21, (self.nLST*len(self.ant), 1))
        overlapInfo = np.matmul(bFg.T, np.matmul(cmatInv, np.matmul(exp21, b21)))
        return overlapInfo
    
    @staticmethod
    def mulDiag(A, B):
        return np.multiply(np.diag(A)[:, None], B)

    @staticmethod
    def invDiag(A):
        B = np.zeros(A.shape)
        np.fill_diagonal(B, 1./np.diag(A))
        return B
