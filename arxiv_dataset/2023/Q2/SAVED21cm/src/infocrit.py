import numpy as np

class InfoCrit:
    def __init__(self, nu, nLST, ant):
        """Initialize the Deviance Information Criterion (DIC).

        Args:
            nu (array): Frequency range
            nLST (int): Number of time bins to fit
            ant (int): List of antenna designs
        """
        self.nu = nu
        self.nLST = nLST
        self.ants = len(ant)
    
    def gridinfo(self, quantity, modesFg, modes21, wgtBasisFg, wgtBasis21, covmatInv, mockObs, file):
        """This method calculates the IC over a grid of foreground and 21cm modes,
        and saves the data in a text file.

        Args:
            quantity (string): Quantity to minimize\
                               'DIC' for Deviance Information Criterion,\
                               'BIC' for Bayesian Information Criterion
            modesFg (int): Number of foreground modes for the grid
            modes21 (int): Number of 21cm modes for the grid
            wgtBasisFg (array): Foreground basis
            wgtBasis21 (array): 21cm basis
            covmatInv (array): Inverse of noise covariance matrix
            mockObs (array): Mock observation
            file (string): Filename to save the information
        """
        print('Info Criterion: Calculating info criterion over grid...', end='', flush=True)
        gridFg = np.linspace(1, modesFg, num=modesFg, dtype=int).tolist()
        grid21 = np.linspace(1, modes21, num=modes21, dtype=int).tolist()
        
        exp21 = np.identity(len(self.nu))
        exp21 = np.tile(exp21, (self.nLST*self.ants, 1))     
           
        fp = open(file, 'w')
        
        for i in range(len(gridFg)):
            for j in range(len(grid21)):
                F_Fg = np.zeros(shape=(len(self.nu)*self.nLST*self.ants, gridFg[i]))
                F_21 = np.zeros(shape=(len(self.nu), grid21[j]))
                
                for mm in range(gridFg[i]):
                    F_Fg.T[mm] = wgtBasisFg[:, mm]
                for nn in range(grid21[j]):
                    F_21.T[nn] = wgtBasis21[:, nn]
                    
                F_x21 = np.matmul(exp21, F_21)
                F = np.concatenate((F_Fg, F_x21), axis=-1)
                
                S = np.matmul(F.T, self.mulDiag(covmatInv, F))
                S = np.linalg.inv(S)
                
                Xi = np.matmul(S, np.matmul(F.T, self.mulDiag(covmatInv, mockObs)))
                reconsObs = np.matmul(F, Xi)
                diff = reconsObs - mockObs
                
                if quantity == 'DIC':
                    ic = np.matmul(diff.T, self.mulDiag(covmatInv, diff)) +\
                        2 * (gridFg[i] + grid21[j])
                if quantity == 'BIC':
                    ic = np.matmul(diff.T, self.mulDiag(covmatInv, diff)) +\
                        (gridFg[i] + grid21[j]) * np.log(len(self.nu))
                
                fp.write(str(gridFg[i]))
                fp.write("\t")
                fp.write(str(grid21[j]))
                fp.write("\t")
                fp.write(str(ic[0][0]))
                fp.write("\n")
        fp.close()
        print('Done!')
    
    def searchMinima(self, file):
        """This method searches for the minima of the information criterion
        from the gridded data.

        Args:
            file (string): Filename that contains the gridded IC data.

        Returns:
            int: Number of foreground modes that minimize the IC,
                 Number of 21cm modes that minimize the IC,
                 Minimized IC value
        """
        print('Info Critetion: Searching for minima over the grid...', end='', flush=True)
        data = np.loadtxt(file)
        Vals = list(data[:, 2])
        minIndex = Vals.index(min(Vals))
        modesFg = int(data[minIndex, 0])
        modes21 = int(data[minIndex, 1])
        minima = data[minIndex, 2]
        print('Done!')
        print('Info Criterion: IC is minimzed for %d FG and %d signal modes.'%(modesFg, modes21))
        return modesFg, modes21, minima
                        
    @staticmethod
    def mulDiag(A, B):
        return np.multiply(np.diag(A)[:, None], B)
    