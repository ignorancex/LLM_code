import numpy as np
import h5py
import datetime

class Modset:
    """This class contains methods for reading in the modelling sets 
        (foreground and 21cm signal) from our standard hdf5 files.
    """
    def __init__(self, nu, nLST, ant):
        """Initialize the modelling sets (foreground and 21cm signal).

        Args:
            nu (array): Frequency range
            nLST (int): Number of time bins to fit
            ant (list): List of antenna designs
        """
        self.nu = nu
        self.nLST = nLST
        self.ant = ant

    def get21modset(self, file, nuMin, nuMax):
        """This method returns the 21cm signal modelling set.

        Args:
            file (string): Path to file
            nuMin (float): Minimum frequency accessible
            nuMax (float): Maximum frequency accessible

        Returns:
            array: 21cm modelling set of shape (n_curves, n_nu)
        """
        print('Modelling set: Reading 21 modelling set...', end='', flush=True)
        hf = h5py.File(file, 'r')
        nu = hf.attrs['frequencies']
        ind = np.where(np.logical_and(nu >= nuMin, nu <= nuMax))
        nu = nu[ind]
        nu = nu[0::5]
        mod21 = hf.get('signal_sample')
        mod21 = np.array(mod21)
        # Restricting the 21cm TS between nu_min and nu_max
        mod21 = ((mod21.T)[ind]).T
        mod21 = mod21.T[0::5]/1000. #! Removed .T used [0::5].T
        print('Done!')
        return mod21

    def getFgmodset(self, file, nLST_tot):
        """This method returns the foreground modelling set of a specific antenna.

        Args:
            file (string): Path to file
            nLST_tot (int): Total number of time bins in the set

        Returns:
            array: Foreground modelling set of shape (n_curves, n_nu*n_t)
        """
        with h5py.File(file, 'r') as hdf:
            modFgRaw = hdf.get('training-set')
            modFgRaw = np.array(modFgRaw)
            n_samp = modFgRaw.shape[0]
            modFgRaw = modFgRaw[:, 0*len(self.nu):nLST_tot*len(self.nu)]
            modFgRaw = modFgRaw.flatten()
            modFgRaw = modFgRaw.reshape(n_samp*self.nLST*int(nLST_tot/self.nLST), len(self.nu))
            modFg = np.zeros(shape=(n_samp*self.nLST, len(self.nu)))
            j = 0
            for i in range(n_samp*self.nLST):
                modFg[i, :] = np.sum(modFgRaw[j:j+int(nLST_tot/self.nLST), :],
                                     axis=0)/int(nLST_tot/self.nLST)
                j = j + int(nLST_tot/self.nLST)
            modFg = modFg.reshape(n_samp, self.nLST*len(self.nu))
        return modFg
    
    def getcFgmodset(self, file, nLST_tot):
        """This method concatenates the foreground modelling set for multiple antennas.

        Args:
            file (string): Path to file
            nLST_tot (int): Total number of time bins in the set

        Returns:
            array: Concatenated foreground modelling set of shape
                   (n_curves, n_nu*n_t*n_ant)
        """
        print('Modelling set: Reading FG modelling set...', end='', flush=True)
        mFg = np.concatenate(([Modset.getFgmodset(self, file='%s'%file+'%s'%self.ant[i]+'.h5',
                                                  nLST_tot=nLST_tot)
                               for i in range(len(self.ant))]), axis=-1)
        mFg = mFg.T
        print('Done!')
        return mFg
    
    def getAvgGivenTime(self, path, date, numReg, fgModel, dateTimeVal, intBins, antenna, intTime=30):
        ''' Calculates average over time: given start time and how many bins to integrate '''
        allInfo = []
        for _ in range(intBins):
            startTime = dateTimeVal.time()
            allInfo.append(np.load("%s/FgTs_%s_%s_30_%s_reg-%d_%s.npy"
                                %(path, date, str(startTime), fgModel, numReg, antenna)))
            dateTimeVal = dateTimeVal + datetime.timedelta(minutes=intTime)
        averageInfo = np.average(allInfo, axis=0)
        return averageInfo

    def getcFgModsetGivenTime(self, path, date, numReg, fgModel, dateTimeList, intBins, antenna):
        ''' Concatenates the foreground ts given the initial time of dateTimeList '''
        print('Modelling set: Reading FG modelling set...', end='', flush=True)
        allInfo = []
        for i in range(len(dateTimeList)):
            allInfo.append(self.getAvgGivenTime(path=path, date=date, numReg=numReg, fgModel=fgModel,
                                                dateTimeVal=dateTimeList[i], intBins=intBins, antenna=antenna))
        concatenatedInfo = np.concatenate(allInfo, axis=1)
        concatenatedInfo = concatenatedInfo.T
        print('Done!')
        return concatenatedInfo
    
    def getcFgModsetGivenTimeAnt(self, path, date, numReg, fgModel, dateTimeList, intBins, antenna):
        ''' Concatenates the foreground ts given the initial time of dateTimeList '''
        print('Modelling set: Reading FG modelling set...', end='', flush=True)
        allInfoAnt = []
        allInfo = []
        for j in range(len(antenna)):
            for i in range(len(dateTimeList)):
                allInfo.append(self.getAvgGivenTime(path=path, date=date, numReg=numReg, fgModel=fgModel,
                                                    dateTimeVal=dateTimeList[i], intBins=intBins, antenna=antenna[j]))
            concatenatedInfo = np.concatenate(allInfo, axis=1)
            del allInfo[:]
            allInfoAnt.append(concatenatedInfo)
        concatenatedInfoAnt = np.concatenate(allInfoAnt, axis=1)
        concatenatedInfoAnt = concatenatedInfoAnt.T
        print('Done!')
        return concatenatedInfoAnt
    
    def concatenateModels(self, *models):
        return np.concatenate(models)
    

class Inputs:
    """This class contains methods for getting the inputs (foreground and 21cm)
        from their training set.
    """
    def __init__(self, nu, nLST, ant):
        """Initialize the inputs (foreground and 21cm).

        Args:
            nu (array): Frequency range
            nLST (int): Number of time bins to git
            ant (list): List of antenna designs
        """
        self.nu = nu
        self.nLST = nLST
        self.ant = ant
        self.ants = len(ant)
    
    def expand(self, y21):
        exp21 = np.identity(len(self.nu))
        exp21 = np.tile(exp21, (self.nLST*self.ants, 1))
        y_x21 = np.matmul(exp21, y21)
        return y_x21
    
    def getExp21(self, modset, ind=0):
        """This method returns one 21cm signal profile from the modelling set.

        Args:
            modset (array): 21cm modelling set
            ind (int): Index to randomly pick one sample

        Returns:
            array: Input 21cm signal profile, and the expanded version
        """
        y21 = modset[:, ind].reshape(len(self.nu), 1)
        # exp21 = np.identity(len(self.nu))
        # exp21 = np.tile(exp21, (self.nLST*self.ants, 1))
        # y_x21 = np.matmul(exp21, y21)
        y_x21 = self.expand(y21)
        return y21, y_x21
    
    def getExpGauss21(self, A=0.25, nu0=85, sigma=15):
        y21 =  -A*np.exp(-pow((self.nu-nu0), 2)/(2*sigma*sigma))
        y21 = y21.reshape(len(self.nu), 1)
        y_x21 = self.expand(y21)
        return y21, y_x21
    
    def getExpEdges21(self, A=0.52, nu0=78.3, w=20.7, tau=6.5):
        B_fac = 4*pow(((self.nu-nu0)/w), 2) * np.log(-np.log((1+np.exp(-tau))/2)/tau)
        y21 = -A*((1-np.exp(-tau*np.exp(B_fac)))/(1-np.exp(-tau)))
        y21 = y21.reshape(len(self.nu), 1)
        y_x21 = self.expand(y21)
        return y21, y_x21
    
    def getFg(self, modset, ind=0):
        """This method returns one foreground spectra from the modelling set.

        Args:
            modset (array): Foreground modelling set
            ind (int): Index to randomly pick one sample

        Returns:
            array: Input foregrounds
        """
        yFg = modset[:, ind].reshape(len(self.nu)*self.nLST*self.ants, 1)
        return yFg
    
    def simFg(self, file, nLST_tot):
        """This method returns the simulated foreground (different from training set).

        Args:
            file (string): Path to simulated foreground file
            nLST_tot (int): Total number of time bins

        Returns:
            array: Input foreground
        """
        with h5py.File(file, 'r') as hdf:
            modFgRaw = hdf.get('simFg')
            modFgRaw = np.array(modFgRaw)
            n_samp = modFgRaw.shape[0]
            modFgRaw = modFgRaw[:, 0*len(self.nu):nLST_tot*len(self.nu)]
            modFgRaw = modFgRaw.flatten()
            modFgRaw = modFgRaw.reshape(n_samp*self.nLST*int(nLST_tot/self.nLST), len(self.nu))
            modFg = np.zeros(shape=(n_samp*self.nLST, len(self.nu)))
            j = 0
            for i in range(n_samp*self.nLST):
                modFg[i, :] = np.sum(modFgRaw[j:j+int(nLST_tot/self.nLST), :],
                                        axis=0)/int(nLST_tot/self.nLST)
                j = j + int(nLST_tot/self.nLST)
            modFg = modFg.reshape(n_samp, self.nLST*len(self.nu))
        return modFg
        
    def getSimFg(self, file, nLST_tot):
        """This method concatenates the simulated foreground for multiple antennas.

        Args:
            file (string): Path to file
            nLST_tot (int): Total number of time bins

        Returns:
            array: Simulated foreground
        """        
        yFg = np.concatenate(([self.simFg(file='%s'%file+'%s'%self.ant[i]+'.h5', nLST_tot=nLST_tot)
                               for i in range(len(self.ant))]), axis=-1)
        return yFg.T
    
    def getAvgGivenTime(self, path, date, fgModel, dateTimeVal, intBins, antenna, intTime=30):
        allInfo = []
        for _ in range(intBins):
            startTime = dateTimeVal.time()
            print("Integrating time bin:", startTime)
            allInfo.append(np.load("%s/mockObs_%s_%s_30_%s_%s.npy"
                                   %(path, date, str(startTime), fgModel, antenna))[1])
            dateTimeVal = dateTimeVal + datetime.timedelta(minutes=intTime)
        
        averageInfo = np.average(allInfo, axis=0)
        return averageInfo

    def getSimFgGivenTime(self, path, date, fgModel, dateTimeList, intBins, antenna):
        allInfo = []
        for i in range(len(dateTimeList)):
            allInfo.append(self.getAvgGivenTime(path=path, date=date, fgModel=fgModel,
                                                dateTimeVal=dateTimeList[i], intBins=intBins, antenna=antenna))
        concatenatedInfo = np.concatenate(allInfo, axis=0)
        concatenatedInfo = concatenatedInfo.reshape(concatenatedInfo.shape[0], 1) - 2.725
        return concatenatedInfo
    
    def concatenateInputs(self, *inputs):
        return np.concatenate((inputs), axis=0)
