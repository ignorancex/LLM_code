import os; import sys
from src.runpipe import Pipeline, Pipeline_P2
from tqdm.notebook import tqdm

class HiddenPrints:
    """To not print statements while calculating statistics over an ensemble.
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class StatQuants:
    def __init__(self, nu, nLST, ant, path21TS, pathFgTS, obsDate, obsDateTime, intBins, numReg, fgModel,
                 dT, modesFg, modes21, quantity, file):
        """Initialize for generating some statistical measures of the fitting.

        Args:
            nu (array): Frequency range
            nLST (int): Number of time bins to fit
            ant (lsit): List of antenna designs
            path21TS (string): Path to 21cm modelling set
            pathFgTS (string): Path to foregrounds modelling set
            dT (int): Integration time in hours
            modesFg (int): Total number of FG modes
            modes21 (int): Total number of 21 modes
            quantity (string): Quantity to minimize\
                               'DIC' for Deviance Information Criterion,\
                               'BIC' for Bayesian Information Criterion
            file (string): Filename to store the gridded IC
        """
        self.nu = nu
        self.nLST = nLST
        self.ant = ant
        self.path21TS = path21TS
        self.pathFgTS = pathFgTS
        self.obsDate = obsDate
        self.obsDateTime = obsDateTime
        self.intBins=intBins
        self.numReg = numReg
        self.fgModel = fgModel
        self.dT = dT
        self.modesFg = modesFg
        self.modes21 = modes21
        self.quantity = quantity
        self.file = file
        
    def getStats(self, fname, iList21, iListFg):
        """Estimates the statistical measures over an ensemble of different foregrounds and 21cm
        signals and store that information in a file.

        Args:
            fname (string): Filename to store the statistical information
            iList21 (array): List of 21cm index to use as the input (from the modelling set)
            iListFg (array): List of foreground index to use as the input (from the modelling set)
        """
        print('--------------- Estimating statistical measures ---------------\n', flush=True)
        if not os.path.exists('StatsOutput/'):
            os.mkdir('StatsOutput/')
        with HiddenPrints():
            fp = open('StatsOutput/%s'%fname, 'w')
            ''' Write the settings for the analysis '''
            fp.write('# Frequency (min, max, step): (%s, %s, %s)\n'
                     %(self.nu[0], self.nu[-1], self.nu[1] - self.nu[0])); 
            fp.write('# Number of LST bins: %s\n'%self.nLST); 
            fp.write('# Antenna Design: %s\n'%self.ant); 
            fp.write('# 21 training set: %s\n'%self.path21TS); 
            fp.write('# Fg training set: %s\n'%self.pathFgTS); 
            fp.write('# Integration Time: %s\n'%self.dT); 
            fp.write('# Total Fg modes: %s\n'%self.modesFg); 
            fp.write('# Total 21 modes: %s\n'%self.modes21); 
            fp.write('\n# mFg\tm21\ticVal\tbias\tnormD\trms\n')
            
            for i21 in tqdm(iList21, desc='i21', colour='#36FAC5'):
                for iFg in tqdm(iListFg, desc='iFg', leave=False, colour='#FCB074'):
                    ind21 = int(i21); indFg = int(iFg)
                    pipe = Pipeline(nu=self.nu, nLST=self.nLST, ant=self.ant, path21TS=self.path21TS,
                                    pathFgTS=self.pathFgTS, dT=self.dT, obsDate=self.obsDate, obsDateTime=self.obsDateTime,
                                    intBins=self.intBins, numReg=self.numReg, fgModel=self.fgModel,
                                    modesFg=self.modesFg, modes21=self.modes21, quantity=self.quantity, file=self.file,
                                    indexFg=indFg, index21=ind21)
                    quants = pipe.runPipeline()
                    for i in range(len(quants)):
                        fp.write(str(quants[i]))
                        fp.write('\t')
                    fp.write('\n')
            fp.close()    


class StatQuants_P2:
    def __init__(self, nu, lst2fit, nLST, ant, ant1, ant2, path21TS, pathFgTS, obsDate,
                 obsDateTime1, obsDateTime2, intBins, numReg, fgModel, dT,
                 modesFg, modes21, quantity, file):
        """Initialize for generating some statistical measures of the fitting.

        Args:
            nu (array): Frequency range
            nLST (int): Number of time bins to fit
            ant (lsit): List of antenna designs
            path21TS (string): Path to 21cm modelling set
            pathFgTS (string): Path to foregrounds modelling set
            dT (int): Integration time in hours
            modesFg (int): Total number of FG modes
            modes21 (int): Total number of 21 modes
            quantity (string): Quantity to minimize\
                               'DIC' for Deviance Information Criterion,\
                               'BIC' for Bayesian Information Criterion
            file (string): Filename to store the gridded IC
        """
        self.nu = nu
        self.lst2fit = lst2fit
        self.nLST = nLST
        self.ant = ant
        self.ant1 = ant1
        self.ant2 = ant2
        self.path21TS = path21TS
        self.pathFgTS = pathFgTS
        self.obsDate = obsDate
        self.obsDateTime1 = obsDateTime1
        self.obsDateTime2 = obsDateTime2
        self.intBins=intBins
        self.numReg = numReg
        self.fgModel = fgModel
        self.dT = dT
        self.modesFg = modesFg
        self.modes21 = modes21
        self.quantity = quantity
        self.file = file
        
    def getStats(self, fname, iList21, iListFg):
        """Estimates the statistical measures over an ensemble of different foregrounds and 21cm
        signals and store that information in a file.

        Args:
            fname (string): Filename to store the statistical information
            iList21 (array): List of 21cm index to use as the input (from the modelling set)
            iListFg (array): List of foreground index to use as the input (from the modelling set)
        """
        print('--------------- Estimating statistical measures ---------------\n', flush=True)
        if not os.path.exists('StatsOutput_P2/'):
            os.mkdir('StatsOutput_P2/')
        with HiddenPrints():
            fp = open('StatsOutput_P2/%s'%fname, 'w')
            ''' Write the settings for the analysis '''
            fp.write('# Frequency (min, max, step): (%s, %s, %s)\n'
                     %(self.nu[0], self.nu[-1], self.nu[1] - self.nu[0])); 
            fp.write('# Number of LST bins: %s\n'%self.nLST); 
            fp.write('# Antenna Design: %s\n'%self.ant); 
            fp.write('# 21 training set: %s\n'%self.path21TS); 
            fp.write('# Fg training set: %s\n'%self.pathFgTS); 
            fp.write('# Integration Time: %s\n'%self.dT); 
            fp.write('# Total Fg modes: %s\n'%self.modesFg); 
            fp.write('# Total 21 modes: %s\n'%self.modes21); 
            fp.write('\n# mFg\tm21\ticVal\tbias\tnormD\trms\n')
            
            for i21 in tqdm(iList21, desc='i21', colour='#36FAC5'):
                for iFg in tqdm(iListFg, desc='iFg', leave=False, colour='#FCB074'):
                    ind21 = int(i21); indFg = int(iFg)
                    pipe = Pipeline_P2(nu=self.nu, lst2fit=self.lst2fit, nLST=self.nLST, ant=self.ant,
                                       ant1=self.ant1, ant2=self.ant2, path21TS=self.path21TS,
                                       pathFgTS=self.pathFgTS, dT=self.dT, obsDate=self.obsDate,
                                       obsDateTime1=self.obsDateTime1, obsDateTime2=self.obsDateTime2,
                                       intBins=self.intBins, numReg=self.numReg, fgModel=self.fgModel,
                                       modesFg=self.modesFg, modes21=self.modes21, quantity=self.quantity,
                                       file=self.file, indexFg=indFg, index21=ind21)
                    quants = pipe.runPipeline()
                    for i in range(len(quants)):
                        fp.write(str(quants[i]))
                        fp.write('\t')
                    fp.write('\n')
            fp.close()
