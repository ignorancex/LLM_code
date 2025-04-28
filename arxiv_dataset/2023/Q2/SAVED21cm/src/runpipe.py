import os
from src.readset import Modset, Inputs
from src.basis import Basis
from src.noise import Noise
from src.infocrit import InfoCrit
from src.extractor import Extractor
from src.visuals import Visual

class Pipeline:   
    def __init__(self, nu, nLST, ant, path21TS, pathFgTS,
                 obsDate, obsDateTime, intBins, numReg, fgModel, 
                 dT=6, modesFg=50, modes21=80, quantity='DIC', 
                 file='test.txt', indexFg=0, index21=0, visual=False, save=False):
        """Initialize to run the pipeline with the given settings.

        Args:
            nu (array): Frequency range
            nLST (int): Number of time bins to fit
            ant (list): List of antenna designs
            path21TS (string): Path to 21cm modelling set
            pathFgTS (string): Path to foregrounds modelling set
            dT (int, optional): Integration time in hours. Defaults to 6.
            modesFg (int, optional): Total number of FG modes. Defaults to 50.
            modes21 (int, optional): Total number of 21 modes. Defaults to 80.
            quantity (string): Quantity to minimize\
                               'DIC' for Deviance Information Criterion,\
                               'BIC' for Bayesian Information Criterion
            file (str, optional): Filename to store the gridded IC. Defaults to 'test.txt'.
            indexFg (int, optional): Index to get input from the FG modelling set. Defaults to 0.
            index21 (int, optional): Index to get input from the 21 modelling set. Defaults to 0.
            visual (bool, optional): Option to plot the extracted signal. Defaults to False.
            save (bool, optional): Option to save the figures. Defaults to False.
        """
        self.nu = nu
        self.nLST = nLST
        self.ant = ant
        self.path21TS = path21TS
        self.pathFgTS = pathFgTS
        self.obsDate = obsDate
        self.obsDateTime = obsDateTime
        self.intBins = intBins
        self.numReg = numReg
        self.fgModel = fgModel
        self.dT = dT
        self.modesFg = modesFg
        self.modes21 = modes21
        self.quantity = quantity
        self.file = file
        self.indFg = indexFg
        self.ind21 = index21
        self.visual = visual
        self.save = save
    
    def runPipeline(self):
        """To run the pipeline.
        """
        print('-------------------- Running the pipeline ---------------------\n')
        ''' Reading in the modelling sets '''
        models = Modset(nu=self.nu, nLST=self.nLST, ant=self.ant)
        m21 = models.get21modset(file=self.path21TS, nuMin=50, nuMax=200)
        mFg = models.getcFgModsetGivenTimeAnt(path=self.pathFgTS, date=self.obsDate,
                                              numReg=self.numReg, fgModel=self.fgModel,
                                              dateTimeList=self.obsDateTime,
                                              intBins=self.intBins, antenna=self.ant)
        
        ''' Generating inputs from the modelling sets '''
        inputs = Inputs(nu=self.nu, nLST=self.nLST, ant=self.ant)
        y21, y_x21 = inputs.getExp21(modset=m21, ind=self.ind21)
        yFg = inputs.getFg(modset=mFg, ind=self.indFg)
        
        ''' Generating the noise and getting its covariance '''
        noise = Noise(nu=self.nu, nLST=self.nLST, ant=self.ant, power=y_x21+yFg,
                      deltaNu=self.nu[1] - self.nu[0], deltaT=self.dT)
        thermRealz = noise.noiseRealz()        
        cmat = noise.covmat()
        cmatInv = noise.covmatInv()
        
        ''' Getting the noise covariance weighted modelling sets '''
        wgt_m21 = noise.wgtTs(modset=m21, opt='21')
        wgt_mFg = noise.wgtTs(modset=mFg, opt='FG')
        
        ''' Generating the mock observation '''
        y = y_x21 + yFg + thermRealz
        
        ''' Weighted SVD for getting the optimal modes '''
        basis = Basis(nu=self.nu, nLST=self.nLST, ant=self.ant)
        b21 = basis.wgtSVDbasis(modset=wgt_m21, covmat=cmat, opt='21')
        bFg = basis.wgtSVDbasis(modset=wgt_mFg, covmat=cmat, opt='FG')
        
        ''' Minimizing information criterion for selecting the number of modes '''
        ic = InfoCrit(nu=self.nu, nLST=self.nLST, ant=self.ant)
        ic.gridinfo(modesFg=self.modesFg, modes21=self.modes21, wgtBasis21=b21, wgtBasisFg=bFg,
                    quantity=self.quantity, covmatInv=cmatInv, mockObs=y, file=self.file)
        icmodesFg, icmodes21, _ = ic.searchMinima(file=self.file)
        
        ''' Finally extracting the signal! '''
        ext = Extractor(nu=self.nu, nLST=self.nLST, ant=self.ant)
        extInfo = ext.extract(modesFg=icmodesFg, modes21=icmodes21,
                              wgtBasisFg=bFg, wgtBasis21=b21,
                              covmatInv=cmatInv, mockObs=y, y21=y21)

        ''' Visuals '''
        if self.visual:
            vis = Visual(nu=self.nu, nLST=self.nLST, ant=self.ant, save=self.save)
            vis.plotModset(set=m21, opt='21', n_curves=1000)
            vis.plotModset(set=mFg, opt='FG', n_curves=100)
            vis.plotMockObs(y21=y21, yFg=yFg, noise=thermRealz)
            vis.plotBasis(basis=b21, opt='21')
            vis.plotBasis(basis=bFg, opt='FG')
            vis.plotInfoGrid(file=self.file, modesFg=self.modesFg, modes21=self.modes21,
                            quantity=self.quantity, minModesFg=icmodesFg, minModes21=icmodes21)
            vis.plotExtSignal(y21=y21, recons21=extInfo[1], sigma21=extInfo[3])
        
        os.system('rm %s'%self.file)
        
        ''' Statistical Measures '''
        qDic = extInfo[8]
        qBias = extInfo[10]
        qNormD = extInfo[11]
        qRms = extInfo[7] * qBias[0]
        
        return icmodesFg, icmodes21, qDic[0][0], qBias[0], qNormD, qRms


class Pipeline_P2:   
    def __init__(self, nu, lst2fit, nLST, ant, ant1, ant2, path21TS, pathFgTS,
                 obsDate, obsDateTime1, obsDateTime2, intBins, numReg, fgModel, 
                 dT=6, modesFg=50, modes21=80, quantity='DIC', 
                 file='test.txt', indexFg=0, index21=0, visual=False, save=False):
        """Initialize to run the pipeline with the given settings.

        Args:
            nu (array): Frequency range
            nLST (int): Number of time bins to fit
            ant (list): List of antenna designs
            path21TS (string): Path to 21cm modelling set
            pathFgTS (string): Path to foregrounds modelling set
            dT (int, optional): Integration time in hours. Defaults to 6.
            modesFg (int, optional): Total number of FG modes. Defaults to 50.
            modes21 (int, optional): Total number of 21 modes. Defaults to 80.
            quantity (string): Quantity to minimize\
                               'DIC' for Deviance Information Criterion,\
                               'BIC' for Bayesian Information Criterion
            file (str, optional): Filename to store the gridded IC. Defaults to 'test.txt'.
            indexFg (int, optional): Index to get input from the FG modelling set. Defaults to 0.
            index21 (int, optional): Index to get input from the 21 modelling set. Defaults to 0.
            visual (bool, optional): Option to plot the extracted signal. Defaults to False.
            save (bool, optional): Option to save the figures. Defaults to False.
        """
        self.nu = nu
        self.lst_2_fit = lst2fit
        self.nLST = nLST
        self.ant1 = ant1
        self.ant2 = ant2
        self.ant = ant
        self.path21TS = path21TS
        self.pathFgTS = pathFgTS
        self.obsDate = obsDate
        self.obsDateTime1 = obsDateTime1
        self.obsDateTime2 = obsDateTime2
        self.intBins = intBins
        self.numReg = numReg
        self.fgModel = fgModel
        self.dT = dT
        self.modesFg = modesFg
        self.modes21 = modes21
        self.quantity = quantity
        self.file = file
        self.indFg = indexFg
        self.ind21 = index21
        self.visual = visual
        self.save = save
    
    def runPipeline(self):
        """To run the pipeline.
        """
        print('-------------------- Running the pipeline ---------------------\n')
        ''' Reading in the modelling sets '''
        models = Modset(nu=self.nu, nLST=self.nLST, ant=self.ant)
        m21 = models.get21modset(file=self.path21TS, nuMin=50, nuMax=200)
        mFg1 = models.getcFgModsetGivenTimeAnt(path=self.pathFgTS, date=self.obsDate,
                                              numReg=self.numReg, fgModel=self.fgModel,
                                              dateTimeList=self.obsDateTime1,
                                              intBins=self.intBins, antenna=self.ant1)
        mFg2 = models.getcFgModsetGivenTimeAnt(path=self.pathFgTS, date=self.obsDate,
                                              numReg=self.numReg, fgModel=self.fgModel,
                                              dateTimeList=self.obsDateTime2,
                                              intBins=self.intBins, antenna=self.ant2)
        mFg = models.concatenateModels(mFg1, mFg2)
        
        ''' Generating inputs from the modelling sets '''
        inputs21 = Inputs(nu=self.nu, nLST=self.nLST, ant=self.ant)
        y21, y_x21 = inputs21.getExp21(modset=m21, ind=self.ind21)

        inputs = Inputs(nu=self.nu, nLST=self.lst_2_fit, ant=self.ant)
        yFg1 = inputs.getFg(modset=mFg1, ind=self.indFg)
        yFg2 = inputs.getFg(modset=mFg2, ind=self.indFg)
        yFg = inputs.concatenateInputs(yFg1, yFg2)
        
        ''' Generating the noise and getting its covariance '''
        noise = Noise(nu=self.nu, nLST=self.nLST, ant=self.ant, power=y_x21+yFg,
                      deltaNu=self.nu[1] - self.nu[0], deltaT=self.dT)
        thermRealz = noise.noiseRealz()        
        cmat = noise.covmat()
        cmatInv = noise.covmatInv()
        
        ''' Getting the noise covariance weighted modelling sets '''
        wgt_m21 = noise.wgtTs(modset=m21, opt='21')
        wgt_mFg = noise.wgtTs(modset=mFg, opt='FG')
        
        ''' Generating the mock observation '''
        y = y_x21 + yFg + thermRealz
        
        ''' Weighted SVD for getting the optimal modes '''
        basis = Basis(nu=self.nu, nLST=self.nLST, ant=self.ant)
        b21 = basis.wgtSVDbasis(modset=wgt_m21, covmat=cmat, opt='21')
        bFg = basis.wgtSVDbasis(modset=wgt_mFg, covmat=cmat, opt='FG')
        
        ''' Minimizing information criterion for selecting the number of modes '''
        ic = InfoCrit(nu=self.nu, nLST=self.nLST, ant=self.ant)
        ic.gridinfo(modesFg=self.modesFg, modes21=self.modes21, wgtBasis21=b21, wgtBasisFg=bFg,
                    quantity=self.quantity, covmatInv=cmatInv, mockObs=y, file=self.file)
        icmodesFg, icmodes21, _ = ic.searchMinima(file=self.file)
        
        ''' Finally extracting the signal! '''
        ext = Extractor(nu=self.nu, nLST=self.nLST, ant=self.ant)
        extInfo = ext.extract(modesFg=icmodesFg, modes21=icmodes21,
                              wgtBasisFg=bFg, wgtBasis21=b21,
                              covmatInv=cmatInv, mockObs=y, y21=y21)

        ''' Visuals '''
        if self.visual:
            vis = Visual(nu=self.nu, nLST=self.nLST, ant=self.ant, save=self.save)
            vis.plotModset(set=m21, opt='21', n_curves=1000)
            vis.plotModset(set=mFg, opt='FG', n_curves=100)
            vis.plotMockObs(y21=y21, yFg=yFg, noise=thermRealz)
            vis.plotBasis(basis=b21, opt='21')
            vis.plotBasis(basis=bFg, opt='FG')
            vis.plotInfoGrid(file=self.file, modesFg=self.modesFg, modes21=self.modes21,
                            quantity=self.quantity, minModesFg=icmodesFg, minModes21=icmodes21)
            vis.plotExtSignal(y21=y21, recons21=extInfo[1], sigma21=extInfo[3])
        
        os.system('rm %s'%self.file)
        
        ''' Statistical Measures '''
        qDic = extInfo[8]
        qBias = extInfo[10]
        qNormD = extInfo[11]
        qRms = extInfo[7] * qBias[0]
        
        return icmodesFg, icmodes21, qDic[0][0], qBias[0], qNormD, qRms
    