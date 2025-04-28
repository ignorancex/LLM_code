''' A python script that illustrates how the formalism works '''

import sys; sys.path.insert(1, './../')
from src.readset import Modset, Inputs
from src.noise import Noise
from src.basis import Basis
from src.visuals import Visual
from src.infocrit import InfoCrit
from src.extractor import Extractor
import settings as set

''' Print the settings for the pipeline '''
set.LST = 2
set.checkSettings()

''' Reading in the modelling sets '''
models = Modset(nu=set.NU, nLST=set.LST, ant=set.ANT)
m21 = models.get21modset(file=set.PATH21TS, nuMin=50, nuMax=200)
mFg = models.getcFgmodset(file=set.PATHFGTS, nLST_tot=144)

''' Generating inputs from the modelling sets '''
inputs = Inputs(nu=set.NU, nLST=set.LST, ant=set.ANT)
y21, y_x21 = inputs.getExp21(modset=m21, ind=467)
yFg = inputs.getFg(modset=mFg, ind=0)

''' Generating the noise and getting its covariance '''
noise = Noise(nu=set.NU, nLST=set.LST, ant=set.ANT, power=y_x21 + yFg,
              deltaNu=set.dNU, deltaT=set.dT)
realz = noise.noiseRealz()
cmat = noise.covmat()
cmatInv = noise.covmatInv()

''' Getting the noise covariance weighted modelling sets '''
wgt_m21 = noise.wgtTs(modset=m21, opt='21')
wgt_mFg = noise.wgtTs(modset=mFg, opt='FG')

''' Generating the mock observation '''
y = y_x21 + yFg + realz

''' Weighted SVD for getting the optimal modes '''
basis = Basis(nu=set.NU, nLST=set.LST, ant=set.ANT)
b21 = basis.wgtSVDbasis(modset=wgt_m21, covmat=cmat, opt='21')
bFg = basis.wgtSVDbasis(modset=wgt_mFg, covmat=cmat, opt='FG')

''' Minimizing information criterion for selecting the number of modes '''
d = InfoCrit(nu=set.NU, nLST=set.LST, ant=set.ANT)
d.gridinfo(quantity=set.QUANTITY, modesFg=set.MODES_FG, modes21=set.MODES_21,
           wgtBasisFg=bFg, wgtBasis21=b21,
           covmatInv=cmatInv, mockObs=y, file=set.FNAME)
modesFg, modes21, dic = d.searchMinima(file=set.FNAME)

''' Finally extracting the signal! '''
ext = Extractor(nu=set.NU, nLST=set.LST, ant=set.ANT)
extInfo = ext.extract(modesFg=modesFg, modes21=modes21,
                      wgtBasisFg=bFg, wgtBasis21=b21,
                      covmatInv=cmatInv, mockObs=y, y21=y21)

''' Visualizing everything! '''
if set.VISUALS:
    vis = Visual(nu=set.NU, nLST=set.LST, ant=set.ANT, save=set.SAVE)
    vis.plotModset(set=m21, opt='21', n_curves=1000)
    vis.plotModset(set=mFg, opt='FG', n_curves=100)
    vis.plotMockObs(y21=y21, yFg=yFg, noise=realz)
    vis.plotBasis(basis=b21, opt='21')
    vis.plotBasis(basis=bFg, opt='FG')
    vis.plotInfoGrid(file=set.FNAME, modesFg=set.MODES_FG, modes21=set.MODES_21,
                     quantity=set.QUANTITY, minModesFg=modesFg, minModes21=modes21)
    vis.plotExtSignal(y21=y21, recons21=extInfo[1], sigma21=extInfo[3])
