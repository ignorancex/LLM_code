import sys; sys.path.insert(1, './../')
from src.visuals import Visual
import settings as set

visual = Visual(nu=set.NU, nLST=set.LST, ant=set.ANT)

antNames = [r'Dipole $(t_{\rm bin} = 1)$', r'Dipole $(t_{\rm bin} = 2)$']
fnames = ['./StatsOutput/Stats_lst-1_ant-dipole_DIC.txt',
          './StatsOutput/Stats_lst-2_ant-dipole_DIC.txt']

visual.plotBiasCDF(antNames=antNames, fnames=fnames, xlim=[0, 10], save=False)
visual.plotNormD(antNames=antNames, fnames=fnames, xlim=[0.5, 1.5], save=False)
visual.plotRmsCDF(antNames=antNames, fnames=fnames, xlim=[1e0, 1e4], save=False)
