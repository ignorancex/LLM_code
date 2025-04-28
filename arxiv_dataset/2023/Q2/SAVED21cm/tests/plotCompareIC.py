import sys; sys.path.insert(1, './../')
import settings as set
from src.visuals import Visual

visual = Visual(nu=set.NU, nLST=set.LST, ant=set.ANT, save=False)

antNames = ['BIC', 'DIC']
fnames = ['./StatsOutput/Stats_lst-2_ant-dipole_BIC.txt',
          './StatsOutput/Stats_lst-2_ant-dipole_DIC.txt']

visual.plotBiasCDF(antNames=antNames, fnames=fnames, xlim=[0, 4])
