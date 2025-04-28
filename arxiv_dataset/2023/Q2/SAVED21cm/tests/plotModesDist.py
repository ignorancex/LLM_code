import sys

from numpy import vsplit; sys.path.insert(1, './../')
from src.visuals import Visual
import settings as set

set.checkSettings()

visual = Visual(nu=set.NU, nLST=set.LST, ant=set.ANT)
visual.plotModesDist(file='./StatsOutput/Stats_lst-1_ant-dipole_DIC.txt',
                     modesFg=set.MODES_FG, modes21=set.MODES_21, contour=True,
                     vmax=10, nLevels=41, zoom=[(1, 10), (1, 30)])
