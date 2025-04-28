import sys; sys.path.insert(1, './../')
from src.getstats import StatQuants
import settings as set
import numpy as np

set.LST = 2
set.QUANTITY = 'BIC'
set.checkSettings()

statquants = StatQuants(nu=set.NU, nLST=set.LST, ant=set.ANT,
                        path21TS=set.PATH21TS, pathFgTS=set.PATHFGTS,
                        dT=set.dT, modesFg=set.MODES_FG, modes21=set.MODES_21,
                        quantity=set.QUANTITY, file=set.FNAME)

statquants.getStats(fname='Stats_lst-%d_ant-%s_%s.txt'
                    %(set.LST, '-'.join(set.ANT), set.QUANTITY),
                    iList21=np.linspace(0, 20_000, 50),
                    iListFg=np.linspace(0, 100, 5))
