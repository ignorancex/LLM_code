''' A simple script to run the pipeline using runPipeline() '''

import sys; sys.path.insert(1, './../')
from src.runpipe import Pipeline
import settings as set

set.checkSettings()
pipe = Pipeline(nu=set.NU, nLST=set.LST, ant=set.ANT, path21TS=set.PATH21TS, pathFgTS=set.PATHFGTS,
                dT=set.dT, modesFg=set.MODES_FG, modes21=set.MODES_21, quantity=set.QUANTITY,
                file=set.FNAME, indexFg=0, index21=1000, visual=set.VISUALS, save=set.SAVE)
pipe.runPipeline()
