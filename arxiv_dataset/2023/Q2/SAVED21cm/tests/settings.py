import numpy as np

NU = np.linspace(50, 200, 151)      # Frequency Range

PATH21TS = '/Users/anchal/Documents/SourceCodes/SAVED21cm/data/TS21/lfcal_training_set_8-2020.hdf5'
PATHFGTS = '/Users/anchal/Documents/SourceCodes/SAVED21cm/data/TSFG/Foreground-training-set-haslam-beta-const-'

LST = 2     # Number of time bins
ANT = ['dipole']  # Antenna designs: 'dipole', 'logspiral', 'sinuous'
dNU = 1.    # Frequency channel width
dT = 6      # Integration time in hour
MODES_FG = 50     # Number of FG modes to search for min IC
MODES_21 = 80     # Number of 21 modes to search for min IC
QUANTITY = 'DIC'  # IC to minimize 'DIC' or 'BIC'
VISUALS = True    # Option to plot figures
SAVE = False      # Option to save figures
FNAME = './%s_search.txt'%QUANTITY    # Filename of IC information

def checkSettings():
      print('\n------------------ Settings for the pipeline ------------------\n')
      print('Frequencies: Between (%d - %d) MHz with step size of %d MHz.'
            %(NU[0], NU[-1], NU[1] - NU[0]))
      print('Number of LST bins:', LST)
      print('Antenna Design:', ANT)
      print('Integration Time:', dT)
      print('Total number of foreground modes:', MODES_FG)
      print('Total number of 21cm modes:', MODES_21)
      print('Information Criterion:', QUANTITY)
      print('Filename to store the info criteria:', FNAME)
      print('Visualization:', VISUALS)
      print('Save figures:', SAVE, '\n')
