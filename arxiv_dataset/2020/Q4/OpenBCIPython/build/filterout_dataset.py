import json
import pandas as pd
import numpy as np
import json
import scipy.linalg as lin
import pandas as pd
import sys
from pandas import DataFrame, Series
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import pandas as pd

properdataset_ranges_bycept =[[1400,7000],[11100, 19500],[26000,36000]]
properdataset_ranges_straight_up =[[1000,8000],[8500, 12500]]

properdataset_ranges = properdataset_ranges_straight_up
# datasource = "/home/runge/openbci/git/OpenBCI_Python/build/dataset2017-5-5_23-55-32new_bycept.csv"
datasource = "/home/runge/openbci/git/OpenBCI_Python/build/dataset2017-5-6_0-0-33new_up.csv"
dataset_filtered_location = "/home/runge/openbci/git/OpenBCI_Python/build/dataset2017-5-5_23-55-32new_straight_up_filttered.csv"

raw_data = pd.read_csv(datasource)
filtered_dataset = pd.DataFrame()
for proper_boundary in properdataset_ranges:
    filtered_dataset=filtered_dataset.append(raw_data[proper_boundary[0]:proper_boundary[1]], ignore_index = True)
    print filtered_dataset.shape
with open(dataset_filtered_location,'w') as f:
    np.savetxt(f, np.array(filtered_dataset), delimiter=',', fmt='%.18e')
