# Copyright (c) 2024, Markus Gaug
# All rights reserved.
# 
# This source code is licensed under the GNU General Public License v3.0 found in the
# LICENSE file in the root directory of this source tree. 
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sbn
from cycler import cycler

def SetUp():
    sbn.set(rc={'figure.figsize':(10, 5)})
    plt.rcParams['font.size'] = 23       
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    # matplotlib stuff
    size=25
    params = {'legend.fontsize': 22,
              'figure.figsize': (8,7),
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size,
              'ytick.labelsize': size,
              'axes.titlepad': 25}
    plt.rcParams.update(params)

    colors = list(plt.cm.tab10(range(10)))
    colors.append(tuple(mcolors.hex2color(mcolors.cnames["crimson"])))
    colors.append(tuple(mcolors.hex2color(mcolors.cnames["indigo"])))
    #print (colors)
    default_cycler = (cycler(color=colors)) + cycler(linestyle=['-', '--', ':', '-.',
                                                                '-', '--', ':', '-.',
                                                                '-', '--', ':', '-.'])
    plt.rc('axes', prop_cycle=default_cycler)
