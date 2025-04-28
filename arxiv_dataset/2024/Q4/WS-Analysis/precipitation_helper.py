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
import pandas as pd
import numpy as np

#
# The following expressions stem from
# B. Ding, K. Yang, J. Qin, L. Wang, Y. Chen, and X. He.
# The dependence of precipitation types on surface elevation and meteorological conditions and its parameterization.
# Journal of Hydrology, 513:154– 163, 2014.
#

#
# All RH's are used in percent and converted internally to [0,1]
#

# Eq. (7)
def DeltaT(RH):
    return 0.215 - 0.099*(RH/100) + 1.018*(RH/100)**2

# Eq. (8)
def DeltaS(RH):
    return 2.374 - 1.634*(RH/100)

# Eq. (9)
def T0(RH,Z=2.2):
    return -5.87 - 0.1042 * Z + 0.0885*Z**2 + 16.06*(RH/100) - 9.614*(RH/100)**2

# Eq. (13)
def Tmin(RH,Z=2.2):
    DT = DeltaT(RH)
    DS = DeltaS(RH)

#    if (isinstance(RH, (int, float))):
#        if (DT/DS <= np.log(2)):
#            return T0(RH,Z)
#
#        return T0(RH,Z) - DS*np.log(np.exp(DT/DS)-2*np.exp(-DT/DS))
#    if (isinstance(RH, (list, tuple, np.ndarray))):
    return np.where(DT/DS <= np.log(2), T0(RH,Z), T0(RH,Z) - DS*np.log(np.exp(DT/DS)-2*np.exp(-DT/DS)) )

# Eq. (14)
def Tmax(RH,Z=2.2):
    DT = DeltaT(RH)
    DS = DeltaS(RH)

#    if (isinstance(RH, (int, float))):
#        if (DT/DS <= np.log(2)):
#            return T0(RH,Z)
#
#        return 2*T0(RH,Z) - Tmin(RH,Z)
#    if (isinstance(RH, (list, tuple, np.ndarray))):
    return np.where(DT/DS <= np.log(2), T0(RH,Z), 2*T0(RH,Z) - Tmin(RH,Z))

#
# calculation of wet bulb temperature according to the Stull approximation
# (Stull, R. Journal of Applied Meteorology and Climatology 50 (2011) 2267
#
def Tw(RH,T):

    a1 = 0.151977
    a2 = 8.313659
    a3 = 1.676331
    a4 = 0.00391838
    a5 = 0.023101
    a6 = 4.686035

    at1 = T*np.arctan(a1*np.power(RH+a2,0.5))
    at2 = np.arctan(T+RH)
    at3 = np.arctan(RH-a3)
    at4 = a4*np.power(RH,1.5)*np.arctan(a5*RH)

    return at1 + at2 - at3 + at4 - a6
        

