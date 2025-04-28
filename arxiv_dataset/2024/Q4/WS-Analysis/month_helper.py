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
from astropy.time import Time
import numpy as np

def isleap(year):
    """Return True for leap years, False for non-leap years."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def monthlen(mjd): 

    mdays = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    February = 2

    if (isinstance(mjd, (list, tuple, np.ndarray))):
        if (len(mjd) == 0):
            return 1
    if (isinstance(mjd, (int, float))):
        t = Time(mjd+55000.,format='mjd')
        print ("MONTH: ", t.ymdhms['month'], "YEAR: ", t.ymdhms['year'])
        return mdays[t.ymdhms['month']] + (t.ymdhms['month'] == February and isleap(t.ymdhms['year']))
    t = Time(mjd+55000.,format='mjd')
    if (len(mjd) == 0):
        return 1
    return mdays[t.ymdhms['month'][0]] + (t.ymdhms['month'][0] == February and isleap(t.ymdhms['year'][0]))


