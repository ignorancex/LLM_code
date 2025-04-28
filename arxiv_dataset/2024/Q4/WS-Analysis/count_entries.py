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

h5file_short = 'WS2003-23_short.h5'
dff = pd.read_hdf(h5file_short)

pd.set_option('display.max_columns', 28)
pd.set_option('display.max_rows', 150)
#print (dff.head(n=20))

# TEST for temperature unreliables
mask1 = ((dff.index > '2010-11-26 15:40') & (dff.index < '2010-11-26 16:10'))
print ('TEST1: ', dff.loc[mask1,'temperature_reliable'])
mask2 = ((dff.index > '2010-10-11 12:40') & (dff.index < '2010-10-11 13:10'))
print ('TEST2: ', dff.loc[mask2,'temperature_reliable'])


for var in ['temperature', 'pressure', 'humidity']:
    mask   = (dff[var].notnull())
    mask_q = (dff[var].notnull() & (dff[var+'_reliable']==True))
    mask_b = (dff[var].isnull())
    mask_u = (dff[var+'_reliable']==False)
    print (var,' all: ', dff[var].value_counts(dropna=False).sum(), 
           '  valid: ',dff[var].value_counts(dropna=True).sum(),
           '  quality: ',dff.loc[mask_q,var].count(),
           ' bad: ',dff[var].isnull().sum(),
           ' unreliable: ',dff.loc[mask_u,var].count())

for var in ['rain', 'windSpeedCurrent', 'windSpeedAverage', 'windGust', 'windDirection', 'windDirectionAverage']:
    mask   = (dff[var].notnull())
    mask_b = (dff[var].isnull())
    print (var,'  all: ', dff[var].value_counts(dropna=False).sum(),
           ' valid: ', dff[var].value_counts(dropna=True).sum(),
           ' bad: ',dff[var].isnull().sum())
    

print ('wind reliable', dff.loc[dff['wind_reliable']==True,'windSpeedCurrent'].count())
