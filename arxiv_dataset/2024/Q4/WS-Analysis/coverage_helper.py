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
from month_helper import monthlen
import timeit, functools

data_spacing_minutes = 2
expected_data_per_day = 60*24/data_spacing_minutes

def apply_coverage(df, data_spacing=data_spacing_minutes, debug=False, fulldebug=False):
    '''
    Add a new column to df containing the data coverage (in percent) of the corresponding month

    To achieve this, we: 
    1) Count the number of entries of a given month. 
       For full coverage, this should lead to: expected_data_per_day * days_per_month 
    2) Use the function monthlen(mjd) to calculate the length of each month in days
    3) Calculate the monthly coverage as  counts per month / month length in days / expected_data_per_day * 100
    4) Use the pd.factorize function to assign an increasing month index to each column of df 
    5) Use the pd.apply function to convert that index to the monthly coverage calculated in 3)
    '''

    if (debug):
        print ('COUNT TEST WITH RESAMPLE: ',df["mjd"].resample('M').count())    
        print ('COUNT TEST WITH GROUPBY: ',df["mjd"].groupby(pd.Grouper(freq = 'M')).count())
        print ('MONTH LENGTH TEST: ',df["mjd"].groupby(pd.Grouper(freq = 'M')).agg(lambda x: monthlen(x)))
        print ('COVERAGE TEST: ',df["mjd"].groupby(pd.Grouper(freq = 'M')).count()/df["mjd"].groupby(pd.Grouper(freq = 'M')).agg(lambda x: monthlen(x))/7.2)
        if (fulldebug):
            t = timeit.Timer(lambda: df['mjd'].groupby(pd.Grouper(freq = 'M')).count()/df['mjd'].groupby(pd.Grouper(freq = 'M')).agg(lambda x: monthlen(x))/7.2)
            print ("TIMEIT GROUPBY:",t.timeit(1))
            t = timeit.Timer(lambda: df['mjd'].resample('M').count()/df['mjd'].resample('M').agg(lambda x: monthlen(x))/7.2)
            print ("TIMEIT RESAMPLE:",t.timeit(1))    
        print ("TEST RESAMPLE2:", df["mjd"].resample('M',offset='15D').median())
        print ("TEST RESAMPLE3:", df["mjd"].resample('M',offset='15D').median().dropna())

        
    # Steps 1-3: calculate coverage
    expected_data = 60*24/data_spacing
    cover = df["mjd"].groupby(pd.Grouper(freq = 'M')).count()/df["mjd"].groupby(pd.Grouper(freq = 'M')).agg(lambda x: monthlen(x)) / expected_data * 100
    if (debug):
        print ('COVER: ', cover)    
    
    # Have to convert the pandas Series object to a numpy array in order to be able to index it
    cover_arr = cover.to_numpy()

    # BEWARE OF DATA GAPS!!
    # pd.factorize does NOT increase index for months with no data, where pd.count and pd.agg DO create an entry for no data
    cover_arr = cover_arr[np.nonzero(cover_arr)]

    # On the other hand, keep the zero entries in the Series 'cover', which may be used later in e.g. resample.mask
    # DO NOT REMOVE THE COMMENT HERE FOR DEFAULT use of these scripts!
    # cover = cover[cover!=0]

    # Step 4: Assign an increasing monthly index to each data point
    df['coverage'] = pd.factorize(df.groupby(pd.Grouper(freq = 'M'))['mjd'].transform('first'))[0]
    
    if (debug):
        if (fulldebug):
            pd.set_option('display.max_rows', 1000)
        print ("COVERAGE1: ", cover)
        print ("COVERAGE_ARR: ", cover_arr)
        if (fulldebug):
            print ("FACTORIZE TEST: ", df['coverage'].to_string())
        print ("FACTORIZE TEST: ", df['coverage'])        

    # Necessary test for if the gap removal has worked out correctly
    if (cover_arr.size-1 != df['coverage'][-1]):
        print ("GAPS PROBLEM IN COVERAGE CALCULATION CANNOT NOT BE SOVLED: cover_arr size-1: ", cover_arr.size-1,
               " vs. last entry of factorize: ", df['coverage'][-1])
        return None, None

        
    # Step 5: Convert the monthly index to the corresponding coverage calculated before
    df['coverage'] = df['coverage'].apply(lambda x: cover_arr[x])       

    if (debug):
        if (fulldebug):        
            print ("FINAL TEST: ", df['coverage'].to_string())
        print ("FINAL TEST: ", df['coverage'])        

    # return both the amplified full data frame and the monthly coverage series
    return df, cover
