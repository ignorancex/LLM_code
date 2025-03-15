"""
Created on Wed Sep 11 11:38:02 2019

@author: Neven Caplar
ncaplar@princeton.edu
www.ncaplar.com
 
"""

import numpy as np
import scipy
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances

def bootstrap_resample(X, n=None):
    """ 
    
    Bootstrap resample an array_like
    
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None
    Results
    -------
    returns X_resamples
    """
    if n == None:
        n = len(X)
        
    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = X[resample_i]
    return X_resample

def sort_and_split_in_quantiles(array_to_sort,colum_to_sort,num_of_quantiles,multi_dim_array=True):
    """
    
    split an array in quantiles, after sorting according to values in a single column
    
    Parameters
    ----------
    array_to_sort : array_like
      array which we wish to sort
    colum_to_sort : int
      according to which column do we wish to sort the array
    num_of_quantiles : int
      split in how many quantiles
    Results
    -------
    returns array, split in num_of_quantiles lists
    """
    sorted_and_split_array=[]
    if multi_dim_array==True:
        for i in range(len(array_to_sort)):
            sorted_and_split_array.append(np.array_split(array_to_sort[i][np.argsort(array_to_sort[i][:,colum_to_sort])],num_of_quantiles))
    if multi_dim_array==False:
            sorted_and_split_array=np.array_split(array_to_sort[np.argsort(array_to_sort[:,colum_to_sort])],num_of_quantiles)
    return sorted_and_split_array

def create_res_delta(delta_g_array):
    """
    
    creates the redshift result
    
    Parameters
    ----------
    delta_g_array : array_like
        array contaning the differnces between two surveys, already split in redshift bins
    -------
    returns 5 arrays
        mean differencea as function of redshift
        median difference as a function of redshift
        error on the mean difference
        error on the median difference
        mean redshift of redshift bins
    """
    res_delta=[]
    res_delta_median=[]
    res_delta_err=[]
    res_delta_median_err=[]
    res_redshift=[]
    for i in range(len(delta_g_array)):
        array_of_differences_at_a_single_redshift=delta_g_array[i][:,0]
        means_of_array_of_differences_at_a_single_redshift=[]
        medians_of_array_of_differences_at_a_single_redshift=[]
        # bootstraping happens here
        for j in range(100):
            resampled_array=bootstrap_resample(array_of_differences_at_a_single_redshift)
            means_of_array_of_differences_at_a_single_redshift.append(np.mean(resampled_array))
            medians_of_array_of_differences_at_a_single_redshift.append(np.median(resampled_array))
            mean_and_median_result=np.mean(means_of_array_of_differences_at_a_single_redshift),np.mean(medians_of_array_of_differences_at_a_single_redshift),\
            np.std(means_of_array_of_differences_at_a_single_redshift),np.std(medians_of_array_of_differences_at_a_single_redshift)
        
        
        
        res_delta.append(mean_and_median_result[0])
        res_delta_median.append(mean_and_median_result[1])         
        res_delta_err.append(mean_and_median_result[2])
        res_delta_median_err.append(mean_and_median_result[3])
        res_redshift.append(np.mean(delta_g_array[i][:,1]))
        
    
    res_delta=np.array(res_delta)
    res_delta_median=np.array(res_delta_median)
    res_delta_err=np.array(res_delta_err)
    res_delta_median_err=np.array(res_delta_median_err)
    res_redshift=np.array(res_redshift)
    
    return res_delta,res_delta_median,res_delta_err,res_delta_median_err,res_redshift

def match_two_catalogs(catalog_1,catalog_2):
    """
    
    matches objects in two catalogues, in our case it matches QSO from SDSS and HSC
    
    Parameters
    ----------
    catalog_1 : array_like
        first catalog
    catalog_2 : array_like
        second catalog        
        
    -------
    returns list with position of same object in both catalogs
    """
    
    
    res_matching=[]
    for j in tqdm(range(len(catalog_1))):
        # finds distance from each of the objects in catalog_1 from the objects in catalog_2
        PositionOfQuasars_euclidean_distances=euclidean_distances([catalog_1[j]],catalog_2)[0]
        # what is the shortest distance that is avaliable
        shortest_distance=np.min(PositionOfQuasars_euclidean_distances)
       
        if shortest_distance<0.01:  
            # element of the catalog_2 that has the shortest distance to the SDSS QSO
            shortest_distance_index=np.where(PositionOfQuasars_euclidean_distances==shortest_distance)[0][0]
            res_matching.append([j,shortest_distance_index])
        else:
            pass
        
    return res_matching


def BendingPL(v,A,v_bend,a_low,a_high,c):
    '''
    TAKEN from https://github.com/samconnolly/DELightcurveSimulation/blob/master/DELCgen.py
    
    
    Bending power law function - returns power at each value of v, 
    where v is an array (e.g. of frequencies)
    
    inputs:
        v (array)       - input values
        A (float)       - normalisation 
        v_bend (float)  - bending frequency
        a_low ((float)  - low frequency index
        a_high float)   - high frequency index
        c (float)       - intercept/offset
    output:
        out (array)     - output powers
    '''
    numer = v**-a_low
    denom = 1 + (v/v_bend)**(a_high-a_low)
    out = A * (numer/denom) + c
    return out


# 
def create_redshift_result(matched_array_filtered,number_of_objects_in_bin,sdss_band_column=4,difference_sdss_HSC_columns=14,\
                           return_median_mag_values=False,separate_in_time_dif=False,time_dif_array=None,return_SDSS_ID=False):
    
    """ Master function to creat mean difference between SDSS and HSC as a function of redshift
    Parameters
    ----------
    matched_array_filtered : array_like,
      array which has magnitudes etc....
    number_of_objects_in_bin : int,
      number of objects in one redshift bin - this is before any possible split in luminosity or time!
    sdss_band_column : int,
        column number in which SDSS magnitudes are set
    difference_sdss_HSC_columns : int,
        how many columns to add to find the column in which HSC magnitudes are set    
    return_median_mag_values : bool
        if true return the mediam mag values of each bin?
    separate_in_time_dif : bool
        separate the result accoring to the time-separation between two measurments
    time_dif_array
        if separate_in_time_dif=True, supply array which contains information about time separation
    return_SDSS_ID :
        return SDSS ID of objects that go in each bin
      
    Results
    -------
    gives everything that you need to give the plot
    """
    if separate_in_time_dif==True:
        assert len(time_dif_array)==len(matched_array_filtered)

    if return_SDSS_ID==True:
        assert separate_in_time_dif==False
        assert return_median_mag_values==True
        
    # sdss magnitude mag - HSC magnitude mag
    # for example, it is sdss psf-g band mag - HSC psf-g band mag if you chose sdss_band_column=4 and difference_sdss_HSC_columns=14
    g_mag_dif=(matched_array_filtered[:,sdss_band_column]-matched_array_filtered[:,difference_sdss_HSC_columns+sdss_band_column]).astype(float)
    # error sdss  mag - HSC band mag
    # for example, error sdss g band mag - HSC g band mag f you chose sdss_band_column=4 and difference_sdss_HSC_columns=14
    g_mag_dif_err=np.sqrt(((matched_array_filtered[:,sdss_band_column+1]).astype(float))**2+((matched_array_filtered[:,difference_sdss_HSC_columns+sdss_band_column+1]).astype(float))**2)
    # insert differences in the catalog
    # this 4 has nothing to do with the ``sdss_band_column=4''
    matched_array_filtered_with_g_mag_dif=np.insert(matched_array_filtered, 4, g_mag_dif, axis=1)
    matched_array_filtered_with_g_mag_dif_and_err=np.insert(matched_array_filtered_with_g_mag_dif, 5, g_mag_dif_err, axis=1)
    
    # array with has delta g as the first column, redshift as the second column
    delta_g_and_redshift=matched_array_filtered_with_g_mag_dif_and_err[:,[4,3]]
    # array with has delta g as the first column, redshift as the second column, 3rd colum is the magnitude of the objects in SDSS
    delta_g_and_redshift_and_g=matched_array_filtered_with_g_mag_dif_and_err[:,[4,3,6]]
    if return_SDSS_ID==True:
        # array with has delta g as the first column, redshift as the second column, 3rd column is the magnitude of the objects, 4th column is the SDSS ID for the objects
        delta_g_and_redshift_and_g_and_SDSS_ID=matched_array_filtered_with_g_mag_dif_and_err[:,[4,3,6,0]]
    
    
    if separate_in_time_dif==True:
        
        # array with has delta g as the first column, redshift as the second column, 3rd is the time separation of the observations
        delta_g_and_redshift=matched_array_filtered_with_g_mag_dif_and_err[:,[4,3]]
        delta_g_and_redshift=np.column_stack((delta_g_and_redshift,time_dif_array))        
        # array with has delta g as the first column, redshift as the second column, 3rd colum is the magnitude of the objects, 4th is the time separation of the observations
        delta_g_and_redshift_and_g=matched_array_filtered_with_g_mag_dif_and_err[:,[4,3,6]]
        delta_g_and_redshift_and_g=np.column_stack((delta_g_and_redshift_and_g,time_dif_array))
        

    # previous array (delta_g_and_redshift), sorted by redshift first and then split in bins, number_of_objects_in_bin objects in each bin
    delta_g_and_redshift_sorted_by_redshift_g_and_split=np.array_split(delta_g_and_redshift[np.argsort(delta_g_and_redshift[:,1])],int(len(delta_g_and_redshift)/number_of_objects_in_bin))
    # previous array (delta_g_and_redshift_and_g), sorted by redshift first and then split in bins, number_of_objects_in_bin objects in each bin
    delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split=np.array_split(delta_g_and_redshift_and_g[np.argsort(delta_g_and_redshift_and_g[:,1])],\
                                                                             int(len(delta_g_and_redshift_and_g)/number_of_objects_in_bin))

    if return_SDSS_ID==True:
        # previous array (delta_g_and_redshift_and_g_and_SDSS_ID), sorted by redshift first and then split in bins, number_of_objects_in_bin objects in each bin
        delta_g_and_redshift_and_g_and_SDSS_ID_sorted_by_redshift_g_and_split=np.array_split(delta_g_and_redshift_and_g_and_SDSS_ID[np.argsort(delta_g_and_redshift_and_g_and_SDSS_ID[:,1])],\
                                                                             int(len(delta_g_and_redshift_and_g_and_SDSS_ID)/number_of_objects_in_bin))
    

    if separate_in_time_dif==True:
        # column with index 2 contains times
        delta_g_and_redshift_sorted_by_redshift_g_and_split_short_time=np.array(sort_and_split_in_quantiles(delta_g_and_redshift_sorted_by_redshift_g_and_split,2,5,True))[:,0]
        delta_g_and_redshift_sorted_by_redshift_g_and_split_long_time=np.array(sort_and_split_in_quantiles(delta_g_and_redshift_sorted_by_redshift_g_and_split,2,5,True))[:,-1]
  
        # column with index 3 contains times
        delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_short_time=np.array(sort_and_split_in_quantiles(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split,3,5,True))[:,0]
        delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_long_time=np.array(sort_and_split_in_quantiles(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split,3,5,True))[:,-1]
    
    # Divided in quantiles, in each quantile put number_of_objects_in_bin/5
    # array to sort, according to which column, into how many separations
    delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g=sort_and_split_in_quantiles(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split,2,5)

    if return_SDSS_ID==True:
        # Divided in quantiles, in each quantile put number_of_objects_in_bin/5
        delta_g_and_redshift_and_g_and_SDSS_ID_sorted_by_redshift_g_and_split_sorted_by_g=sort_and_split_in_quantiles(delta_g_and_redshift_and_g_and_SDSS_ID_sorted_by_redshift_g_and_split,2,5)

    # median g with redshift for the whole sample
    # this is mostly for checking purposes and consistency - the results should be same as for median_g_with_redshift_40_60
    median_g_with_redshift=[]
    for i in range(len(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split)):
        median_g_with_redshift.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split[i][:,[2]]))
    if separate_in_time_dif==True:
        median_g_with_redshift_short_time=[]
        for i in range(len(delta_g_and_redshift_sorted_by_redshift_g_and_split_short_time)):
            median_g_with_redshift_short_time.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_short_time[i][:,[2]]))
        median_g_with_redshift_long_time=[]
        for i in range(len(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_long_time)):
            median_g_with_redshift_long_time.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_long_time[i][:,[2]]))
            
        
    # now separated in the luminosity bins
    # list which containts delta g-values and redshift
    delta_g_and_redshift_0_20=[]
    # list which containts median g-values
    median_g_with_redshift_0_20=[]
    delta_g_and_redshift_20_40=[]
    median_g_with_redshift_20_40=[]
    delta_g_and_redshift_40_60=[]
    median_g_with_redshift_40_60=[]
    delta_g_and_redshift_60_80=[]
    median_g_with_redshift_60_80=[]
    delta_g_and_redshift_80_100=[]
    median_g_with_redshift_80_100=[]
    
    delta_g_and_redshift_0_20_short_time=[]
    median_g_with_redshift_0_20_short_time=[]
    delta_g_and_redshift_20_40_short_time=[]
    median_g_with_redshift_20_40_short_time=[]
    delta_g_and_redshift_40_60_short_time=[]
    median_g_with_redshift_40_60_short_time=[]
    delta_g_and_redshift_60_80_short_time=[]
    median_g_with_redshift_60_80_short_time=[]
    delta_g_and_redshift_80_100_short_time=[]
    median_g_with_redshift_80_100_short_time=[]
    
    delta_g_and_redshift_0_20_long_time=[]
    median_g_with_redshift_0_20_long_time=[]
    delta_g_and_redshift_20_40_long_time=[]
    median_g_with_redshift_20_40_long_time=[]
    delta_g_and_redshift_40_60_long_time=[]
    median_g_with_redshift_40_60_long_time=[]
    delta_g_and_redshift_60_80_long_time=[]
    median_g_with_redshift_60_80_long_time=[]
    delta_g_and_redshift_80_100_long_time=[]
    median_g_with_redshift_80_100_long_time=[]
    
    for i in range(len(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g)):
        if separate_in_time_dif==False:
            delta_g_and_redshift_0_20.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][0][:,[0,1]])
            delta_g_and_redshift_20_40.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][1][:,[0,1]])
            delta_g_and_redshift_40_60.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][2][:,[0,1]])
            delta_g_and_redshift_60_80.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][3][:,[0,1]])
            delta_g_and_redshift_80_100.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][4][:,[0,1]])

            median_g_with_redshift_0_20.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][0][:,[2]]))
            median_g_with_redshift_20_40.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][1][:,[2]]))
            median_g_with_redshift_40_60.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][2][:,[2]]))
            median_g_with_redshift_60_80.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][3][:,[2]]))
            median_g_with_redshift_80_100.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][4][:,[2]]))
        if separate_in_time_dif==True:

            # same as when separate_in_time_dif==False
            delta_g_and_redshift_0_20.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][0][:,[0,1]])
            delta_g_and_redshift_20_40.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][1][:,[0,1]])
            delta_g_and_redshift_40_60.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][2][:,[0,1]])
            delta_g_and_redshift_60_80.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][3][:,[0,1]])
            delta_g_and_redshift_80_100.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][4][:,[0,1]])

            median_g_with_redshift_0_20.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][0][:,[2]]))
            median_g_with_redshift_20_40.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][1][:,[2]]))
            median_g_with_redshift_40_60.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][2][:,[2]]))
            median_g_with_redshift_60_80.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][3][:,[2]]))
            median_g_with_redshift_80_100.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][4][:,[2]]))   
 
            # brightest objects separated in time
            delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time=\
            sort_and_split_in_quantiles(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][0],3,5,False)
            #import pdb; pdb.set_trace()
            delta_g_and_redshift_0_20_short_time.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[0][:,[0,1]])
            median_g_with_redshift_0_20_short_time.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[0][:,[2]]))
            delta_g_and_redshift_0_20_long_time.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[4][:,[0,1]])
            median_g_with_redshift_0_20_long_time.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[4][:,[2]]))
            # 20%-40% objects separated in time
            delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time=\
            sort_and_split_in_quantiles(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][1],3,5,False)
            delta_g_and_redshift_20_40_short_time.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[0][:,[0,1]])
            median_g_with_redshift_20_40_short_time.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[0][:,[2]]))
            delta_g_and_redshift_20_40_long_time.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[4][:,[0,1]])
            median_g_with_redshift_20_40_long_time.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[4][:,[2]]))
            # 40%-60% objects separated in time
            delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time=\
            sort_and_split_in_quantiles(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][2],3,5,False)
            delta_g_and_redshift_40_60_short_time.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[0][:,[0,1]])
            median_g_with_redshift_40_60_short_time.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[0][:,[2]]))
            delta_g_and_redshift_40_60_long_time.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[4][:,[0,1]])    
            median_g_with_redshift_40_60_long_time.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[4][:,[2]]))
            # 60%-80% objects separated in time
            delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time=\
            sort_and_split_in_quantiles(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][3],3,5,False)
            delta_g_and_redshift_60_80_short_time.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[0][:,[0,1]])
            median_g_with_redshift_60_80_short_time.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[0][:,[2]]))
            delta_g_and_redshift_60_80_long_time.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[4][:,[0,1]])
            median_g_with_redshift_60_80_long_time.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[4][:,[2]]))
            # 80%-100% objects separated in time
            delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time=\
            sort_and_split_in_quantiles(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g[i][4],3,5,False)
            delta_g_and_redshift_80_100_short_time.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[0][:,[0,1]])
            median_g_with_redshift_80_100_short_time.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[0][:,[2]]))
            delta_g_and_redshift_80_100_long_time.append(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[4][:,[0,1]])
            median_g_with_redshift_80_100_long_time.append(np.median(delta_g_and_redshift_and_g_sorted_by_redshift_g_and_split_sorted_by_g_and_split_sorted_by_time[4][:,[2]]))


    # full array
    res_delta_redshift_via_redshift,res_delta_redshift_via_redshift_median,res_delta_redshift_via_redshift_err,\
    res_delta_redshift_via_redshift_median_err,res_redshift=create_res_delta(delta_g_and_redshift_sorted_by_redshift_g_and_split)

    # 0-20
    res_delta_redshift_via_redshift_0_20,res_delta_redshift_via_redshift_median_0_20,res_delta_redshift_via_redshift_err_0_20,\
    res_delta_redshift_via_redshift_median_err_0_20,res_redshift_0_20=create_res_delta(delta_g_and_redshift_0_20)
    # 20-40
    res_delta_redshift_via_redshift_20_40,res_delta_redshift_via_redshift_median_20_40,res_delta_redshift_via_redshift_err_20_40,\
    res_delta_redshift_via_redshift_median_err_20_40,res_redshift_20_40=create_res_delta(delta_g_and_redshift_20_40)
    # 40-60
    res_delta_redshift_via_redshift_40_60,res_delta_redshift_via_redshift_median_40_60,res_delta_redshift_via_redshift_err_40_60,\
    res_delta_redshift_via_redshift_median_err_40_60,res_redshift_40_60=create_res_delta(delta_g_and_redshift_40_60)
    # 60-80
    res_delta_redshift_via_redshift_60_80,res_delta_redshift_via_redshift_median_60_80,res_delta_redshift_via_redshift_err_60_80,\
    res_delta_redshift_via_redshift_median_err_60_80,res_redshift_60_80=create_res_delta(delta_g_and_redshift_60_80)
    # 80-100
    res_delta_redshift_via_redshift_80_100,res_delta_redshift_via_redshift_median_80_100,res_delta_redshift_via_redshift_err_80_100,\
    res_delta_redshift_via_redshift_median_err_80_100,res_redshift_80_100=create_res_delta(delta_g_and_redshift_80_100)
    
    if separate_in_time_dif==True:
        # everything
        res_delta_redshift_via_redshift_short_time,res_delta_redshift_via_redshift_median_short_time,res_delta_redshift_via_redshift_err_short_time,\
        res_delta_redshift_via_redshift_median_err_short_time,res_redshift_short_time=create_res_delta(delta_g_and_redshift_sorted_by_redshift_g_and_split_short_time)        
        # 0-20
        res_delta_redshift_via_redshift_0_20_short_time,res_delta_redshift_via_redshift_median_0_20_short_time,res_delta_redshift_via_redshift_err_0_20_short_time,\
        res_delta_redshift_via_redshift_median_err_0_20_short_time,res_redshift_0_20_short_time=create_res_delta(delta_g_and_redshift_0_20_short_time)
        # 20-40
        res_delta_redshift_via_redshift_20_40_short_time,res_delta_redshift_via_redshift_median_20_40_short_time,res_delta_redshift_via_redshift_err_20_40_short_time,\
        res_delta_redshift_via_redshift_median_err_20_40_short_time,res_redshift_20_40_short_time=create_res_delta(delta_g_and_redshift_20_40_short_time)
        # 40-60
        res_delta_redshift_via_redshift_40_60_short_time,res_delta_redshift_via_redshift_median_40_60_short_time,res_delta_redshift_via_redshift_err_40_60_short_time,\
        res_delta_redshift_via_redshift_median_err_40_60_short_time,res_redshift_40_60_short_time=create_res_delta(delta_g_and_redshift_40_60_short_time)
        # 60-80
        res_delta_redshift_via_redshift_60_80_short_time,res_delta_redshift_via_redshift_median_60_80_short_time,res_delta_redshift_via_redshift_err_60_80_short_time,\
        res_delta_redshift_via_redshift_median_err_60_80_short_time,res_redshift_60_80_short_time=create_res_delta(delta_g_and_redshift_60_80_short_time)
        # 80-100
        res_delta_redshift_via_redshift_80_100_short_time,res_delta_redshift_via_redshift_median_80_100_short_time,res_delta_redshift_via_redshift_err_80_100_short_time,\
        res_delta_redshift_via_redshift_median_err_80_100_short_time,res_redshift_80_100_short_time=create_res_delta(delta_g_and_redshift_80_100_short_time)

        # everything
        res_delta_redshift_via_redshift_long_time,res_delta_redshift_via_redshift_median_long_time,res_delta_redshift_via_redshift_err_long_time,\
        res_delta_redshift_via_redshift_median_err_long_time,res_redshift_long_time=create_res_delta(delta_g_and_redshift_sorted_by_redshift_g_and_split_long_time)           
        # 0-20
        res_delta_redshift_via_redshift_0_20_long_time,res_delta_redshift_via_redshift_median_0_20_long_time,res_delta_redshift_via_redshift_err_0_20_long_time,\
        res_delta_redshift_via_redshift_median_err_0_20_long_time,res_redshift_0_20_long_time=create_res_delta(delta_g_and_redshift_0_20_long_time)
        # 20-40
        res_delta_redshift_via_redshift_20_40_long_time,res_delta_redshift_via_redshift_median_20_40_long_time,res_delta_redshift_via_redshift_err_20_40_long_time,\
        res_delta_redshift_via_redshift_median_err_20_40_long_time,res_redshift_20_40_long_time=create_res_delta(delta_g_and_redshift_20_40_long_time)
        # 40-60
        res_delta_redshift_via_redshift_40_60_long_time,res_delta_redshift_via_redshift_median_40_60_long_time,res_delta_redshift_via_redshift_err_40_60_long_time,\
        res_delta_redshift_via_redshift_median_err_40_60_long_time,res_redshift_40_60_long_time=create_res_delta(delta_g_and_redshift_40_60_long_time)
        # 60-80
        res_delta_redshift_via_redshift_60_80_long_time,res_delta_redshift_via_redshift_median_60_80_long_time,res_delta_redshift_via_redshift_err_60_80_long_time,\
        res_delta_redshift_via_redshift_median_err_60_80_long_time,res_redshift_60_80_long_time=create_res_delta(delta_g_and_redshift_60_80_long_time)
        # 80-100
        res_delta_redshift_via_redshift_80_100_long_time,res_delta_redshift_via_redshift_median_80_100_long_time,res_delta_redshift_via_redshift_err_80_100_long_time,\
        res_delta_redshift_via_redshift_median_err_80_100_long_time,res_redshift_80_100_long_time=create_res_delta(delta_g_and_redshift_80_100_long_time)
        
        
    # full fit
    p20=np.poly1d(np.polyfit(res_redshift,res_delta_redshift_via_redshift,2,w=1/res_delta_redshift_via_redshift_err))
    p20_median=np.poly1d(np.polyfit(res_redshift,res_delta_redshift_via_redshift_median,2,w=1/res_delta_redshift_via_redshift_median_err))

    # fit to each quantile
    p20_0_20=np.poly1d(np.polyfit(res_redshift,res_delta_redshift_via_redshift_0_20,2,w=1/res_delta_redshift_via_redshift_err_0_20))
    p20_median_0_20=np.poly1d(np.polyfit(res_redshift,res_delta_redshift_via_redshift_median_0_20,2,w=1/res_delta_redshift_via_redshift_median_err_0_20))
    p20_20_40=np.poly1d(np.polyfit(res_redshift,res_delta_redshift_via_redshift_20_40,2,w=1/res_delta_redshift_via_redshift_err_20_40))
    p20_median_20_40=np.poly1d(np.polyfit(res_redshift,res_delta_redshift_via_redshift_median_20_40,2,w=1/res_delta_redshift_via_redshift_median_err_20_40))
    p20_40_60=np.poly1d(np.polyfit(res_redshift,res_delta_redshift_via_redshift_40_60,2,w=1/res_delta_redshift_via_redshift_err_40_60))
    p20_median_40_60=np.poly1d(np.polyfit(res_redshift,res_delta_redshift_via_redshift_median_40_60,2,w=1/res_delta_redshift_via_redshift_median_err_40_60))
    p20_60_80=np.poly1d(np.polyfit(res_redshift,res_delta_redshift_via_redshift_60_80,2,w=1/res_delta_redshift_via_redshift_err_60_80))
    p20_median_60_80=np.poly1d(np.polyfit(res_redshift,res_delta_redshift_via_redshift_median_60_80,2,w=1/res_delta_redshift_via_redshift_median_err_60_80))
    p20_80_100=np.poly1d(np.polyfit(res_redshift,res_delta_redshift_via_redshift_80_100,2,w=1/res_delta_redshift_via_redshift_err_80_100))
    p20_median_80_100=np.poly1d(np.polyfit(res_redshift,res_delta_redshift_via_redshift_median_80_100,2,w=1/res_delta_redshift_via_redshift_median_err_80_100))
    
    if separate_in_time_dif==True:
        # short time
        p20_short_time=np.poly1d(np.polyfit(res_redshift_short_time,res_delta_redshift_via_redshift_short_time,2,w=1/res_delta_redshift_via_redshift_err_short_time))
        p20_median_short_time=np.poly1d(np.polyfit(res_redshift_short_time,res_delta_redshift_via_redshift_median_short_time,2,w=1/res_delta_redshift_via_redshift_median_err_short_time))

        # fit to each quantile
        p20_0_20_short_time=np.poly1d(np.polyfit(res_redshift_short_time,res_delta_redshift_via_redshift_0_20_short_time,2,w=1/res_delta_redshift_via_redshift_err_0_20_short_time))
        p20_median_0_20_short_time=np.poly1d(np.polyfit(res_redshift_short_time,res_delta_redshift_via_redshift_median_0_20_short_time,2,w=1/res_delta_redshift_via_redshift_median_err_0_20_short_time))
        p20_20_40_short_time=np.poly1d(np.polyfit(res_redshift_short_time,res_delta_redshift_via_redshift_20_40_short_time,2,w=1/res_delta_redshift_via_redshift_err_20_40_short_time))
        p20_median_20_40_short_time=np.poly1d(np.polyfit(res_redshift_short_time,res_delta_redshift_via_redshift_median_20_40_short_time,2,w=1/res_delta_redshift_via_redshift_median_err_20_40_short_time))
        p20_40_60_short_time=np.poly1d(np.polyfit(res_redshift_short_time,res_delta_redshift_via_redshift_40_60_short_time,2,w=1/res_delta_redshift_via_redshift_err_40_60_short_time))
        p20_median_40_60_short_time=np.poly1d(np.polyfit(res_redshift_short_time,res_delta_redshift_via_redshift_median_40_60_short_time,2,w=1/res_delta_redshift_via_redshift_median_err_40_60_short_time))
        p20_60_80_short_time=np.poly1d(np.polyfit(res_redshift_short_time,res_delta_redshift_via_redshift_60_80_short_time,2,w=1/res_delta_redshift_via_redshift_err_60_80_short_time))
        p20_median_60_80_short_time=np.poly1d(np.polyfit(res_redshift_short_time,res_delta_redshift_via_redshift_median_60_80_short_time,2,w=1/res_delta_redshift_via_redshift_median_err_60_80_short_time))
        p20_80_100_short_time=np.poly1d(np.polyfit(res_redshift_short_time,res_delta_redshift_via_redshift_80_100_short_time,2,w=1/res_delta_redshift_via_redshift_err_80_100_short_time))
        p20_median_80_100_short_time=np.poly1d(np.polyfit(res_redshift_short_time,res_delta_redshift_via_redshift_median_80_100_short_time,2,w=1/res_delta_redshift_via_redshift_median_err_80_100_short_time))
    
        # long time 
        p20_long_time=np.poly1d(np.polyfit(res_redshift_long_time,res_delta_redshift_via_redshift_long_time,2,w=1/res_delta_redshift_via_redshift_err_long_time))
        p20_median_long_time=np.poly1d(np.polyfit(res_redshift_long_time,res_delta_redshift_via_redshift_median_long_time,2,w=1/res_delta_redshift_via_redshift_median_err_long_time))

        # fit to each quantile
        p20_0_20_long_time=np.poly1d(np.polyfit(res_redshift_long_time,res_delta_redshift_via_redshift_0_20_long_time,2,w=1/res_delta_redshift_via_redshift_err_0_20_long_time))
        p20_median_0_20_long_time=np.poly1d(np.polyfit(res_redshift_long_time,res_delta_redshift_via_redshift_median_0_20_long_time,2,w=1/res_delta_redshift_via_redshift_median_err_0_20_long_time))
        p20_20_40_long_time=np.poly1d(np.polyfit(res_redshift_long_time,res_delta_redshift_via_redshift_20_40_long_time,2,w=1/res_delta_redshift_via_redshift_err_20_40_long_time))
        p20_median_20_40_long_time=np.poly1d(np.polyfit(res_redshift_long_time,res_delta_redshift_via_redshift_median_20_40_long_time,2,w=1/res_delta_redshift_via_redshift_median_err_20_40_long_time))
        p20_40_60_long_time=np.poly1d(np.polyfit(res_redshift_long_time,res_delta_redshift_via_redshift_40_60_long_time,2,w=1/res_delta_redshift_via_redshift_err_40_60_long_time))
        p20_median_40_60_long_time=np.poly1d(np.polyfit(res_redshift_long_time,res_delta_redshift_via_redshift_median_40_60_long_time,2,w=1/res_delta_redshift_via_redshift_median_err_40_60_long_time))
        p20_60_80_long_time=np.poly1d(np.polyfit(res_redshift_long_time,res_delta_redshift_via_redshift_60_80_long_time,2,w=1/res_delta_redshift_via_redshift_err_60_80_long_time))
        p20_median_60_80_long_time=np.poly1d(np.polyfit(res_redshift_long_time,res_delta_redshift_via_redshift_median_60_80_long_time,2,w=1/res_delta_redshift_via_redshift_median_err_60_80_long_time))
        p20_80_100_long_time=np.poly1d(np.polyfit(res_redshift_long_time,res_delta_redshift_via_redshift_80_100_long_time,2,w=1/res_delta_redshift_via_redshift_err_80_100_long_time))
        p20_median_80_100_long_time=np.poly1d(np.polyfit(res_redshift_long_time,res_delta_redshift_via_redshift_median_80_100_long_time,2,w=1/res_delta_redshift_via_redshift_median_err_80_100_long_time))    
    # the ordering of these arrays is always:
    # full fit
    # fit to the brightest 20% of objects
    # fit to the 20-40% of objects
    # fit to the 40-60% of objects
    # fit to the 60-80% of objects
    # fit to the 80-100% of objects (dimmest 20%)
    res_delta_redshift_via_redshift_array=[res_delta_redshift_via_redshift,res_delta_redshift_via_redshift_0_20,res_delta_redshift_via_redshift_20_40,
                                          res_delta_redshift_via_redshift_40_60,res_delta_redshift_via_redshift_60_80,res_delta_redshift_via_redshift_80_100]
    res_delta_redshift_via_redshift_err_array=[res_delta_redshift_via_redshift_err,res_delta_redshift_via_redshift_err_0_20,res_delta_redshift_via_redshift_err_20_40,
                                          res_delta_redshift_via_redshift_err_40_60,res_delta_redshift_via_redshift_err_60_80,res_delta_redshift_via_redshift_err_80_100]    
    
    res_delta_redshift_via_redshift_median_array=[res_delta_redshift_via_redshift_median,res_delta_redshift_via_redshift_median_0_20,res_delta_redshift_via_redshift_median_20_40,
                                          res_delta_redshift_via_redshift_median_40_60,res_delta_redshift_via_redshift_median_60_80,res_delta_redshift_via_redshift_median_80_100]    
    res_delta_redshift_via_redshift_median_err_array=[res_delta_redshift_via_redshift_median_err,res_delta_redshift_via_redshift_median_err_0_20,res_delta_redshift_via_redshift_median_err_20_40,
                                          res_delta_redshift_via_redshift_median_err_40_60,res_delta_redshift_via_redshift_median_err_60_80,res_delta_redshift_via_redshift_median_err_80_100]    
    
    res_redshift_array=[res_redshift,res_redshift_0_20,res_redshift_20_40,res_redshift_40_60,res_redshift_60_80,res_redshift_80_100]
    
    median_g_with_redshift_array=[median_g_with_redshift,median_g_with_redshift_0_20,median_g_with_redshift_20_40,median_g_with_redshift_40_60,median_g_with_redshift_60_80,median_g_with_redshift_80_100]

    p20_array=[p20,p20_0_20,p20_20_40,p20_40_60,p20_60_80,p20_80_100]
    p20_median_array=[p20_median,p20_median_0_20,p20_median_20_40,p20_median_40_60,p20_median_60_80,p20_median_80_100]
    
    if separate_in_time_dif==True:
        # short time
        res_delta_redshift_via_redshift_short_time_array=[res_delta_redshift_via_redshift_short_time,res_delta_redshift_via_redshift_0_20_short_time,res_delta_redshift_via_redshift_20_40_short_time,
                                              res_delta_redshift_via_redshift_40_60_short_time,res_delta_redshift_via_redshift_60_80_short_time,res_delta_redshift_via_redshift_80_100_short_time]
        res_delta_redshift_via_redshift_err_short_time_array=[res_delta_redshift_via_redshift_err_short_time,res_delta_redshift_via_redshift_err_0_20_short_time,res_delta_redshift_via_redshift_err_20_40_short_time,
                                              res_delta_redshift_via_redshift_err_40_60_short_time,res_delta_redshift_via_redshift_err_60_80_short_time,res_delta_redshift_via_redshift_err_80_100_short_time]    

        res_delta_redshift_via_redshift_median_short_time_array=[res_delta_redshift_via_redshift_median_short_time,res_delta_redshift_via_redshift_median_0_20_short_time,res_delta_redshift_via_redshift_median_20_40_short_time,
                                              res_delta_redshift_via_redshift_median_40_60_short_time,res_delta_redshift_via_redshift_median_60_80_short_time,res_delta_redshift_via_redshift_median_80_100_short_time]    
        res_delta_redshift_via_redshift_median_err_short_time_array=[res_delta_redshift_via_redshift_median_err_short_time,res_delta_redshift_via_redshift_median_err_0_20_short_time,res_delta_redshift_via_redshift_median_err_20_40_short_time,
                                              res_delta_redshift_via_redshift_median_err_40_60_short_time,res_delta_redshift_via_redshift_median_err_60_80_short_time,res_delta_redshift_via_redshift_median_err_80_100_short_time]    

        res_redshift_short_time_array=[res_redshift_short_time,res_redshift_0_20_short_time,res_redshift_20_40_short_time,res_redshift_40_60_short_time,res_redshift_60_80_short_time,res_redshift_80_100_short_time]

        median_g_with_redshift_short_time_array=[median_g_with_redshift_short_time,median_g_with_redshift_0_20_short_time,median_g_with_redshift_20_40_short_time,median_g_with_redshift_40_60_short_time,median_g_with_redshift_60_80_short_time,median_g_with_redshift_80_100_short_time]
        
        p20_short_time_array=[p20_short_time,p20_0_20_short_time,p20_20_40_short_time,p20_40_60_short_time,p20_60_80_short_time,p20_80_100_short_time]
        p20_median_short_time_array=[p20_median_short_time,p20_median_0_20_short_time,p20_median_20_40_short_time,p20_median_40_60_short_time,p20_median_60_80_short_time,p20_median_80_100_short_time]
        
        # long time
        res_delta_redshift_via_redshift_long_time_array=[res_delta_redshift_via_redshift_long_time,res_delta_redshift_via_redshift_0_20_long_time,res_delta_redshift_via_redshift_20_40_long_time,
                                              res_delta_redshift_via_redshift_40_60_long_time,res_delta_redshift_via_redshift_60_80_long_time,res_delta_redshift_via_redshift_80_100_long_time]
        res_delta_redshift_via_redshift_err_long_time_array=[res_delta_redshift_via_redshift_err_long_time,res_delta_redshift_via_redshift_err_0_20_long_time,res_delta_redshift_via_redshift_err_20_40_long_time,
                                              res_delta_redshift_via_redshift_err_40_60_long_time,res_delta_redshift_via_redshift_err_60_80_long_time,res_delta_redshift_via_redshift_err_80_100_long_time]    

        res_delta_redshift_via_redshift_median_long_time_array=[res_delta_redshift_via_redshift_median_long_time,res_delta_redshift_via_redshift_median_0_20_long_time,res_delta_redshift_via_redshift_median_20_40_long_time,
                                              res_delta_redshift_via_redshift_median_40_60_long_time,res_delta_redshift_via_redshift_median_60_80_long_time,res_delta_redshift_via_redshift_median_80_100_long_time]    
        res_delta_redshift_via_redshift_median_err_long_time_array=[res_delta_redshift_via_redshift_median_err_long_time,res_delta_redshift_via_redshift_median_err_0_20_long_time,res_delta_redshift_via_redshift_median_err_20_40_long_time,
                                              res_delta_redshift_via_redshift_median_err_40_60_long_time,res_delta_redshift_via_redshift_median_err_60_80_long_time,res_delta_redshift_via_redshift_median_err_80_100_long_time]    

        res_redshift_long_time_array=[res_redshift_long_time,res_redshift_0_20_long_time,res_redshift_20_40_long_time,res_redshift_40_60_long_time,res_redshift_60_80_long_time,res_redshift_80_100_long_time]

        median_g_with_redshift_long_time_array=[median_g_with_redshift_long_time,median_g_with_redshift_0_20_long_time,median_g_with_redshift_20_40_long_time,median_g_with_redshift_40_60_long_time,median_g_with_redshift_60_80_long_time,median_g_with_redshift_80_100_long_time]

        p20_long_time_array=[p20_long_time,p20_0_20_long_time,p20_20_40_long_time,p20_40_60_long_time,p20_60_80_long_time,p20_80_100_long_time]
        p20_median_long_time_array=[p20_median_long_time,p20_median_0_20_long_time,p20_median_20_40_long_time,p20_median_40_60_long_time,p20_median_60_80_long_time,p20_median_80_100_long_time]
    

    
    if return_median_mag_values==False:
        return res_delta_redshift_via_redshift_array,res_delta_redshift_via_redshift_median_array,res_delta_redshift_via_redshift_err_array,\
        res_delta_redshift_via_redshift_median_err_array,res_redshift_array,p20_array,p20_median_array
    else:
        if separate_in_time_dif==False:
            if return_SDSS_ID==False:
                return res_delta_redshift_via_redshift_array,res_delta_redshift_via_redshift_median_array,res_delta_redshift_via_redshift_err_array,\
                res_delta_redshift_via_redshift_median_err_array,res_redshift_array,p20_array,p20_median_array,median_g_with_redshift_array
            if return_SDSS_ID==True:
                return res_delta_redshift_via_redshift_array,res_delta_redshift_via_redshift_median_array,res_delta_redshift_via_redshift_err_array,\
                res_delta_redshift_via_redshift_median_err_array,res_redshift_array,p20_array,p20_median_array,median_g_with_redshift_array,\
                delta_g_and_redshift_and_g_and_SDSS_ID_sorted_by_redshift_g_and_split_sorted_by_g
            
        if separate_in_time_dif==True:
            return [[res_delta_redshift_via_redshift_array,res_delta_redshift_via_redshift_median_array,res_delta_redshift_via_redshift_err_array,\
            res_delta_redshift_via_redshift_median_err_array,res_redshift_array,p20_array,p20_median_array,median_g_with_redshift_array],\
                   [res_delta_redshift_via_redshift_short_time_array,res_delta_redshift_via_redshift_median_short_time_array,res_delta_redshift_via_redshift_err_short_time_array,\
            res_delta_redshift_via_redshift_median_err_short_time_array,res_redshift_short_time_array,p20_short_time_array,p20_median_short_time_array,median_g_with_redshift_short_time_array],
                   [res_delta_redshift_via_redshift_long_time_array,res_delta_redshift_via_redshift_median_long_time_array,res_delta_redshift_via_redshift_err_long_time_array,\
            res_delta_redshift_via_redshift_median_err_long_time_array,res_redshift_long_time_array,p20_long_time_array,p20_median_long_time_array,median_g_with_redshift_long_time_array]]
        
def create_p20_values(res_redshift_array_single,res_delta_redshift_via_redshift_array_single,res_delta_redshift_via_redshift_err_array_single,time_result=False,print_res=False):
    """
    
    creates linear fit to the data
    
    Parameters
    ----------
    res_redshift_array_single : array_like
        array of redshift (x-axis)
    res_delta_redshift_via_redshift_array_single : array_like
        array of changes in the mean magnitude (y-axis)        
    res_delta_redshift_via_redshift_err_array_single : array_like   
        array of uncertanties on the data
    -------
    returns 1st order polynomial for the mean, upper and lower limit
    """
    poly_fit_order=1
    
    p20, covar =np.polyfit(res_redshift_array_single,res_delta_redshift_via_redshift_array_single,poly_fit_order,w=1/res_delta_redshift_via_redshift_err_array_single,cov=True)
    p20_main=np.poly1d(p20)
    #print(covar)
    #sigma_p20 = np.sqrt(np.diagonal(covar))
    #print(np.diagonal(covar))
    #print(np.sqrt(np.diagonal(covar)))
    #p20p=p20+sigma_p20
    #p20m=p20-sigma_p20
    
    # bootstraping to determine confidence region
    list_of_p20=[]
    for i in range(100):
        p20=np.polyfit(res_redshift_array_single,res_delta_redshift_via_redshift_array_single+np.random.normal(0,res_delta_redshift_via_redshift_err_array_single),poly_fit_order)
        list_of_p20.append(p20)

    multi_fit_p20=[]   
    for i in range(100):
        multi_fit_p20.append(np.poly1d(list_of_p20[i])(res_redshift_array_single))
    
    multi_fit_p20=np.array(multi_fit_p20) 
    errors_confidence=np.std(multi_fit_p20,axis=0)/2
    

    # determining the redshift evolution of errors
    p_redshift_evolution_of_errors=np.poly1d(np.polyfit(res_redshift_array_single,res_delta_redshift_via_redshift_err_array_single,poly_fit_order))
    errors_prediction=p_redshift_evolution_of_errors(res_redshift_array_single)
    
    
    # fitting in time will work like this
    #(a*xprime + b) /. xprime -> (time/(1 + x))
    # b + (a time)/(1 + x) 
    
    # only works for first order fit
    time_between_surveys=14.85
    
    p20_time, covar_time =np.polyfit(14.85/(1+res_redshift_array_single),res_delta_redshift_via_redshift_array_single,poly_fit_order,w=1/res_delta_redshift_via_redshift_err_array_single,cov=True)
    p20_main_time_result=p20_time[1]+(p20_time[0]*time_between_surveys)/(1+res_redshift_array_single)
    
    if print_res==True:
        print('redshift fit '+str(p20_main))
        print('time fit '+str(np.poly1d(p20_time)))
    
    if time_result==False:
        return p20_main(res_redshift_array_single),p20_main(res_redshift_array_single)+errors_confidence+errors_prediction,p20_main(res_redshift_array_single)-errors_confidence-errors_prediction      
    if time_result==True:
        return p20_main_time_result,p20_main_time_result+errors_confidence+errors_prediction,p20_main_time_result-errors_confidence-errors_prediction

   
    '''
     #p20, covar =np.polyfit(res_redshift_array_single,res_delta_redshift_via_redshift_array_single,poly_fit_order,w=1/res_delta_redshift_via_redshift_err_array_single,cov=True)
 
    
    p20_main=np.poly1d(np.polyfit(res_redshift_array_single,res_delta_redshift_via_redshift_array_single,poly_fit_order,w=1/res_delta_redshift_via_redshift_err_array_single))
    #sigma_p20 = np.sqrt(np.diagonal(covar))
    #p20p=p20+sigma_p20
    #p20m=p20-sigma_p20
    
    list_of_p20=[]
    for i in range(100):
        p20=np.polyfit(res_redshift_array_single,res_delta_redshift_via_redshift_array_single+np.random.normal(0,res_delta_redshift_via_redshift_err_array_single),poly_fit_order)
        list_of_p20.append(p20)

    multi_fit_p20=[]   
    for i in range(100):
        multi_fit_p20.append(np.poly1d(list_of_p20[i])(res_redshift_array_single))
    
    multi_fit_p20=np.array(multi_fit_p20)  

    print(np.std(multi_fit_p20,axis=0))
    
    return p20_main(res_redshift_array_single),p20_main(res_redshift_array_single)+np.std(multi_fit_p20,axis=0)/2,p20_main(res_redshift_array_single)-np.std(multi_fit_p20,axis=0)       
    '''


#### 
# modeling below    


def create_interpolation_for_v_l_a_L_E(v,l,a,L,E,means_all_LC_redshift_fit,means_all_LC_redshift_values,complete_return=False,run=2):

    """return polynomial fit given frequency break, lamda*, lower slope and lower limit of Edd. ratio
    @param 
    """

    #print(run)
    #v_list=np.unique(means_all_LC_redshift_fit[:,0])
    #l_list=np.unique(means_all_LC_redshift_fit[:,1])
    #a_list=np.unique(means_all_LC_redshift_fit[:,2])
    #L_list=np.unique(means_all_LC_redshift_fit[:,3])
    if run==2:
        v_list=np.linspace(-10,-8,6)
        l_list=np.round(np.linspace(-2,0.0,6),2)
        a_list=[8,1.0,1.2,1.4,1.6,1.8]
        L_list=np.linspace(-5,-3,6)
    if run==3:
        
        transformed_v_list=np.log10(means_all_LC_redshift_fit[:,0])
        transformed_l_list=np.log10(means_all_LC_redshift_fit[:,1])
        transformed_a_list=means_all_LC_redshift_fit[:,2]
        transformed_L_list=np.log10(means_all_LC_redshift_fit[:,3])
        
        v_list=np.unique(transformed_v_list)
        #v_list=np.round(np.linspace(-11,-8,7),2)
        l_list=np.unique(transformed_l_list)
        #l_list=np.round(np.linspace(-2,0.0,7),2)
        a_list=np.unique(transformed_a_list)
        #a_list=[0.55,0.7,0.85,1.0,1.15,1.35,1.45]
        L_list=np.unique(transformed_L_list)
        #L_list=np.round(np.linspace(-4.5,-2,7),2)
    

    v_near=find_nearest(v_list,v)
    
    l_near=find_nearest(l_list,l)
    #print('l_near'+str(l))
    #print('l_near'+str(l_near))

    a_near=find_nearest(a_list,a)
    L_near=find_nearest(L_list,L)

    v_spread=np.abs(np.abs(v_near[0])-np.abs(v_near[1]))
    l_spread=np.abs(np.abs(l_near[0])-np.abs(l_near[1]))
    a_spread=np.abs(np.abs(a_near[0])-np.abs(a_near[1]))
    L_spread=np.abs(np.abs(L_near[0])-np.abs(L_near[1]))

    selected_means_near_par=[]
    selected_mean_values_near_par=[]
    distance_par=[]
    list_of_points=[]
    for v_near_i in range(0,2):
        for l_near_i in range(0,2):
            for a_near_i in range(0,2):
                for L_near_i in range(0,2):
                    
                    #print([v_near[v_near_i],l_near[l_near_i],a_near[a_near_i],L_near[L_near_i]])

                    #transformed_v_list=np.round(np.log10(means_all_LC_redshift_fit[:,0]),2)
                    #transformed_l_list=np.round(np.log10(means_all_LC_redshift_fit[:,1]),2)
                    #transformed_a_list=means_all_LC_redshift_fit[:,2]
                    #transformed_L_list=np.round(np.log10(means_all_LC_redshift_fit[:,3]),2)
                    
                    #print(np.unique(transformed_v_list))
                    #print(np.unique(transformed_l_list))
                    #print(np.unique(transformed_a_list))
                    #print(np.unique(transformed_L_list))
                    
                    #print(np.sum(transformed_v_list==v_near[v_near_i]))
                    #print(np.sum(transformed_l_list==l_near[l_near_i]))
                    #print(np.sum(transformed_a_list==a_near[a_near_i]))
                    #print(np.sum(transformed_L_list==L_near[L_near_i]))

                    
                    selected_means_single_par=means_all_LC_redshift_fit[(transformed_v_list==v_near[v_near_i])&\
                                                                        (transformed_l_list==l_near[l_near_i])&\
                                                                        (transformed_a_list==a_near[a_near_i])&\
                                                                        (transformed_L_list==L_near[L_near_i])]                    
                    
                    
                    #selected_means_single_par=means_all_LC_redshift_fit[(means_all_LC_redshift_fit[:,0]==v_near[v_near_i])&\
                    #                                                    (means_all_LC_redshift_fit[:,1]==l_near[l_near_i])&\
                    #                                                    (means_all_LC_redshift_fit[:,2]==a_near[a_near_i])&\
                    #                                                    (means_all_LC_redshift_fit[:,3]==L_near[L_near_i])]
                                        
                    selected_mean_values_single_par=means_all_LC_redshift_values[(transformed_v_list==v_near[v_near_i])&\
                                                                                 (transformed_l_list==l_near[l_near_i])&\
                                                                                 (transformed_a_list==a_near[a_near_i])&\
                                                                                 (transformed_L_list==L_near[L_near_i])]
                    
                    #selected_mean_values_single_par=means_all_LC_redshift_values[(means_all_LC_redshift_fit[:,0]==v_near[v_near_i])&\
                    #                                                             (means_all_LC_redshift_fit[:,1]==l_near[l_near_i])&\
                    #                                                             (means_all_LC_redshift_fit[:,2]==a_near[a_near_i])&\
                    #                                                             (means_all_LC_redshift_fit[:,3]==L_near[L_near_i])]
                                       
                    #print(selected_mean_values_single_par)
                    
                    E_list_single=selected_means_single_par[:,4]
                    # all the Eddington avaliable from this set of parameters
                    #print('E_list_single'+str(E_list_single))
                    #print(E)
                    E_near=find_nearest(E_list_single,E)
                    E_spread=np.abs(np.abs(E_near[0])-np.abs(E_near[1]))
                    #print(E_near)
                    for E_near_i in range(0,2):
                        selected_means_single_par_single_E=selected_means_single_par[selected_means_single_par[:,4]==E_near[E_near_i]]
                        selected_mean_values_single_par_single_E=selected_mean_values_single_par[selected_mean_values_single_par[:,4]==E_near[E_near_i]]
                        
                        
                        selected_means_near_par.append(selected_means_single_par_single_E)
                        selected_mean_values_near_par.append(selected_mean_values_single_par_single_E[0])
                        
                        #distance=( (np.abs(v-v_near[v_near_i])/v_spread) +  (np.abs(l-l_near[l_near_i])/l_spread) +\
                        #                    (np.abs(a-a_near[a_near_i])/a_spread) + (np.abs(L-L_near[L_near_i])/L_spread) + \
                        #          (np.abs(E-E_near[E_near_i])/E_spread) )/((5))   
                        #distance_par.append( distance     )



                    
                        #z=np.array([selected_means_single_par_single_E[0][-3],selected_means_single_par_single_E[0][-2],selected_means_single_par_single_E[0][-1]])
                        #p = np.poly1d(z)
                        #points=p(selection_of_times_as_redshift_in_HSC_SDSS)
                        #list_of_points.append(points[(selection_of_times_as_redshift_in_HSC_SDSS>0)&(selection_of_times_as_redshift_in_HSC_SDSS<4)])
    
    
    selected_mean_values_near_par=np.array(selected_mean_values_near_par)

    #print('np.log10(selected_mean_values_near_par[:,1])'+str(np.log10(selected_mean_values_near_par[:,1])))
    # Create coordinate pairs
    cartcoord = list(zip(np.log10(selected_mean_values_near_par[:,0]),\
                         np.log10(selected_mean_values_near_par[:,1]),\
                         selected_mean_values_near_par[:,2],\
                        np.log10(selected_mean_values_near_par[:,3]),\
                         selected_mean_values_near_par[:,4]))

    values_0=selected_mean_values_near_par[:,5]
    values_1=selected_mean_values_near_par[:,6]
    values_2=selected_mean_values_near_par[:,7]
    values_3=selected_mean_values_near_par[:,8]
    values_4=selected_mean_values_near_par[:,9]
    values_5=selected_mean_values_near_par[:,10]
    values_6=selected_mean_values_near_par[:,11]

    #print(cartcoord)
    #print(values_0)
    #print(len(cartcoord))
    #print(len(values_0))
    
    # Approach 1
    interp_0 = scipy.interpolate.LinearNDInterpolator(cartcoord, values_0, fill_value=-99)
    interp_1 = scipy.interpolate.LinearNDInterpolator(cartcoord, values_1, fill_value=0)
    interp_2 = scipy.interpolate.LinearNDInterpolator(cartcoord, values_2, fill_value=0)
    interp_3 = scipy.interpolate.LinearNDInterpolator(cartcoord, values_3, fill_value=0)
    interp_4 = scipy.interpolate.LinearNDInterpolator(cartcoord, values_4, fill_value=0)
    interp_5 = scipy.interpolate.LinearNDInterpolator(cartcoord, values_5, fill_value=0)
    interp_6 = scipy.interpolate.LinearNDInterpolator(cartcoord, values_6, fill_value=0)

    
    interpolated_scipy_points=np.array([interp_0(v,l,a,L,E),interp_1(v,l,a,L,E),interp_2(v,l,a,L,E),interp_3(v,l,a,L,E),interp_4(v,l,a,L,E),interp_5(v,l,a,L,E),interp_6(v,l,a,L,E)])
    
    #weights_par=1/np.array(distance_par)**2
    #weights_par=weights_par/np.sum(weights_par)
    #weights_par[np.isnan(weights_par)] = 1
    #print(weights_par)
    #array_of_points=np.array(list_of_points)
    #weights_par=np.array(weights_par)

    #interpolated_points=[]
    #for i in range(array_of_points.shape[1]):
    #    interpolated_points.append(np.mean(array_of_points[:,i]*weights_par)*(2**5))
    
    redshift_points=np.array([3.51996875, 2.615975  , 1.85471711, 1.169585  , 0.74966532,
       0.39075962, 0.0847925 ])

    #print('interpolated_scipy_points '+str(interpolated_scipy_points))
    z_interpolated=np.polyfit(redshift_points,interpolated_scipy_points,2)
    p_interpolated=np.poly1d(z_interpolated)
    
    if complete_return is False:
        return p_interpolated
    if complete_return is True:
        return p_interpolated,array_of_points,weights_par,interpolated_points,np.array(selected_mean_values_near_par)
     
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
    array_2=np.delete(array,idx)
    idx2 = (np.abs(array_2 - value)).argmin()
    return array[idx],array_2[idx2]

#def l_as_fun_z(z):
#    return -1.85+2.5*np.log10(1+z)

def l_as_fun_z(z):
    if z<2:
        return -1.85+2.0*np.log10(1+z)
    if z>=2:
        return -1.85+2.0*np.log10(1+2)