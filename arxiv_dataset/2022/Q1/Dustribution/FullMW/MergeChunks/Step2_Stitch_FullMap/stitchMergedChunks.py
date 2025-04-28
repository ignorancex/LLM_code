import numpy as np
import pandas as pd
import os
import subprocess
from glob import glob
import time
from memory_profiler import profile
from numba import njit

from stitchFileList import dirnames
from plotStructure_FullMW import plot_GP_Pred_Dens_Slices_AlongDist, plot_GP_Pred_Ext_Slices_AlongDist, plot_GP_Pred_ExtCumilative, plot_GP_Pred_ExtCumilative_Gal0inCent_Hammer


def load_mergeChunks(direc):

    l_bounds_filename = glob(direc + "/" + "*_merge_l_bounds.pkl.npy")[0]
    merge_l_bounds = np.load(l_bounds_filename, allow_pickle=True)

    b_bounds_filename = glob(direc + "/" + "*_merge_b_bounds.pkl.npy")[0]
    merge_b_bounds = np.load(b_bounds_filename, allow_pickle=True)

    d_bounds_filename = glob(direc + "/" + "*_merge_d_bounds.pkl.npy")[0]
    merge_d_bounds = np.load(d_bounds_filename, allow_pickle=True)

    # print("driec ==", direc)
    # print("merge_l_bounds =", merge_l_bounds)
    # print("merge_b_bounds =", merge_b_bounds)
    # print("merge_d_bounds =", merge_d_bounds)

    dens_Weighted_Median_filename = glob(direc + "/" + "*_MWchunk_Dens_Weighted_Median.pkl.npy")[0]
    dens_Weighted_Median = np.load(dens_Weighted_Median_filename, allow_pickle=True)

    dens_16P_filename = glob(direc + "/" + "*_MWchunk_Dens_16P.pkl.npy")[0]
    dens_16P = np.load(dens_16P_filename, allow_pickle=True)

    dens_84P_filename = glob(direc + "/" + "*_MWchunk_Dens_84P.pkl.npy")[0]
    dens_84P = np.load(dens_84P_filename, allow_pickle=True)

    ext_Weighted_Median_filename = glob(direc + "/" + "*_MWchunk_CumExt_Weighted_Median.pkl.npy")[0]
    ext_Weighted_Median = np.load(ext_Weighted_Median_filename, allow_pickle=True)

    ext_16P_filename = glob(direc + "/" + "*_MWchunk_CumExt_16P.pkl.npy")[0]
    ext_16P = np.load(ext_16P_filename, allow_pickle=True)

    ext_84P_filename = glob(direc + "/" + "*_MWchunk_CumExt_84P.pkl.npy")[0]
    ext_84P = np.load(ext_84P_filename, allow_pickle=True)

    return merge_l_bounds, merge_b_bounds, merge_d_bounds, dens_Weighted_Median, dens_16P, dens_84P, ext_Weighted_Median, ext_16P, ext_84P

@profile
def uniquetol(A, tol = 0.001):
    """ Find the unique subset of values in an array within some tolerance""" 
    return A[~(np.triu(np.abs(A[:,None] - A) <= tol,1)).any(0)]

@njit
def filter_extinction_dips(ext_Weighted_Median):
    """ filter out artefacts in extinction-distance curves

    """

    for i in range(ext_Weighted_Median.shape[0]): #iterate over l
        for j in range(ext_Weighted_Median.shape[1]):#iterate over b
            ext_d_last = ext_Weighted_Median[i,j,0]
            for k in range(1,ext_Weighted_Median.shape[2]): #now the real work
                if ext_Weighted_Median[i,j,k] < ext_d_last:
                    ext_Weighted_Median[i,j,k] = ext_d_last
                else:
                    ext_d_last = ext_Weighted_Median[i,j,k]
    return ext_Weighted_Median



@profile
def merge_FullMW():

    #First we read in all the bounds to figure out what our full set of bounds will be:
    full_lbounds = np.array([])
    full_bbounds = np.array([])
    full_dbounds = np.array([]) 

    tol = 0

    for direc in dirnames:

        print("direc ==", direc)

        #Check if the LBol Chunk file exists because it was cut in a previous run
        direc_exists = os.path.isdir(direc)

        merge_l_bounds, merge_b_bounds, merge_d_bounds, dens_Weighted_Median, dens_16P, dens_84P, ext_Weighted_Median, ext_16P, ext_84P = load_mergeChunks(direc)

        full_lbounds = np.append(full_lbounds, merge_l_bounds)
        full_bbounds = np.append(full_bbounds, merge_b_bounds)
        full_dbounds = np.append(full_dbounds, merge_d_bounds)

        if tol == 0:

            #Now we can find the unique set of cell boundaries from the entire list
            #The b boundaries seem to suffer from a machine-precision issue, so merging them with unique produces duplicate entries
            #We are therefore going to merge them within a tolerance which is 1% of the step size between the first two values
            tol = (full_bbounds[1] - full_bbounds[0])* 0.01 #
            dtol = (full_dbounds[1] - full_dbounds[0]) * 0.01
    
        full_lbounds = np.sort(uniquetol(full_lbounds, tol = tol))#np.unique(full_lbounds) #It appears that the lbounds *also* suffer from this machine-precision issue
        full_bbounds = np.sort(uniquetol(full_bbounds, tol = tol))#np.unique(full_bbounds)
        full_dbounds = np.sort(uniquetol(full_dbounds, tol = dtol))
    print("full_l_bounds", full_lbounds)
    print("full_b_bounds", full_bbounds)
    print("full_d_bounds", full_dbounds)


    #Size of the array to hold the merge data
    size_full_lbounds = len(full_lbounds)-1
    size_full_bbounds = len(full_bbounds)-1
    size_full_dbounds = len(full_dbounds)-1

    #Save full MW LBD grids
    np.save("FullMW_merged_l_bounds.pkl",full_lbounds, allow_pickle=True)
    np.save("FullMW_merged_b_bounds.pkl",full_bbounds, allow_pickle=True)
    np.save("FullMW_merged_d_bounds.pkl",full_dbounds, allow_pickle=True)


    #Mergeing extinction and density chunks together to make full MW maps
    print("Creating arrays for stitched map:")
    print("Median ext")
    ext_Weighted_Median_full = np.full((size_full_lbounds, size_full_bbounds, size_full_dbounds+1), np.nan)
    print("16P ext")
    ext_16P_full = np.full((size_full_lbounds, size_full_bbounds, size_full_dbounds+1), np.nan)
    print("84P ext")
    ext_84P_full = np.full((size_full_lbounds, size_full_bbounds, size_full_dbounds+1), np.nan)

    print("Median dense")
    dens_Weighted_Median_full = np.full((size_full_lbounds, size_full_bbounds, size_full_dbounds+1), np.nan)
    print("16P dense")
    dens_16P_full = np.full((size_full_lbounds, size_full_bbounds, size_full_dbounds+1), np.nan)
    print("84P dense")
    dens_84P_full = np.full((size_full_lbounds, size_full_bbounds, size_full_dbounds+1), np.nan)

    print("Starting to loop over all chunks and stitch them together")

    for direc in dirnames:
        #Check if the LBol Chunk file exists because it was cut in a previous run
        direc_exists = os.path.isdir(direc)
        print("direc ==", direc)

        merge_l_bounds, merge_b_bounds, merge_d_bounds, dens_Weighted_Median, dens_16P, dens_84P, ext_Weighted_Median, ext_16P, ext_84P = load_mergeChunks(direc)
        ext_Weighted_Median = filter_extinction_dips(ext_Weighted_Median)
        ext_16P = filter_extinction_dips(ext_16P)
        ext_84P = filter_extinction_dips(ext_84P)

        n_l = len(merge_l_bounds) - 1
        if n_l > ext_Weighted_Median.shape[0]:
            n_l = ext_Weighted_Median.shape[0]
        n_b = len(merge_b_bounds) - 1
        n_d = len(merge_d_bounds) - 1

        #now we figure out where this chunk fits in to the full bounds:
        l_start = np.where(np.abs(full_lbounds - merge_l_bounds[0]) < tol)[0][0]
        b_start = np.where(np.abs(full_bbounds - merge_b_bounds[0]) < tol)[0][0]
        d_start = np.where(np.abs(full_dbounds - merge_d_bounds[0]) < dtol )[0][0]

        ext_Weighted_Median_full[l_start:l_start+n_l, b_start:b_start+n_b, d_start:d_start+n_d+1] = ext_Weighted_Median
        ext_16P_full[l_start:l_start+n_l, b_start:b_start+n_b, d_start:d_start+n_d+1] = ext_16P
        ext_84P_full[l_start:l_start+n_l, b_start:b_start+n_b, d_start:d_start+n_d+1] = ext_84P
        
        dens_Weighted_Median_full[l_start:l_start+n_l, b_start:b_start+n_b, d_start:d_start+n_d+1] = dens_Weighted_Median
        dens_16P_full[l_start:l_start+n_l, b_start:b_start+n_b, d_start:d_start+n_d+1] = dens_16P
        dens_84P_full[l_start:l_start+n_l, b_start:b_start+n_b, d_start:d_start+n_d+1] = dens_84P

    print("Saving output files")
    #Save full MW cumilative extinction cube
    np.save("FullMW_CumExt_Weighted_Median.pkl", ext_Weighted_Median_full, allow_pickle=True) 
    np.save("FullMW_CumExt_16P.pkl", ext_16P_full, allow_pickle=True)
    np.save("FullMW_CumExt_84P.pkl", ext_84P_full, allow_pickle=True)

    #Save full MW Density cube
    np.save("FullMW_Dens_Weighted_Median.pkl", dens_Weighted_Median_full, allow_pickle=True) 
    np.save("FullMW_Dens_16P.pkl", dens_16P_full, allow_pickle=True)
    np.save("FullMW_Dens_84P.pkl", dens_84P_full, allow_pickle=True)



    #Plot extinction
    print("Begin plotting extinction")
    plot_GP_Pred_ExtCumilative(full_lbounds, full_bbounds, ext_Weighted_Median_full)
    plot_GP_Pred_ExtCumilative_Gal0inCent_Hammer(full_lbounds, full_bbounds, ext_Weighted_Median_full)
    plot_GP_Pred_Ext_Slices_AlongDist(full_lbounds, full_bbounds, full_dbounds, ext_Weighted_Median_full) 
    print("End plotting extinction")

    #Plot density
    print("Begin plotting Density")
    #Plot selected slices along distance of predicted extinction and density 
    plot_GP_Pred_Dens_Slices_AlongDist(size_full_lbounds, size_full_bbounds, size_full_dbounds, full_lbounds, full_bbounds, full_dbounds, dens_Weighted_Median_full)




if __name__=="__main__":

    start_time = time.time()
    print("code run start time = ", start_time)

    merge_FullMW()
    

    print("Code Run Time --- %s seconds ---" % (time.time() - start_time))















