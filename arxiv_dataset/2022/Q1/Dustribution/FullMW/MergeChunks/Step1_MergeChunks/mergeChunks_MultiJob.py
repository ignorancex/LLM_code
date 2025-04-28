import numpy as np
import torch
import time
import xarray as xr
from xarray.core import weighted
from memory_profiler import profile
import gc
import argparse
import os

from mergeFileList import dirnames
from plotStructure import plot_GP_Pred_Dens_Slices_AlongDist, plot_GP_Pred_Ext_Slices_AlongDist, plot_GP_Pred_ExtCumilative 
#from plotStructure import plot_GP_Pred_cartXY_ExtCumilative, plot_GP_Pred_cartXY_Density

chunk_boundaries_lower = [10, 600, 1200, 1800] #] #10, 600, 1200, 1800
chunk_boundaries_upper = [1000, 1590, 2190, 2790] #, ] #1000, 1590, 2190, 2790

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def load_coords(direc):
    l_bounds_pred = np.load(direc+"/l_bounds_pred.pkl.npy", allow_pickle=True)
    print("l_bounds_pred ==", l_bounds_pred)
    b_bounds_pred = np.load(direc+"/b_bounds_pred.pkl.npy", allow_pickle=True)
    #print("b_bounds_pred ==", b_bounds_pred)
    d_bounds_pred = np.load(direc+"/d_bounds_pred.pkl.npy", allow_pickle=True)
    print("d_bounds_pred ==", d_bounds_pred)

    return l_bounds_pred, b_bounds_pred, d_bounds_pred


def load_samples(filename, n_l, n_b, n_d):
    if filename[-3:] == "npy":
        try:
            samples = np.reshape(np.load(filename, allow_pickle=True), (1000, n_l, n_b, n_d+1))
        except ValueError: #If we get a ValueError, it's because there wasn't enough entries in the file to reshape with 1000 for the samples direction, so it must be a case with only 200 samples in it
            samples = np.reshape(np.load(filename, allow_pickle=True), (200, n_l, n_b, n_d+1))
    else:
        try:
            samples = torch.load(filename, map_location=torch.device('cpu')).numpy().reshape((1000, n_l, n_b, n_d))
        except (RuntimeError, ValueError):
            samples = torch.load(filename, map_location=torch.device('cpu')).numpy().reshape((200, n_l, n_b, n_d))
    return samples


def uniquetol(A, tol = 0.001):
    """ Find the unique subset of values in an array within some tolerance""" 
    return A[~(np.triu(np.abs(A[:,None] - A) <= tol,1)).any(0)]


@profile
def full_MergeFunc():

    #This script deals with merging one red square by gathering all the purple squares that overlap with it.

    #Another script is required to define the l,b ranges of the red squares to tessalte the whole sky

    #another script will be required to read in all the merged products of the red squares and join them along their edges for the final map

    start_time = time.time()
    print("code run start time = ", start_time)

    #What are the l, b edges of the region we are going to merge?
    #Check commnad line arguments and update file with values such as filename, lmin/max, etc. 
    parser = argparse.ArgumentParser(description="wieghted median merge each set of fully overlapping areas")
    parser.add_argument("merge_l_min", metavar="merge_l_min_val", type=float, help="l min in deg")
    parser.add_argument("merge_l_max", metavar="merge_l_max_val", type=float, help="l max in deg")
    parser.add_argument("merge_b_min", metavar="merge_b_min_val", type=float, help="b min in deg")
    parser.add_argument("merge_b_max", metavar="merge_b_max_val", type=float, help="b max in deg")
    parser.add_argument("--min_d_bounds_pred_Dchunk", metavar="min_d_bounds_pred_Dchunk_val", action="extend", nargs="+", type=float, help="min d bounds chunk")
    parser.add_argument("--dweight_lower_cutoff", metavar="dweight_lower_cutoff_val", action="extend", nargs="+", type=float, help="d weight lower cutoff")
    parser.add_argument("--dweight_upper_cutoff", metavar="dweight_upper_cutoff_val", action="extend", nargs="+", type=float, help="d weight upper cutoff")
    

    args = parser.parse_args()
    print("Input details == ", args)
    merge_l_min = args.merge_l_min
    merge_l_max = args.merge_l_max
    merge_b_min = args.merge_b_min
    merge_b_max = args.merge_b_max

    min_d_bounds_pred_Dchunk = args.min_d_bounds_pred_Dchunk
    dweight_lower_cutoff = args.dweight_lower_cutoff
    dweight_upper_cutoff = args.dweight_upper_cutoff

    merge_coords = str(merge_l_min)+"_"+str(merge_l_max)+ "_" + str(merge_b_min)+"_"+str(merge_b_max)
    merge_coords_print = str(merge_l_min) + "_" +str(merge_l_max) + "_" +str("%.5f" % merge_b_min) + "_" +str("%.5f" % merge_b_max)

    #Number of GP samples to take from extinction or density all samples cube
    n_samples_per_map = 100
    n_overlap = 16 #8 #2 #4 #Number of overlapping chunks at any point + padding

    #Chunks which contain the smaller area to merge which are even slightly overlapping to create the small merge area
    overlapping_chunks = []

    #Division for the sigmoid to rescale it to a useful range
    sig_div = 20

    #Find the larger chunks in all distance folders which will in someway contribute to our smaller fully overlapped chunk
    for direc in dirnames:
        #Check if the LBol Chunk file exists because it was cut in a previous run
        direc_exists = os.path.isdir(direc)
        if direc_exists:
            direc_lmin, direc_lmax, direc_bmin, direc_bmax = direc.split("/")[-1].split("_") #direc.strip("../").strip(str(folder)).strip("/").split("_")
            if (args.merge_l_min-0.001 <= float(direc_lmin) < args.merge_l_max+0.001) or (args.merge_l_min-0.001 < float(direc_lmax) <= args.merge_l_max+0.001):
                if (args.merge_b_min-0.001 <= float(direc_bmin) < args.merge_b_max+0.001) or (args.merge_b_min-0.001 < float(direc_bmax) <= args.merge_b_max+0.001):

                    overlapping_chunks.append(direc)
                    print("branch 1.1: ", direc)
                elif (float(direc_bmin) < args.merge_b_min) and (float(direc_bmax) > args.merge_b_max):
                    overlapping_chunks.append(direc)
                    print("branch 1.2: ", direc)
            elif (float(direc_lmin) > float(direc_lmax)) and ((args.merge_l_min+0.001 >= float(direc_lmin) > args.merge_l_max-0.001) or (args.merge_l_min+0.001 > float(direc_lmax) >= args.merge_l_max-0.001)):
                if (args.merge_b_min-0.001 <= float(direc_bmin) < args.merge_b_max+0.001) or (args.merge_b_min-0.001 < float(direc_bmax) <= args.merge_b_max+0.001):
                    overlapping_chunks.append(direc)
                    print("branch 2.1: ", direc)
                elif (float(direc_bmin) < args.merge_b_min) and (float(direc_bmax) > args.merge_b_max):
                    overlapping_chunks.append(direc)
                    print("branch 2.2: ", direc)
            elif (float(direc_lmin) < args.merge_l_min) and (float(direc_lmax) > args.merge_l_max): #what happens if the chunk we're reading is larger than the chunk we're merging into?
                if (args.merge_b_min-0.001 <= float(direc_bmin) < args.merge_b_max+0.001) or (args.merge_b_min-0.001 < float(direc_bmax) <= args.merge_b_max+0.001):
                    overlapping_chunks.append(direc)
                    print("branch 3.1: ", direc)
                elif (float(direc_bmin) < args.merge_b_min) and (float(direc_bmax) > args.merge_b_max):
                    overlapping_chunks.append(direc)
                    print("branch 3.2: ", direc)
            elif (float(direc_lmin) > float(direc_lmax)) and ((float(direc_lmin) < args.merge_l_min) or (float(direc_lmax) > args.merge_l_max)): 
                if (args.merge_b_min-0.001 <= float(direc_bmin) < args.merge_b_max+0.001) or (args.merge_b_min-0.001 < float(direc_bmax) <= args.merge_b_max+0.001):
                    overlapping_chunks.append(direc)
                    print("branch 4.1: ", direc, float(direc_lmin), float(direc_lmax))
                elif (float(direc_bmin) < args.merge_b_min) and (float(direc_bmax) > args.merge_b_max):
                    overlapping_chunks.append(direc)
                    print("branch 4.2: ", direc, direc_lmin, direc_lmax)

    n_overlapping_chunks = len(overlapping_chunks)
    #First we need to know which files we have to read from. 
    print("overlapping chunk names ==", overlapping_chunks) #File list given in mergeFileList.py and imported as a variable in this file
    print("number of overlapping_chunks ==", n_overlapping_chunks)

    #exit()

    #First we read in all the bounds to figure out what our full set of bounds will be:
    full_lbounds = np.array([])
    full_bbounds = np.array([])
    full_dbounds = np.array([]) #Only necessary if we start chunking along d as well
    
    #Load all the lbd bounds from all the overlapping lbd chunks to make the grids for merging 
    for i, direc in enumerate(overlapping_chunks):
        l_bounds_pred, b_bounds_pred, d_bounds_pred = load_coords(direc)
        #First, the correct way once we know that all models are run on subsets of the same coordinate grids:
        full_lbounds = np.append(full_lbounds, l_bounds_pred)
        full_bbounds = np.append(full_bbounds, b_bounds_pred)
        full_dbounds = np.append(full_dbounds, d_bounds_pred)
    
    #Now we can find the unique set of cell boundaries from the entire list
    #The b boundaries seem to suffer from a machine-precision issue, so merging them with unique produces duplicate entries
    #We are therefore going to merge them within a tolerance which is 1% of the step size between the first two values
    tol = (full_bbounds[1] - full_bbounds[0])* 0.01 #
    dtol = (full_dbounds[1] - full_dbounds[0]) * 0.01
    
    full_lbounds = np.sort(uniquetol(full_lbounds, tol = tol))#np.unique(full_lbounds) #It appears that the lbounds *also* suffer from this machine-precision issue
    full_bbounds = np.sort(uniquetol(full_bbounds, tol = tol))#np.unique(full_bbounds)
    full_dbounds = np.sort(uniquetol(full_dbounds, tol = dtol))
    print("allOverlap_l_bounds", full_lbounds)
    print("allOverlap_b_bounds", full_bbounds)
    print("allOverlap_d_bounds", full_dbounds)



    #Cut out only lbd coords which are within the small regions which is completely overlapped and to be merged
    merge_lbounds = full_lbounds[(((args.merge_l_min-1e-5 <= full_lbounds) & (full_lbounds <= args.merge_l_max+1e-5)) |
                                  ((args.merge_l_min-1e-5 <= full_lbounds+360) & (full_lbounds+360 <= args.merge_l_max+1e-5))
    )]  #1e-5/+5 added to account for numerical prescision offsets
    if np.any(merge_lbounds < 0) and args.merge_l_min > 0:
        merge_lbounds[merge_lbounds <= 0] = merge_lbounds[merge_lbounds <= 0 ] + 360
    merge_lbounds = np.sort(uniquetol(merge_lbounds, tol = tol))
    merge_bbounds = full_bbounds[((args.merge_b_min-1e-5 <= full_bbounds) & (full_bbounds <= args.merge_b_max+1e-5))] 
    merge_dbounds = full_dbounds #We want all the distances for any given fully overlapped chunk. So we always take all the distances into account

    print("Merge_lbounds: ", merge_lbounds)
    print("Merge_bbounds: ", merge_bbounds)
    print("Merge_dbounds: ", merge_dbounds)

    #Size of the array to hold the merge data
    size_l_merge = len(merge_lbounds)-1
    size_b_merge = len(merge_bbounds)-1
    size_d_merge = len(merge_dbounds)-1


    #Save full MW LBD grids
    np.save(merge_coords_print+"_merge_l_bounds.pkl",merge_lbounds, allow_pickle=True)
    np.save(merge_coords_print+"_merge_b_bounds.pkl",merge_bbounds, allow_pickle=True)
    np.save(merge_coords_print+"_merge_d_bounds.pkl",merge_dbounds, allow_pickle=True)


    #Extinction merging
    print("Begin Extinction Merging")
    merged_ext_samples = np.full((n_samples_per_map * n_overlap, size_l_merge, size_b_merge, size_d_merge+1), np.nan)
    weights = np.zeros((n_samples_per_map * n_overlap, size_l_merge, size_b_merge, size_d_merge+1))

    #We also create a smaller array to record how many models cover each point so we can cut out regions later
    cov = np.zeros((size_l_merge, size_b_merge, size_d_merge+1), dtype=int)
    fillmask = np.zeros((n_overlap, size_l_merge, size_b_merge, size_d_merge+1), dtype=bool)

    #Now we return to iterating over all the results, this time to read in the results and put a sample into the right place in the big array allocated above
    for i, direc in enumerate(overlapping_chunks):
        print(direc)
        #We have to read in the cell boundaries again
        l_bounds_pred, b_bounds_pred, d_bounds_pred = load_coords(direc)
        n_l = len(l_bounds_pred) - 1
        n_b = len(b_bounds_pred) - 1
        n_d = len(d_bounds_pred) - 1

        if np.abs(l_bounds_pred[-1] - merge_lbounds[0]) <= tol:
            continue #Since the upper bound of the chunk is equal to the lower bound of the area to be merged, there isn't actually any overlap and we should skip this chunk
        if np.abs(l_bounds_pred[0] - merge_lbounds[-1]) <= tol or np.abs(l_bounds_pred[0]+360 - merge_lbounds[-1]) <= tol:
            continue #The opposite of the above case - the lower bound of the chunk is equla to the upper bound of the area to be merged.
        if np.abs(b_bounds_pred[-1] - merge_bbounds[0]) <= tol:
            continue #Since the upper bound of the chunk is equal to the lower bound of the area to be merged, there isn't actually any overlap and we should skip this chunk
        if np.abs(b_bounds_pred[0] - merge_bbounds[-1]) <= tol:
            continue #The opposite of the above case - the lower bound of the chunk is equla to the upper bound of the area to be merged.
        if np.abs(d_bounds_pred[-1] - merge_dbounds[0]) <= dtol:
            continue #Since the upper bound of the chunk is equal to the lower bound of the area to be merged, there isn't actually any overlap and we should skip this chunk
        if np.abs(d_bounds_pred[0] - merge_dbounds[-1]) <= dtol:
            continue #The opposite of the above case - the lower bound of the chunk is equla to the upper bound of the area to be merged.

        if l_bounds_pred[0] == 0.0 and l_bounds_pred[-1] == 360.0:
            weights_local = np.ones((l_bounds_pred[:-1].shape[0], b_bounds_pred[:-1].shape[0]))
        else:
            #let's set up a triangle function for the weights to start us off
            l_grid, b_grid = np.meshgrid((l_bounds_pred[:-1] + l_bounds_pred[1:])/2, (b_bounds_pred[:-1] + b_bounds_pred[1:])/2, indexing='ij')
            ## l_grid, b_grid, d_grid = np.meshgrid((l_bounds_pred[:-1] + l_bounds_pred[1:])/2, (b_bounds_pred[:-1] + b_bounds_pred[1:])/2, (d_bounds_pred[:-1] + d_bounds_pred[1:])/2, indexing='ij') #This verion shouldn't be required because the weights along (l,b) and d are independent - so we can just multiply by broadcasting along the different dimensions. This gives a *massive* RAM saving
            r = (l_grid - np.average(l_bounds_pred))**2 + (b_grid - np.average(b_bounds_pred))**2
            #weights_local = (1 - (r / np.max(r)))**2
            weights_local = ( 1 - (0.9999* ( r / np.max(r) ) ) )**2 #multiplying by 0.9999 ensures that there are no zeros and the smallest weight is 1e-4. This avoids nans in the final map
            #now we add a weighting function along distance
            #the initial weighting function is a three-part function - it is linear from 1e-4 to 1 in the first part of the chunk, the is one, then linearly falls from 1 to 1e-4 at the end of the chunk

        if d_bounds_pred[-1] < 600.:
            lower_sigmoid_lims = (0, d_bounds_pred.min())
            upper_sigmoid_lims = (d_bounds_pred.max(), d_bounds_pred.min())
            if l_bounds_pred[0] == 0.0 and l_bounds_pred[-1] == 360.0: #This case is the solar neighbourhood with the 100 pc scale length
                lower_cutoff = 0
                upper_cutoff = 200
            else: #So this is the Solar Neighbourhood but with the 10 pc scale length
                lower_cutoff = 100
                upper_cutoff = 200
        else:
            try:
                dist_chunk_num = np.argwhere(np.min(d_bounds_pred) < min_d_bounds_pred_Dchunk)[0][0]
            except IndexError:
                print("The minimum distance for this chunk is larger than any of the values in min_d_bounds_pred_Dchunk. \nPlease check that the input is correct.")
                print("Chunk ==", direc)
                print("Minimum distance ==", np.min(d_bounds_pred))
                print("input min_d_bounds_pred_Dchunk ==", min_d_bounds_pred_Dchunk)
                raise IndexError
            try:
                lower_cutoff = dweight_lower_cutoff[dist_chunk_num]
            except IndexError:
                print("The number of entries in min_d_bounds_pred_Dchunk doesn't match up with the number of lower cutoffs provided. \nPlease check that the input is correct")
                raise IndexError
            try:
                upper_cutoff = dweight_upper_cutoff[dist_chunk_num]
            except IndexError:
                print("The number of entries in min_d_bounds_pred_Dchunk doesn't match up with the number of upper cutoffs provided. \nPlease check that the input is correct")
                raise IndexError
            try:
                lower_sigmoid_lims = (chunk_boundaries_upper[dist_chunk_num-1], chunk_boundaries_lower[dist_chunk_num])
            except IndexError:
                lower_sigmoid_lims = (0, chunk_boundaries_lower[dist_chunk_num])
            except NameError:
                lower_sigmoid_lims = (0, chunk_boundaries_lower[0])

            if lower_sigmoid_lims[0] > d_bounds_pred.max() + 100:
                lower_sigmoid_lims = (0, chunk_boundaries_lower[dist_chunk_num])

            try:
                upper_sigmoid_lims = (chunk_boundaries_upper[dist_chunk_num], chunk_boundaries_lower[dist_chunk_num+1])
            except IndexError:
                upper_sigmoid_lims = (chunk_boundaries_upper[dist_chunk_num], np.max(d_bounds_pred))
            except NameError:
                upper_sigmoid_lims = (chunk_boundaries_upper[0], np.max(d_bounds_pred))

        # decay_constant = 0.1 #10 #modify this to change how quickly the weights decay in the exponential decay case!
        # mask_lower = d_bounds_pred < lower_cutoff
        # mask_upper = d_bounds_pred > upper_cutoff
        # mask_middle = np.logical_not(np.logical_or(mask_lower, mask_upper))
        mask_lower = d_bounds_pred < lower_sigmoid_lims[0] #lower_cutoff
        mask_upper = d_bounds_pred > upper_sigmoid_lims[1] # upper_cutoff
        mask_middle = np.logical_not(np.logical_or(mask_lower, mask_upper))
        if d_bounds_pred[-1] < 600.:
            weights_d = np.piecewise(d_bounds_pred, 
                                    [mask_lower, mask_middle, mask_upper], 
                                    [lambda x: 1e-4 + sigmoid((x - 
                                                            (
                                                                (lower_sigmoid_lims[0] + 
                                                                lower_sigmoid_lims[1])
                                                                /2
                                                            ))
                                                            /sig_div
                                                            ), #0.9999*((x-x.min())/(lower_cutoff-x.min())), 
                                     #lambda x: 1e-4 + 0.9999*((x-x.min())/(lower_cutoff-x.min())), 
                                     lambda x: 1, 
                                     #Three different cut off methods of distance weighting below. We only pick one at a time!
                                     lambda x: (1- sigmoid((x - ((upper_sigmoid_lims[0] + upper_sigmoid_lims[1])/2) )/sig_div)) #sigmoid decay
                                     #lambda x: (1- 0.9999*((x-upper_cutoff)/(np.max(d_bounds_pred) - upper_cutoff))) #symetric linear decay
                                     #lambda x: 1e-4 + 0.9999*np.exp(-decay_constant*(x - upper_cutoff))  #Exponential decay - N = N0 e**(-alpha d). In our case N0 is 0.9999, because we want to decay from one to 1e-4
                                     #lambda x: 1e-4 #sharp cutoff
                                     ]
                                    )
        else:
            mask_under_95 = d_bounds_pred < 95.
            weights_d = np.piecewise(d_bounds_pred, 
                                    [mask_under_95,mask_lower, mask_middle, mask_upper], 
                                    [lambda x: 1e-2,
                                     lambda x: 1e-4 + sigmoid((x - ((lower_sigmoid_lims[0] + lower_sigmoid_lims[1])/2))/sig_div), #0.9999*((x-x.min())/(lower_cutoff-x.min())), 
                                     #lambda x: 1e-4 + 0.9999*((x-x.min())/(lower_cutoff-x.min())), 
                                     lambda x: 1, 
                                     #Three different cut off methods of distance weighting below. We only pick one at a time!
                                     lambda x: (1- sigmoid((x - ((upper_sigmoid_lims[0] + upper_sigmoid_lims[1])/2) )/sig_div)) #sigmoid decay
                                     #lambda x: (1- 0.9999*((x-upper_cutoff)/(np.max(d_bounds_pred) - upper_cutoff))) #symetric linear decay
                                     #lambda x: 1e-4 + 0.9999*np.exp(-decay_constant*(x - upper_cutoff))  #Exponential decay - N = N0 e**(-alpha d). In our case N0 is 0.9999, because we want to decay from one to 1e-4
                                     #lambda x: 1e-4 #sharp cutoff
                                     ]
                                    )
        # if l_bounds_pred[0] == 0.0 and l_bounds_pred[-1] == 360.0:
        #     weights_d*=4
        weights_local = weights_local[..., np.newaxis] * weights_d

        #now we can read the density/extinctions (only do one at a time!)
        ext_samples = load_samples(direc+"/ext_all_cube.pkl.npy", n_l, n_b, n_d)

        if np.min(l_bounds_pred) < 0.:

            # import pdb; pdb.set_trace()
            #In this case, we've loaded a chunk with l_min = 351, l_max = 9 (or similar) which Dustribution has converted so that l_min = -9
            #Now we need to make sure we consider the correct part of it depending on what we're looking at here.
            if (l_bounds_pred[0] < merge_lbounds[0]-tol) and (l_bounds_pred[-1] > merge_lbounds[-1]+tol):
                #In this case, the chunk to be merged is bigger than the box we're merging into, so it needs special treatment
                l_start_chunk = np.where(np.abs(l_bounds_pred - merge_lbounds[0]) < tol)[0][0]
                l_end_chunk = np.where(np.abs(l_bounds_pred - merge_lbounds[-1]) < tol)[0][0]
                l_start = 0
                l_end = None
            elif (l_bounds_pred[0]+360 < merge_lbounds[0]-tol) and (l_bounds_pred[-1]+360 > merge_lbounds[-1]+tol): #this means we're merging into a box that is approaching 360 and our chunk to be merged is bigger than the whole box we're merging into
                l_start_chunk = np.where(np.abs(l_bounds_pred - (merge_lbounds[0]-360)) < tol)[0][0]
                l_end_chunk = np.where(np.abs(l_bounds_pred - (merge_lbounds[-1] -360)) < tol)[0][0]
                l_start = 0
                l_end = None
            elif l_bounds_pred[-1] >= merge_lbounds[0]-tol:
                #in this case, we are merging into a box that starts at 0
                l_start_chunk = np.where(np.abs(l_bounds_pred - merge_lbounds[0]) < tol)[0][0]
                l_start = 0
                l_end_chunk = None
                l_end = np.where(np.abs(merge_lbounds - l_bounds_pred[-1]) < tol)[0][0]
            else:
                #in this case, we must be merging into a box that ends at 360 - this is a little more complicated
                l_start_chunk = 0
                l_start = np.where(np.abs(merge_lbounds - (l_bounds_pred[0] + 360)) < tol)[0][0]
                l_end_chunk = np.where(np.abs((l_bounds_pred+360) - merge_lbounds[-1]) < tol)[0][0]
                if l_end_chunk >= len(l_bounds_pred):
                    l_end_chunk = None
                l_end = None
        else:
            #Now we have to figure out where they fit in the full array
            if l_bounds_pred[0] >= merge_lbounds[0]-tol: #This needs to be modified for a tolerance! #Is the start of the chunk inside the edge of the red square in l?
                l_start_chunk = 0
                l_start = np.where(np.abs(merge_lbounds - l_bounds_pred[0]) < tol)[0][0]    
            else:
                l_start_chunk = np.where(np.abs(l_bounds_pred - merge_lbounds[0]) < tol)[0][0]
                l_start = 0

            if l_bounds_pred[-1] <= merge_lbounds[-1]+tol:#This needs to be modified for a tolerance! #Is the end of the chunk inside the edge of the red square in l?
                # in this case we need to go from l_start_chunk to the end of the chunk
                l_end_chunk=None
                l_end = np.where(np.abs(merge_lbounds - l_bounds_pred[-1]) < tol)[0][0]
            else:
                l_end_chunk = np.where(np.abs(l_bounds_pred - merge_lbounds[-1]) < tol)[0][0]
                l_end = None

        
        if b_bounds_pred[0] >= merge_bbounds[0]-tol: #This needs to be modified for a tolerance! #Is the start of the chunk inside the edge of the red square in l?
            b_start_chunk = 0
            b_start = np.where(np.abs(merge_bbounds - b_bounds_pred[0]) < tol)[0][0]
        else:
            b_start_chunk = np.where(np.abs(b_bounds_pred - merge_bbounds[0]) < tol)[0][0]
            b_start = 0

        if b_bounds_pred[-1] <= merge_bbounds[-1]+tol:#This needs to be modified for a tolerance! #Is the end of the chunk inside the edge of the red square in l?
            # in this case we need to go from l_start_chunk to the end of the chunk
            b_end_chunk=None
            b_end = np.where(np.abs(merge_bbounds - b_bounds_pred[-1]) < tol)[0][0]
        else:
            b_end_chunk = np.where(np.abs(b_bounds_pred - merge_bbounds[-1]) < tol)[0][0]
            b_end = None


        d_start = np.where(np.abs(merge_dbounds - d_bounds_pred[0]) < dtol )[0][0]
        
        cov[l_start:l_end, b_start:b_end, d_start:d_start+n_d+1]+=1

        #More difficult is figuring out where along the samples axis we will be putting them. This has to be deterministic, otherwise things will get weird
        for i in range(n_overlap):
            s = np.sum(fillmask[i, l_start:l_end, b_start:b_end, d_start:d_start+n_d+1])
            if not s: #fillmask is boolean, so this is only True if all elements of the range of fillmask are False
                fillmask[i, l_start:l_end, b_start:b_end, d_start:d_start+n_d+1] = True
                s_start = i * n_samples_per_map #This way we always fill from the start if there is room. I'm not sure if clashes are possible
                break
        else:
            #Something went badly wrong and no available slots were found!!
            raise IndexError("Failed to find a segment of the merged array that isn't already filled with samples for model {0}!".format(direc))

        #Now we could generate some random integers to select a subsample of the samples
        #But the samples generated by the GP should be unsorted and unbiased, so we can just take the first (or last, or any other) n_samples_per_map samples from each model
        #So we end up with:
        merged_ext_samples[s_start:s_start+n_samples_per_map, l_start:l_end, b_start:b_end, d_start:d_start+n_d+1] = ext_samples[:n_samples_per_map, l_start_chunk:l_end_chunk, b_start_chunk:b_end_chunk,...]
        weights[s_start:s_start+n_samples_per_map, l_start:l_end, b_start:b_end, d_start:d_start+n_d+1] = weights_local[np.newaxis, l_start_chunk:l_end_chunk, b_start_chunk:b_end_chunk, ...]

    #debugging nans:
    print("sum and shape of nan vals mergerged ext samples ==", np.sum(np.isnan(merged_ext_samples)), np.prod(merged_ext_samples.shape))
    #merged_ext_median = np.nanmedian(merged_ext_samples, axis=0)

    samples_xr = xr.DataArray(merged_ext_samples, dims=['samples', 'l', 'b', 'd'])
    print(samples_xr.dims)
    weights_xr = xr.DataArray(weights, dims=['samples', 'l', 'b', 'd'])#.expand_dims(dim=['d'])
    #samples_xr_b, weights_xr_b = xr.broadcast(samples_xr, weights_xr) #We should be able to remove this line, because now the two arrays have the same shape
    print(weights_xr.dims)
    weighted_samples = weighted.DataArrayWeighted(samples_xr, weights_xr)
    weighted_median_ext = weighted_samples.quantile(0.5, dim='samples', skipna=True)
    merged_ext_16P = weighted_samples.quantile(0.16, dim='samples', skipna=True)
    merged_ext_84P = weighted_samples.quantile(0.84, dim='samples', skipna=True)


    #Save full MW cumilative extinction cubes
    np.save(merge_coords_print+"_MWchunk_CumExt_Weighted_Median.pkl", weighted_median_ext.to_numpy(), allow_pickle=True) #merged_ext_weighted_mean, allow_pickle=True)#_median, allow_pickle=True)
    np.save(merge_coords_print+"_MWchunk_CumExt_16P.pkl", merged_ext_16P.to_numpy(), allow_pickle=True)
    np.save(merge_coords_print+"_MWchunk_CumExt_84P.pkl", merged_ext_84P.to_numpy(), allow_pickle=True)

    #Plot extinction
    print("Begin plotting extinction")
    
    #Plot final cumilative extinction 
    plot_GP_Pred_ExtCumilative(merge_lbounds, merge_bbounds, weighted_median_ext, merge_coords_print)#merged_ext_weighted_mean)# median)#ext_med_cube)
    plot_GP_Pred_Ext_Slices_AlongDist(merge_lbounds, merge_bbounds, merge_dbounds, weighted_median_ext, merge_coords_print) #merged_ext_weighted_mean)#_median) #ext_med_cube)


    print("End plotting extinction")

    print("cov mean == ", np.mean(cov))
    print("cov std.dev", np.std(cov))

    print("Extinction Merge Done")



    print("Begin Density Merging")
    ## Now we want to move on to doing density - 
    #we don't do both density and extinction at the same time because we want to minimise memory usage - the merged extinction/density array requires ~156 GB of RAM, so by doing one then the other we allow a wider range of machines to compute it
    #First we're going to dump the memory used by all the arrays so far, and reset any arrays we'll be reusing
    del merged_ext_samples
    del merged_ext_16P#Delete ext data
    del merged_ext_84P  #Delete ext data
    del samples_xr #Delete 
    del weighted_median_ext
    del weights_xr
    del weights
    cov[...] = 0  #Reuse cov for density
    fillmask[...] = False #Reuse fillmask for density 

    gc.collect()


    #denseity merging
    merged_dense_samples = np.full((n_samples_per_map * n_overlap, size_l_merge, size_b_merge, size_d_merge+1), np.nan)
    weights = np.zeros((n_samples_per_map * n_overlap, size_l_merge, size_b_merge, size_d_merge+1))

    #We also create a smaller array to record how many models cover each point so we can cut out regions later
    cov = np.zeros((size_l_merge, size_b_merge, size_d_merge+1), dtype=int)
    fillmask = np.zeros((n_overlap, size_l_merge, size_b_merge, size_d_merge+1), dtype=bool)

    #Now we return to iterating over all the results, this time to read in the results and put a sample into the right place in the big array allocated above
    for i, direc in enumerate(overlapping_chunks):
        print(direc)
        #We have to read in the cell boundaries again
        l_bounds_pred, b_bounds_pred, d_bounds_pred = load_coords(direc)
        n_l = len(l_bounds_pred) - 1
        n_b = len(b_bounds_pred) - 1
        n_d = len(d_bounds_pred) - 1

        if np.abs(l_bounds_pred[-1] - merge_lbounds[0]) <= tol:
            continue #Since the upper bound of the chunk is equal to the lower bound of the area to be merged, there isn't actually any overlap and we should skip this chunk
        if np.abs(l_bounds_pred[0] - merge_lbounds[-1]) <= tol or np.abs(l_bounds_pred[0]+360 - merge_lbounds[-1]) <= tol:
            continue #The opposite of the above case - the lower bound of the chunk is equla to the upper bound of the area to be merged.
        if np.abs(b_bounds_pred[-1] - merge_bbounds[0]) <= tol:
            continue #Since the upper bound of the chunk is equal to the lower bound of the area to be merged, there isn't actually any overlap and we should skip this chunk
        if np.abs(b_bounds_pred[0] - merge_bbounds[-1]) <= tol:
            continue #The opposite of the above case - the lower bound of the chunk is equla to the upper bound of the area to be merged.
        if np.abs(d_bounds_pred[-1] - merge_dbounds[0]) <= dtol:
            continue #Since the upper bound of the chunk is equal to the lower bound of the area to be merged, there isn't actually any overlap and we should skip this chunk
        if np.abs(d_bounds_pred[0] - merge_dbounds[-1]) <= dtol:
            continue #The opposite of the above case - the lower bound of the chunk is equla to the upper bound of the area to be merged.

        if l_bounds_pred[0] == 0.0 and l_bounds_pred[-1] == 360.0:
            weights_local = np.ones((l_bounds_pred[:-1].shape[0], b_bounds_pred[:-1].shape[0]))
        else:
            #let's set up a triangle function for the weights to start us off
            l_grid, b_grid = np.meshgrid((l_bounds_pred[:-1] + l_bounds_pred[1:])/2, (b_bounds_pred[:-1] + b_bounds_pred[1:])/2, indexing='ij')
            r = (l_grid - np.average(l_bounds_pred))**2 + (b_grid - np.average(b_bounds_pred))**2
            #weights_local = (1 - (r / np.max(r)))**2
            weights_local = ( 1 - (0.9999* ( r / np.max(r) ) ) )**2 #multiplying by 0.9999 ensures that there are no zeros and the smallest weight is 1e-4. This avoids nans in the final map


        if d_bounds_pred[-1] < 600.:
            lower_sigmoid_lims = (0, d_bounds_pred.min())
            upper_sigmoid_lims = (d_bounds_pred.max(), d_bounds_pred.min())
            if l_bounds_pred[0] == 0.0 and l_bounds_pred[-1] == 360.0: #This case is the solar neighbourhood with the 100 pc scale length
                lower_cutoff = 0
                upper_cutoff = 200
            else: #So this is the Solar Neighbourhood but with the 10 pc scale length
                lower_cutoff = 100
                upper_cutoff = 200
        else:
            try:
                dist_chunk_num = np.argwhere(np.min(d_bounds_pred) < min_d_bounds_pred_Dchunk)[0][0]
            except IndexError:
                print("The minimum distance for this chunk is larger than any of the values in min_d_bounds_pred_Dchunk. \nPlease check that the input is correct.")
                print("Chunk ==", direc)
                print("Minimum distance ==", np.min(d_bounds_pred))
                print("input min_d_bounds_pred_Dchunk ==", min_d_bounds_pred_Dchunk)
                raise IndexError
            try:
                lower_cutoff = dweight_lower_cutoff[dist_chunk_num]
            except IndexError:
                print("The number of entries in min_d_bounds_pred_Dchunk doesn't match up with the number of lower cutoffs provided. \nPlease check that the input is correct")
                raise IndexError
            try:
                upper_cutoff = dweight_upper_cutoff[dist_chunk_num]
            except IndexError:
                print("The number of entries in min_d_bounds_pred_Dchunk doesn't match up with the number of upper cutoffs provided. \nPlease check that the input is correct")
                raise IndexError
            try:
                lower_sigmoid_lims = (chunk_boundaries_upper[dist_chunk_num-1], chunk_boundaries_lower[dist_chunk_num])
            except IndexError:
                lower_sigmoid_lims = (0, chunk_boundaries_lower[dist_chunk_num])
            except NameError:
                lower_sigmoid_lims = (0, chunk_boundaries_lower[0])

            if lower_sigmoid_lims[0] > d_bounds_pred.max() + 100:
                lower_sigmoid_lims = (0, chunk_boundaries_lower[dist_chunk_num])

            try:
                upper_sigmoid_lims = (chunk_boundaries_upper[dist_chunk_num], chunk_boundaries_lower[dist_chunk_num+1])
            except IndexError:
                upper_sigmoid_lims = (chunk_boundaries_upper[dist_chunk_num], np.max(d_bounds_pred))
            except NameError:
                upper_sigmoid_lims = (chunk_boundaries_upper[0], np.max(d_bounds_pred))

        # decay_constant = 0.1 #10 #modify this to change how quickly the weights decay in the exponential decay case!
        # mask_lower = d_bounds_pred < lower_cutoff
        # mask_upper = d_bounds_pred > upper_cutoff
        # mask_middle = np.logical_not(np.logical_or(mask_lower, mask_upper))

        mask_lower = d_bounds_pred < lower_sigmoid_lims[0] #lower_cutoff
        mask_upper = d_bounds_pred > upper_sigmoid_lims[1] # upper_cutoff
        mask_middle = np.logical_not(np.logical_or(mask_lower, mask_upper))
        if d_bounds_pred[-1] < 600.:
            weights_d = np.piecewise(d_bounds_pred, 
                                    [mask_lower, mask_middle, mask_upper], 
                                    [lambda x: 1e-4 + sigmoid((x - 
                                                            (
                                                                (lower_sigmoid_lims[0] + 
                                                                lower_sigmoid_lims[1])
                                                                /2
                                                            ))
                                                            /sig_div
                                                            ), #0.9999*((x-x.min())/(lower_cutoff-x.min())), 
                                     #lambda x: 1e-4 + 0.9999*((x-x.min())/(lower_cutoff-x.min())), 
                                     lambda x: 1, 
                                     #Three different cut off methods of distance weighting below. We only pick one at a time!
                                     lambda x: (1- sigmoid((x - ((upper_sigmoid_lims[0] + upper_sigmoid_lims[1])/2) )/sig_div)) #sigmoid decay
                                     #lambda x: (1- 0.9999*((x-upper_cutoff)/(np.max(d_bounds_pred) - upper_cutoff))) #symetric linear decay
                                     #lambda x: 1e-4 + 0.9999*np.exp(-decay_constant*(x - upper_cutoff))  #Exponential decay - N = N0 e**(-alpha d). In our case N0 is 0.9999, because we want to decay from one to 1e-4
                                     #lambda x: 1e-4 #sharp cutoff
                                     ]
                                    )
        else:
            mask_under_95 = d_bounds_pred < 95.
            weights_d = np.piecewise(d_bounds_pred, 
                                    [mask_under_95,mask_lower, mask_middle, mask_upper], 
                                    [lambda x: 1e-2,
                                     lambda x: 1e-4 + sigmoid((x - ((lower_sigmoid_lims[0] + lower_sigmoid_lims[1])/2))/sig_div), #0.9999*((x-x.min())/(lower_cutoff-x.min())), 
                                     #lambda x: 1e-4 + 0.9999*((x-x.min())/(lower_cutoff-x.min())), 
                                     lambda x: 1, 
                                     #Three different cut off methods of distance weighting below. We only pick one at a time!
                                     lambda x: (1- sigmoid((x - ((upper_sigmoid_lims[0] + upper_sigmoid_lims[1])/2) )/sig_div)) #sigmoid decay
                                     #lambda x: (1- 0.9999*((x-upper_cutoff)/(np.max(d_bounds_pred) - upper_cutoff))) #symetric linear decay
                                     #lambda x: 1e-4 + 0.9999*np.exp(-decay_constant*(x - upper_cutoff))  #Exponential decay - N = N0 e**(-alpha d). In our case N0 is 0.9999, because we want to decay from one to 1e-4
                                     #lambda x: 1e-4 #sharp cutoff
                                     ]
                                    )
        # if l_bounds_pred[0] == 0.0 and l_bounds_pred[-1] == 360.0:
        #     weights_d*=4
        weights_local = weights_local[..., np.newaxis] * weights_d

        #now we can read the denseity/denseitys (only do one at a time!)
        dense_samples = load_samples(direc+"/gpy_dens_samples_all.out", n_l, n_b, n_d)

        if np.min(l_bounds_pred) < 0.:
            #In this case, we've loaded a chunk with l_min = 351, l_max = 9 (or similar) which Dustribution has converted so that l_min = -9
            #Now we need to make sure we consider the correct part of it depending on what we're looking at here.
            if (l_bounds_pred[0] < merge_lbounds[0]-tol) and (l_bounds_pred[-1] > merge_lbounds[-1]+tol):
                #In this case, the chunk to be merged is bigger than the box we're merging into, so it needs special treatment
                l_start_chunk = np.where(np.abs(l_bounds_pred - merge_lbounds[0]) < tol)[0][0]
                l_end_chunk = np.where(np.abs(l_bounds_pred - merge_lbounds[-1]) < tol)[0][0]
                l_start = 0
                l_end = None
            elif (l_bounds_pred[0]+360 < merge_lbounds[0]-tol) and (l_bounds_pred[-1]+360 > merge_lbounds[-1]+tol): #this means we're merging into a box that is approaching 360 and our chunk to be merged is bigger than the whole box we're merging into
                l_start_chunk = np.where(np.abs(l_bounds_pred - (merge_lbounds[0]-360)) < tol)[0][0]
                l_end_chunk = np.where(np.abs(l_bounds_pred - (merge_lbounds[-1] -360)) < tol)[0][0]
                l_start = 0
                l_end = None
            elif l_bounds_pred[-1] >= merge_lbounds[0]-tol:
                #in this case, we are merging into a box that starts at 0
                l_start_chunk = np.where(np.abs(l_bounds_pred - merge_lbounds[0]) < tol)[0][0]
                l_start = 0
                l_end_chunk = None
                l_end = np.where(np.abs(merge_lbounds - l_bounds_pred[-1]) < tol)[0][0]
            else:
                #in this case, we must be merging into a box that ends at 360 - this is a little more complicated
                l_start_chunk = 0
                l_start = np.where(np.abs(merge_lbounds - (l_bounds_pred[0] + 360)) < tol)[0][0]
                l_end_chunk = np.where(np.abs((l_bounds_pred+360) - merge_lbounds[-1]) < tol)[0][0]
                l_end = None
        else:
            #Now we have to figure out where they fit in the full array
            if l_bounds_pred[0] >= merge_lbounds[0]-tol: #This needs to be modified for a tolerance! #Is the start of the chunk inside the edge of the red square in l?
                l_start_chunk = 0
                l_start = np.where(np.abs(merge_lbounds - l_bounds_pred[0]) < tol)[0][0]    
            else:
                l_start_chunk = np.where(np.abs(l_bounds_pred - merge_lbounds[0]) < tol)[0][0]
                l_start = 0

            if l_bounds_pred[-1] <= merge_lbounds[-1]+tol:#This needs to be modified for a tolerance! #Is the end of the chunk inside the edge of the red square in l?
                # in this case we need to go from l_start_chunk to the end of the chunk
                l_end_chunk=None
                l_end = np.where(np.abs(merge_lbounds - l_bounds_pred[-1]) < tol)[0][0]
            else:
                l_end_chunk = np.where(np.abs(l_bounds_pred - merge_lbounds[-1]) < tol)[0][0]
                l_end = None

        
        if b_bounds_pred[0] >= merge_bbounds[0]-tol: #This needs to be modified for a tolerance! #Is the start of the chunk inside the edge of the red square in l?
            b_start_chunk = 0
            b_start = np.where(np.abs(merge_bbounds - b_bounds_pred[0]) < tol)[0][0]
        else:
            b_start_chunk = np.where(np.abs(b_bounds_pred - merge_bbounds[0]) < tol)[0][0]
            b_start = 0

        if b_bounds_pred[-1] <= merge_bbounds[-1]+tol:#This needs to be modified for a tolerance! #Is the end of the chunk inside the edge of the red square in l?
            # in this case we need to go from l_start_chunk to the end of the chunk
            b_end_chunk=None
            b_end = np.where(np.abs(merge_bbounds - b_bounds_pred[-1]) < tol)[0][0]
        else:
            b_end_chunk = np.where(np.abs(b_bounds_pred - merge_bbounds[-1]) < tol)[0][0]
            b_end = None


        d_start = np.where(np.abs(merge_dbounds - d_bounds_pred[0]) < dtol )[0][0]
        
        cov[l_start:l_end, b_start:b_end, d_start:d_start+n_d]+=1

        #More difficult is figuring out where along the samples axis we will be putting them. This has to be deterministic, otherwise things will get weird
        for i in range(n_overlap):
            s = np.sum(fillmask[i, l_start:l_end, b_start:b_end, d_start:d_start+n_d])
            if not s: #fillmask is boolean, so this is only True if all elements of the range of fillmask are False
                fillmask[i, l_start:l_end, b_start:b_end, d_start:d_start+n_d] = True
                s_start = i * n_samples_per_map #This way we always fill from the start if there is room. I'm not sure if clashes are possible
                break
        else:
            #Something went badly wrong and no available slots were found!!
            raise IndexError("Failed to find a segment of the merged array that isn't already filled with samples for model {0}!".format(direc))

        #Now we could generate some random integers to select a subsample of the samples
        #But the samples generated by the GP should be unsorted and unbiased, so we can just take the first (or last, or any other) n_samples_per_map samples from each model
        #So we end up with:
        merged_dense_samples[s_start:s_start+n_samples_per_map, l_start:l_end, b_start:b_end, d_start:d_start+n_d] = dense_samples[:n_samples_per_map, l_start_chunk:l_end_chunk, b_start_chunk:b_end_chunk,...]
        weights[s_start:s_start+n_samples_per_map, l_start:l_end, b_start:b_end, d_start:d_start+n_d] = weights_local[np.newaxis, l_start_chunk:l_end_chunk, b_start_chunk:b_end_chunk, :n_d]

    #debugging nans:
    print("sum and shape of nan vals mergerged denseity samples ==", np.sum(np.isnan(merged_dense_samples)), np.prod(merged_dense_samples.shape))
    #merged_dense_median = np.nanmedian(merged_dense_samples, axis=0)

    samples_xr = xr.DataArray(merged_dense_samples, dims=['samples', 'l', 'b', 'd'])
    print(samples_xr.dims)
    weights_xr = xr.DataArray(weights, dims=['samples', 'l', 'b', 'd'])#.expand_dims(dim=['d'])
    samples_xr_b, weights_xr_b = xr.broadcast(samples_xr, weights_xr) #hopefully this line can be removed
    print(weights_xr.dims)
    weighted_samples = weighted.DataArrayWeighted(samples_xr_b, weights_xr_b)
    weighted_median_dense = weighted_samples.quantile(0.5, dim='samples', skipna=True)
    merged_dense_16P = weighted_samples.quantile(0.16, dim='samples', skipna=True)
    merged_dense_84P = weighted_samples.quantile(0.84, dim='samples', skipna=True)


    np.save(merge_coords_print+"_MWchunk_Dens_Weighted_Median.pkl", weighted_median_dense.to_numpy(), allow_pickle=True) #merged_ext_weighted_mean, allow_pickle=True)#_median, allow_pickle=True)
    np.save(merge_coords_print+"_MWchunk_Dens_16P.pkl", merged_dense_16P.to_numpy(), allow_pickle=True)
    np.save(merge_coords_print+"_MWchunk_Dens_84P.pkl", merged_dense_84P.to_numpy(), allow_pickle=True)

    
    print("Begin plotting Density")


    #Plot selected slices along distance of predicted denseity 
    plot_GP_Pred_Dens_Slices_AlongDist(size_l_merge, size_b_merge, size_d_merge, merge_lbounds, merge_bbounds, merge_dbounds, weighted_median_dense, merge_coords_print) #merged_dense_weighted_mean)#_median) #gpy_dense_median)


    
    print("Density Merge Done")

    print("Code Run Time --- %s seconds ---" % (time.time() - start_time))



if __name__=="__main__":


    full_MergeFunc()


















