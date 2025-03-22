"""
Created on Tue Jul 30 18:06:44 2019

@author: Neven Caplar
ncaplar@princeton.edu
www.ncaplar.com
 
"""

import numpy as np
import socket
import time
import pickle
import sys

print(str(socket.gethostname())+': Starting calculation at: '+time.ctime())   
time_start_single=time.time()
def bootstrap_resample(X, n=None):
    """ Bootstrap resample an array_like
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


def create_meanLC_std_mean_median_as_function_of_time(Edd_ratio_bin,return_array_of_res_cut_as_function_of_Edd_ratio_many_LC=False):
    
    list_of_res_cut_as_function_of_Edd_ratio_many_LC=[]
    list_of_rec_LC_values_as_function_of_Edd_ratio_many_LC=[]
    
    # this loops over different LCs
    for l in range(len(positions_of_many_specific_Edd_ratios_many_LC)):

        # take positions for a single LC
        positions_of_many_specific_Edd_ratios=positions_of_many_specific_Edd_ratios_many_LC[l]
        # extract single LC
        LC=array_of_LC[l]

        list_of_res_cut_as_function_of_Edd_ratio_single_LC=[]
        list_of_rec_LC_values_as_function_of_Edd_ratio_single_LC=[]
        # select one bin of starting Eddington ratio positions
        j=Edd_ratio_bin

        for dt in selection_of_times:
            res_cut=[]
            rec_LC_values=[]
            # for each starting Eddington ratio, find differences
            for i in positions_of_many_specific_Edd_ratios[j]:
                if (i+selection_of_times[-1])>len(LC):
                    pass
                else:
                    rec_LC_values.append(LC[i])
                    res_cut.append(-LC[i]+LC[i+dt])
            list_of_res_cut_as_function_of_Edd_ratio_single_LC.append(res_cut)
            list_of_rec_LC_values_as_function_of_Edd_ratio_single_LC.append(rec_LC_values)

        list_of_res_cut_as_function_of_Edd_ratio_many_LC.append(list_of_res_cut_as_function_of_Edd_ratio_single_LC)
        list_of_rec_LC_values_as_function_of_Edd_ratio_many_LC.append(list_of_rec_LC_values_as_function_of_Edd_ratio_single_LC)

    # creates array which has differences of Eddington values, separated in time bins
    array_of_res_cut_as_function_of_Edd_ratio_many_LC=np.array(list_of_res_cut_as_function_of_Edd_ratio_many_LC)

    # creates array which has starting values, separated in time bins
    array_of_rec_LC_values_as_function_of_Edd_ratio_many_LC=np.array(list_of_rec_LC_values_as_function_of_Edd_ratio_many_LC)
    
    
    # measured SF as a function of (starting) Eddington ratio
    measured_std_as_fun_of_Edd_ratio=[]
    # measured mean dif. of Eddington ratio as a function of (starting) Eddington ratio
    measured_mean_as_fun_of_Edd_ratio=[]
    # measured median dif. of Eddington ratio as a function of (starting) Eddington ratio
    measured_median_as_fun_of_Edd_ratio=[]

    mean_LC_values=[]
    # each one of the analyzed LCs has been divided in the same amount of bins, so we can take any    
    for i in range(len(array_of_res_cut_as_function_of_Edd_ratio_many_LC[0])):
        bootstrap_sample_std=[]
        bootstrap_sample_mean=[]
        bootstrap_sample_median=[]
        
        
        #print(array_of_res_cut_as_function_of_Edd_ratio_many_LC.shape)
        #print(i)
        #print(array_of_res_cut_as_function_of_Edd_ratio_many_LC[:,i])
        array_of_res_cut_as_function_of_Edd_ratio_many_LC_single_Edd_ratio_bin=np.concatenate(array_of_res_cut_as_function_of_Edd_ratio_many_LC[:,i]).ravel()
        array_of_res_LC_values_as_function_of_Edd_ratio_many_LC_single_Edd_ratio_bin=np.concatenate(array_of_rec_LC_values_as_function_of_Edd_ratio_many_LC[:,i]).ravel()
        
        mean_LC_values.append(np.mean(array_of_res_LC_values_as_function_of_Edd_ratio_many_LC_single_Edd_ratio_bin))
        
        for j in range(21):
            bootstrap_resample_single_run=bootstrap_resample(array_of_res_cut_as_function_of_Edd_ratio_many_LC_single_Edd_ratio_bin)

            bootstrap_sample_std.append(np.std(bootstrap_resample_single_run))
            bootstrap_sample_mean.append(np.mean(bootstrap_resample_single_run))
            bootstrap_sample_median.append(np.median(bootstrap_resample_single_run))

        measured_std_as_fun_of_Edd_ratio.append([np.mean(bootstrap_sample_std),np.std(bootstrap_sample_std)])
        measured_mean_as_fun_of_Edd_ratio.append([np.mean(bootstrap_sample_mean),np.std(bootstrap_sample_mean)])
        measured_median_as_fun_of_Edd_ratio.append([np.mean(bootstrap_sample_median),np.std(bootstrap_sample_median)])

    # first column = mean (i.e., SF), second column = std (uncertainity on the SF)
    measured_std_as_fun_of_Edd_ratio=np.array(measured_std_as_fun_of_Edd_ratio)

    # first column = mean difference, second column = std of mean differences
    measured_mean_as_fun_of_Edd_ratio=np.array(measured_mean_as_fun_of_Edd_ratio)

    # first column = median difference, second column = std of median differences
    measured_median_as_fun_of_Edd_ratio=np.array(measured_median_as_fun_of_Edd_ratio)
    if return_array_of_res_cut_as_function_of_Edd_ratio_many_LC==False:
        return mean_LC_values,measured_std_as_fun_of_Edd_ratio,measured_mean_as_fun_of_Edd_ratio,measured_median_as_fun_of_Edd_ratio
    if return_array_of_res_cut_as_function_of_Edd_ratio_many_LC==True:
        return array_of_res_cut_as_function_of_Edd_ratio_many_LC,array_of_rec_LC_values_as_function_of_Edd_ratio_many_LC


v_number=sys.argv[1]
l_number=sys.argv[2]
a_number=sys.argv[3]
L_number=sys.argv[4]

print('v_'+str(v_number)+'_l_'+str(l_number)+'_a_'+str(a_number)+'_L_'+str(L_number)) 
# largest separation that is of interest
dt_max=1000

selection_of_times=np.unique((10**np.arange(0,5.5,0.1)).astype(int))


  
#array_of_LC=[]
#for i in range(1,3):
#    ER_curve = np.zeros(2**24, dtype = float)   
#    ER_curve = np.fromfile('/tigress/ncaplar/GpuData/results_test1_'+str(i)+'.bin', dtype = float)
#    array_of_LC.append(np.log10(ER_curve))
#array_of_LC=np.array(array_of_LC)    

array_of_LC=np.load('/tigress/ncaplar/GpuData/results_v_'+str(v_number)+'_l_'+str(l_number)+'_a_'+str(a_number)+'_L_'+str(L_number)+'_array_of_LC_16.npy')
# maximal value in the light-curves
max_LC_value=np.max(array_of_LC)
# minimal value in the light-curves
min_LC_value=np.min(array_of_LC)    


# takes around a minute for set of 20 LC
positions_of_many_specific_Edd_ratios_many_LC=[]

for l in range(len(array_of_LC)):
    positions_of_many_specific_Edd_ratios=[]
    for Edd in np.arange(min_LC_value,max_LC_value,0.1):
        positions_of_specific_Edd_ratio=np.nonzero((array_of_LC[l][:len(array_of_LC[l])-dt_max]>Edd)&(array_of_LC[l][:len(array_of_LC[l])-dt_max]<(Edd+0.1)))[0]
        if len(positions_of_specific_Edd_ratio)<2500:
            positions_of_many_specific_Edd_ratios.append(positions_of_specific_Edd_ratio)        
        else:
            positions_of_specific_Edd_ratio_random=np.random.choice(positions_of_specific_Edd_ratio,2500)
            positions_of_many_specific_Edd_ratios.append(positions_of_specific_Edd_ratio_random)            
        
    positions_of_many_specific_Edd_ratios_many_LC.append(np.array(positions_of_many_specific_Edd_ratios))
    
positions_of_many_specific_Edd_ratios_many_LC=np.array(positions_of_many_specific_Edd_ratios_many_LC)    

list_of_single_LC_result=[]
for Edd_rat in range(len(positions_of_many_specific_Edd_ratios)):
    mean_LC_values_with_time,measured_std_as_fun_of_Edd_ratio_with_time,measured_mean_as_fun_of_Edd_ratio_with_time,\
    measured_median_as_fun_of_Edd_ratio_with_time=create_meanLC_std_mean_median_as_function_of_time(Edd_rat)
    list_of_single_LC_result.append([mean_LC_values_with_time,measured_std_as_fun_of_Edd_ratio_with_time,measured_mean_as_fun_of_Edd_ratio_with_time,\
    measured_median_as_fun_of_Edd_ratio_with_time])

#np.save('/tigress/ncaplar/GpuData/AnalysisResults/list_of_single_LC_result',list_of_single_LC_result)

with open("/tigress/ncaplar/GpuData/AnalysisResults/list_of_single_LC_result"+'v_'+str(v_number)+'_l_'+str(l_number)+'_a_'+str(a_number)+'_L_'+str(L_number)+".txt", "wb") as fp:   #Pickling
    pickle.dump(list_of_single_LC_result, fp)


print('Time when total script finished was: '+time.ctime())     
time_end=time.time()   
print('Total time taken was  '+str(time_end-time_start_single)+' seconds')
