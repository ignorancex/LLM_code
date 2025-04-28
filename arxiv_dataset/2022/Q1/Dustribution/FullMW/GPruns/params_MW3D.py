import numpy as np

#Sky l range 
l_sky_start = 0
l_sky_start_overlap = 9
l_sky_end = 360
l_chunk_size = 18

#Sky b range
b_lower = -90 #-30 #we don't need this for l as it's a simple linear list
b_upper = 90 #30
b_step = 8 #4

#All other d chunks ranges (all except the first)
first_d_chunk = False #Is this the first (0-xpc) dist chunk where we don't need to add the past dist chunk cum_ext during the training phase
d_min_new_approx = 1800 #Approximate start value (d_min) for next d chunk. The exact value will be extracted from the pre Dist chunk #To be updated based on the distance chunk we are running

#First d chunk range
d_min_first_d_chunk = 10 #pc
d_max_first_d_chunk = 1000 #pc
diff_first_d_chunk = d_max_first_d_chunk - d_min_first_d_chunk

#Rerun statuses if we want to recreate grids or retrain GP from the stars
recalc_grid_train = True #True/False #If we do/do not want to recalculate the training grid defined based on the lbd coordinates and n resolution above - must be recalculated if the coordinates or resolutions deifned above change
recalc_grid_pred = True #True/False #If we do/do not want to recalculate the predicting grid defined based on the lbd coordinates and n resolution above - must be recalculated if the coordinates or resolutions deifned above change
recheck_sourcebounds = True #True/False #If we do/do not want to reheck source positions in source data frame to make sure their within our input coorindates and removing any which are not. Also re-precomputes some source indeces for integration optimisation
retrain_gp = True #True/False  #If we do/do not want to rerun the full GP training based parameters given above - must be recalculated if any of the parameters above change except the prediciting grid parameters
repredict_gp = True #True/False  #If we do/do not want to rerun the GP prediction based parameters given above - must be recalculated if any of the parameters above change

#Run Algorithm on GPUs or CPUs
train_gpu = True #Set True for GPU run or set False for CPU run for the GP training
pred_gpu = True #Set True for GPU run or set False for CPU run for the GP predicting. #We also need to set this to False if we're only plotting in a machine with No GPU
plot_gpu = True  #Set True for plotting in the GPU or set False for CPU for plotting following GPU training especially if plotting is done outside a machine with a GPU

#Pixle size for each l,b,d chunk. #number of cells within boundaries - defines resolution of grid
n_l_train = 18 
n_b_train = 14
n_d_train = 245

n_l_pred = 44
n_b_pred = 34 #44*(14/18)
n_d_pred = 585

#GP hyperparameter priors for the RBF Kernel - Setsup the staring point for the hyperparameters - they are allowed to be optimised with the GP
scale_length_x = 10.0 #10.0 #Scale length - parsec units - #Approximated from literature given size of structure in region of interest dependent on how close/far we are from the source allowing us to recover the smaller/larger structure. If the region of interest is further away from us we will likely only be able to recover the large scale structure so we need a larger scale length to reflect the size of the large scale structure. But if we are close by we maybe sensitive to the small scale structure as well so we need a smaller scale lenght to reflect the smaller stucture sizes.  
scale_length_y = 10.0 #10.0
scale_length_z = 10.0 #10.0 
mean_ext_dens = -3.333 #-3.333 #Mean density - log10(Mags per pc) units - Approximate mean slope of the extinction (units=Mag per pc) - determined from literature
exp_scalefac = -1.215 #-1.215 #ln(Scale factor) kernel - approximate size of offset expected from the mean -  determined from literature and fit trials

#Pyro/ADAMW ELBO optimisation paramters
learning_rate = 0.01 #0.01 #0.001 #Scales the size of steps taken when optimising the free paramters with each itteration - Bigger lr means fewer itterations and less likely to get stuck in a local minimum, but if the lr is too small we'll need too many steps to reach a solution
learning_eps = 1e-8 #1e-6 #1e-8 #Epsilon value for ADAMW - To stabalise the steps taken and avoid large negative ELBO jumps
num_iter = 2500 #800 #500 #500 #Number of gradient descent steps - needs to run until the ELBO flattens out and the model has converged - we use a <1% change in ELBO over the last 10 itterations to stop
num_particles = 32 #32 #Number of sample graident calculations at each gradient descent step - power of two values work best
num_inducing = 1000 #500 #1000 #1000 #Number of inducing points to use - number of positions in the conditon grid used to optimise the density distribution. 
min_iter_GalPlane = 2000 #600 #300 #800 #1500 #1000 #2000 #500 #Minimum number of itterations to run before imposing stopping criterion
min_iter_HighLowLat = 1500 #500 #300 #600 #1000 #1000
stop_prcnt = 0.001 #0.01 #Percentage (fractional) change of the ELBO below which the training stops. We use 0.1%(0.001) 
stop_iter = 10 #Number of iterations to look back at to calculate the average ELBO change to impose the stopping criterion
snapshot_iter = 5 #At each snapshot_iter take a snapshot of the full GP so that it can be loaded and training can begin from their rather than fully restarting if needed

#Prediction set up
pred_chunk_size = 5000 #Number of prediction grid chunks to be used at one given time to predict on till the full predict grid is filled
pred_sample_size = 1000 #Number of density/extinction samples to draw from final gp to obtain the percentiles for prediction - We obtain the 16th, 50th, 84th percentiles



######### No varaibles need changing from here #####################
l_set1 = np.arange(l_sky_start, l_sky_end+1, l_chunk_size) #Get regions of l=18 x b=15 degrees where we can set 1 deg = 1 cell
l_set2 = np.arange(l_sky_start_overlap, l_sky_end, l_chunk_size) #To make sure we have overlapping cells to avoid edge effects 

b_set = np.rad2deg(np.arcsin(np.linspace( np.sin(np.deg2rad(b_lower)), np.sin(np.deg2rad(b_upper)), (2*b_step)+1 )))
b_set1 = b_set[0::2]  #Get regions of 18 x 18 degrees where we can set 1 deg = 1 cell -- (-30, 30, 15)
b_set2 = b_set[1::2] #To make sure we have overlapping cells to avoid edge effects -- (-22.5, 30, 15)

if first_d_chunk:
    d_min = d_min_first_d_chunk #pc
    d_max = d_max_first_d_chunk #pc
else:
    #Identify distance min and max which is fixed for each invidual distance chunk runs
    d_bounds_past = np.load("../preDistChunk/FullMW_merged_d_bounds.pkl.npy", allow_pickle=True) #Previous distance chunk boundary

    d_array_diff = np.absolute(d_bounds_past-d_min_new_approx) 
    nearest_index = d_array_diff.argmin()

    d_min = d_bounds_past[nearest_index] #Take the nearest value to the selected d_min we want from the past array to ensure distance boundaries overlaps properly
    d_max = d_min + diff_first_d_chunk #d_max is always d_min+990 which is what we decided at the very begining
    print("d_min new d chunk ==", d_min)
    print("d_max new d chunk ==", d_max)





    