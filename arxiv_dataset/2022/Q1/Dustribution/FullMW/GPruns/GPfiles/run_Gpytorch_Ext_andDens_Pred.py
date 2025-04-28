import numpy as np
import pandas as pd
import time
import sys
import torch
from astropy import units as u
from astropy.coordinates import spherical_to_cartesian
import argparse

import global_seed
import random
random.seed(global_seed.seed)
torch.manual_seed(global_seed.seed)
np.random.seed(global_seed.seed)


from topLevel_routines import read_inputTable, reCalc_TrainGrid, checkSource_bounds, reCalc_PredGrid, reTrain_GP, rePredict_GP, past_CumExt_Calc
from plotStructure import plot_GP_Pred_Dens_Slices_AlongDist, plot_GP_Pred_Ext_Slices_AlongDist, plot_GP_Pred_ExtCumilative
from plotResiduals import plot_Res_ExtHist, plot_Res_Subtract_ExtHist_NormbyUnc
from plotPerformance import plot_PerformanceMetrics


"""

Run file for algorithm - 3D density map of Milky Way 

Uses latent varaible GPs combined with variational inference

All files executed by running this top level file 

Input data needed: lb coordinates, distances+uncertainties and extinctions+uncertainties for a set of stars in the region of interest; lb coordinates of the region of interest - a ~rectangle in l and b 

We also need to decide on the required resolution of the grid on which to trian the GP on - while we can train on high resolution grids the number of stars within grid cells will dictate the real resolution

We can predict on a higher resolution than the training grid for visualisation purposes - this does not mean we recover that resolution for the structure though!

It will also help to have an idea of the size of expected structure to be recoved in pc as well as mean density of region and variation from mean to inititate the GP hyperparameters

This script takes in the input region coordinates and resolutions and first builds the training and predicting grids stored for later. 
Then the training grid is used within the GP along with GP setup parameters to train the GP on the input extinctions 
Once the GP is trained it is used to predict on the prediction grid to out put the 16th,50th,84th percentiles of the extinction and density maps of our region which is saved and ready for visualisation

Code run time: GP Run Time 

"""


if __name__=="__main__":

    #Check commnad line arguments and update file with values such as filename, lmin/max, etc. 
    parser = argparse.ArgumentParser(description="3D dust mapping with Dustribution")
    parser.add_argument("catfile_name", metavar="source_cat", type=str, help="source catalogue name")
    parser.add_argument("l_min", metavar="l_min_val", type=float, help="l min in deg")
    parser.add_argument("l_max", metavar="l_max_val", type=float, help="l max in deg")
    parser.add_argument("b_min", metavar="b_min_val", type=float, help="b min in deg")
    parser.add_argument("b_max", metavar="b_max_val", type=float, help="b max in deg")
    parser.add_argument("d_min", metavar="d_min_val", type=float, help="d min in pc")
    parser.add_argument("d_max", metavar="d_max_val", type=float, help="d max in pc")

    parser.add_argument("first_d_chunk", metavar="first_d_chunk", choices=('True', 'False'), help='is this the first distance chunk')

    parser.add_argument("recalc_grid_train", metavar="recalc_grid_train", choices=('True', 'False'), help='should the training grid be recalculated')
    parser.add_argument("recalc_grid_pred", metavar="recalc_grid_pred", choices=('True', 'False'), help='should the predicting grid be recalculated')
    parser.add_argument("recheck_sourcebounds", metavar="recheck_sourcebounds", choices=('True', 'False'), help='is this the first distance chunk')
    parser.add_argument("retrain_gp", metavar="retrain_gp", choices=('True', 'False'), help='should the gp be retrained')
    parser.add_argument("repredict_gp", metavar="repredict_gp", choices=('True', 'False'), help='should the gp be repredicted')
    parser.add_argument("train_gpu", metavar="train_gpu", choices=('True', 'False'), help='Use the GPU for training')
    parser.add_argument("pred_gpu", metavar="pred_gpu", choices=('True', 'False'), help='Use the GPU for predicting')
    parser.add_argument("plot_gpu", metavar="plot_gpu", choices=('True', 'False'), help='Use the GPU for plotting')

    parser.add_argument("n_l_train", metavar="n_l_train", type=int, help="l train resolution")
    parser.add_argument("n_b_train", metavar="n_b_train", type=int, help="b train resolution")
    parser.add_argument("n_d_train", metavar="n_d_train", type=int, help="d train resolution")
    parser.add_argument("n_l_pred", metavar="n_l_pred", type=int, help="l pred resolution")
    parser.add_argument("n_b_pred", metavar="n_b_pred", type=int, help="b pred resolution")
    parser.add_argument("n_d_pred", metavar="n_d_pred", type=int, help="d pred resolution")

    parser.add_argument("scale_length_x", metavar="scale_length_x", type=float, help="gp scale length x")
    parser.add_argument("scale_length_y", metavar="scale_length_y", type=float, help="gp scale length y")
    parser.add_argument("scale_length_z", metavar="scale_length_z", type=float, help="gp scale length z")
    parser.add_argument("mean_ext_dens", metavar="mean_ext_dens", type=float, help="gp mean_ext_dens")
    parser.add_argument("exp_scalefac", metavar="exp_scalefac", type=float, help="gp exp_scalefac")

    parser.add_argument("learning_rate", metavar="learning_rate", type=float, help="gp pyro/adamw learning rate")
    parser.add_argument("learning_eps", metavar="learning_eps", type=float, help="gp pyro/adamw learning epsilon")
    parser.add_argument("num_iter", metavar="num_iter", type=int, help="gp pyro/adamw iteration number")
    parser.add_argument("num_particles", metavar="num_particles", type=int, help="gp pyro/adamw particles number")
    parser.add_argument("num_inducing", metavar="num_inducing", type=int, help="gp pyro/adamw inducing points number")
    parser.add_argument("min_iter", metavar="min_iter", type=int, help="gp pyro/adamw minimum number of iterations")
    parser.add_argument("stop_prcnt", metavar="stop_prcnt", type=float, help="gp pyro/adamw stop percentage variation of elbo/iter gradient")
    parser.add_argument("stop_iter", metavar="stop_iter", type=int, help="gp pyro/adamw stop variation of elbo/iter gradient along n iterations")
    parser.add_argument("snapshot_iter", metavar="snapshot_iter", type=int, help="gp pyro/adamw iteration gap for snapshotting")

    parser.add_argument("pred_chunk_size", metavar="pred_chunk_size", type=int, help="gp prediction cube size")
    parser.add_argument("pred_sample_size", metavar="pred_sample_size", type=int, help="gp number of samples to predict from")

    parser.add_argument("--resume_training", action='store_true', help="restart training from snapshot")


    args = parser.parse_args()
    print("Input details == ", args)


    start_time = time.time()
    print("code run start time = ", start_time)


    ###### Start of input parameters which need to be set by user ######

    #Read in input data from source Table
    input_filename = str(args.catfile_name)
    source_df = read_inputTable(input_filename)

    #Size of input source sample to condition the GP on - i.e: number of soures to fit the model extinctions on
    subsample_size = len(source_df) #Give number if we only want that number of sources from the full input table

    #To account for the edge of the Milky Way case where we go over the two edges 
    if args.l_min > args.l_max:
        #this should indicate that we are crossing zero in l, and therefore need to go from (l_min - 360) to l_max instead
        #First, we need to change the l coordinates in the input catalogue so they are in the correct range
        source_df['l'].loc[source_df['l'] > args.l_min] = source_df['l'].loc[source_df['l'] > args.l_min] - 360
        #then we update l_min to be in the correct range
        args.l_min = args.l_min - 360


    #Define condition grid boundaries in lbd coordinates - defines the map region and resolution
    #These numbers need to be changed to match the part of the sky which is being mapped 
    l_lower_train = args.l_min #lognitude l in degrees
    l_upper_train = args.l_max
    n_l_train = args.n_l_train #number of cells within boundaries - defines resolution of grid  

    b_lower_train = args.b_min #latitude b in degrees
    b_upper_train = args.b_max
    n_b_train = args.n_b_train

    d_min_train = args.d_min #Ditance d  in parsecs
    d_max_train = args.d_max
    n_d_train = args.n_d_train


    #Define predicting grid - enhance visualisation capabilities of grid
    #These numbers need to be changed to match the part of the sky which is being mapped - The l,b,d coordinates must match the train grid coordiantes above. 
    l_lower_pred = args.l_min
    l_upper_pred = args.l_max
    n_l_pred = args.n_l_pred 
   
    b_lower_pred = args.b_min
    b_upper_pred = args.b_max
    n_b_pred = args.n_b_pred #44*(14/18)
    
    d_min_pred = args.d_min
    d_max_pred = args.d_max
    n_d_pred = args.n_d_pred
    
    #Is this the first (0-xpc) dist chunk where we don't need to add the past dist chunk cum_ext during the training phase
    first_d_chunk = args.first_d_chunk == 'True'
    print("First d chunk status ==", str(first_d_chunk))

    #GP hyperparameter priors for the RBF Kernel - Setsup the staring point for the hyperparameters - they are allowed to be optimised with the GP
    scale_length_x = args.scale_length_x #Scale length - parsec units - #Approximated from literature given size of structure in region of interest dependent on how close/far we are from the source allowing us to recover the smaller/larger structure. If the region of interest is further away from us we will likely only be able to recover the large scale structure so we need a larger scale length to reflect the size of the large scale structure. But if we are close by we maybe sensitive to the small scale structure as well so we need a smaller scale lenght to reflect the smaller stucture sizes.  
    scale_length_y = args.scale_length_y
    scale_length_z = args.scale_length_z 
    mean_ext_dens = args.mean_ext_dens #-3.333 #Mean density - log10(Mags per pc) units - Approximate mean slope of the extinction (units=Mag per pc) - determined from literature
    exp_scalefac = args.exp_scalefac #-1.215 #ln(Scale factor) kernel - approximate size of offset expected from the mean -  determined from literature and fit trials


    #Pyro/ADAMW ELBO optimisation paramters
    learning_rate = args.learning_rate #Scales the size of steps taken when optimising the free paramters with each itteration - Bigger lr means fewer itterations and less likely to get stuck in a local minimum, but if the lr is too small we'll need too many steps to reach a solution
    learning_eps = args.learning_eps #Epsilon value for ADAMW - To stabalise the steps taken and avoid large negative ELBO jumps
    num_iter = args.num_iter #2500 #Number of gradient descent steps - needs to run until the ELBO flattens out and the model has converged - we use a <1% change in ELBO over the last 10 itterations to stop
    num_particles = args.num_particles #32 #Number of sample graident calculations at each gradient descent step - power of two values work best
    num_inducing = args.num_inducing #1000 #Number of inducing points to use - number of positions in the conditon grid used to optimise the density distribution. 
    min_iter = args.min_iter #1000 #Minimum number of itterations to run before imposing stopping criterion
    stop_prcnt = args.stop_prcnt #0.01 #Percentage (fractional) change of the ELBO below which the training stops. We use 0.1%(0.001) 
    stop_iter = args.stop_iter #Number of iterations to look back at to calculate the average ELBO change to impose the stopping criterion
    snapshot_iter = args.snapshot_iter #At each snapshot_iter take a snapshot of the full GP so that it can be loaded and training can begin from their rather than fully restarting if needed
   
    #Prediction set up
    pred_chunk_size = args.pred_chunk_size #Number of prediction grid chunks to be used at one given time to predict on till the full predict grid is filled
    pred_sample_size = args.pred_sample_size #Number of density/extinction samples to draw from final gp to obtain the percentiles for prediction - We obtain the 16th, 50th, 84th percentiles

    #Rerun statuses if we want to recreate grids or retrain GP from the stars
    recalc_grid_train = args.recalc_grid_train == 'True' #True/False #If we do/do not want to recalculate the training grid defined based on the lbd coordinates and n resolution above - must be recalculated if the coordinates or resolutions deifned above change
    recalc_grid_pred = args.recalc_grid_pred == 'True' #True/False #If we do/do not want to recalculate the predicting grid defined based on the lbd coordinates and n resolution above - must be recalculated if the coordinates or resolutions deifned above change
    recheck_sourcebounds = args.recheck_sourcebounds == 'True' #True/False #If we do/do not want to reheck source positions in source data frame to make sure their within our input coorindates and removing any which are not. Also re-precomputes some source indeces for integration optimisation
    retrain_gp = args.retrain_gp == 'True' #True/False  #If we do/do not want to rerun the full GP training based parameters given above - must be recalculated if any of the parameters above change except the prediciting grid parameters
    repredict_gp = args.repredict_gp == 'True' #True/False  #If we do/do not want to rerun the GP prediction based parameters given above - must be recalculated if any of the parameters above change
 
    #Run Algorithm on GPUs or CPUs
    train_gpu = args.train_gpu == 'True' #Set True for GPU run or set False for CPU run for the GP training
    pred_gpu = args.pred_gpu == 'True' #Set True for GPU run or set False for CPU run for the GP predicting. #We also need to set this to False if we're only plotting in a machine with No GPU
    plot_gpu = args.plot_gpu == 'True'  #Set True for plotting in the GPU or set False for CPU for plotting following GPU training especially if plotting is done outside a machine with a GPU
    
    
    ###### End of input parameters which need to be set by user ######


    #Setting if we want to train the full GP from the start or resume training of the GP from a snapshot
    #Will be set as commnad line arguments from NAME script when the run begins
    resume_training = False #Default is to not resume training from a snapshot and start the full process from scratch
    if args.resume_training:
        resume_training = True
        recalc_grid_train = False #We don't want to recalculate a grid everytime the GP training restarts from a previous snapshot
        recalc_grid_pred = False #We don't want to recalculate a grid everytime the GP training restarts from a previous snapshot
        recheck_sourcebounds = False #We don't want to reheck source positions in source data frame to make sure their within our input coorindates and removing any which are not. Also re-precomputes some source indeces for integration optimisation

    #Build/load train grid
    print("Train Grid Calculating")
    l_bounds_train, b_bounds_train, d_bounds_train, threeDGrid_train = reCalc_TrainGrid(recalc_grid_train, source_df, l_lower_train, l_upper_train, n_l_train, 
                                                                                            b_lower_train, b_upper_train, n_b_train, 
                                                                                            d_min_train, d_max_train, n_d_train)    


    #Checking source positions in source data frame to make sure their within our input coorindates and removing any which are not
    #Also precomputes some source indeces for integration optimisation in the dens_integTorch.py file
    print("Checking if all the sources fall within grid boundaries and removing any which are not")
    source_df, subsample_size, l_inds, b_inds, d_inds = checkSource_bounds(recheck_sourcebounds, source_df, subsample_size, l_bounds_train, b_bounds_train, d_bounds_train)


    #Build/load predict grid
    print("Pred Grid Calculating")
    l_bounds_pred, b_bounds_pred, d_bounds_pred, threeDGrid_pred = reCalc_PredGrid(recalc_grid_pred, source_df, l_lower_pred, l_upper_pred, n_l_pred, 
                                                                                        b_lower_pred, b_upper_pred, n_b_pred, 
                                                                                        d_min_pred, d_max_pred, n_d_pred)
    
    print("Grids calculated and/or loaded")
    # print("l_bounds_train ==", l_bounds_train)
    # print("l_bounds_pred ==", l_bounds_pred)
    # print("b_bounds_train ==", b_bounds_train)
    # print("b_bounds_pred ==", b_bounds_pred)
    print("d_bounds_train ==", d_bounds_train)
    print("d_bounds_pred ==", d_bounds_pred)
    # exit()
    
    

    print("Begin GP stage")

    #Is this the first (0-xpc) dist chunk where we don't need to add the past dist chunk cum_ext during the training phase
    if first_d_chunk:
        cum_ext_past = None
               
    else:
        #Obtain the correct slice of the past distance chunk Full MW map required for cum_ext addition for the new ext during the training phase
        cum_ext_past, l_inds_past, b_inds_past = past_CumExt_Calc(l_lower_train, l_upper_train, b_lower_train, b_upper_train, d_min_train, source_df)

    

    #Fully train/retrain the GP
    gp, condition_grid = reTrain_GP(retrain_gp, subsample_size, scale_length_x, scale_length_y, scale_length_z, mean_ext_dens, exp_scalefac, 
                                        learning_rate, learning_eps, num_iter, num_particles, num_inducing, min_iter, 
                                        stop_prcnt, stop_iter, snapshot_iter, resume_training,
                                        l_bounds_train, b_bounds_train, d_bounds_train, threeDGrid_train, source_df, train_gpu, pred_gpu, cum_ext_past = cum_ext_past)

    print("GP Model trained and/or loaded")

    
    #Use the GP to predict density and extinction on a chosen Grid
    gpy_dens_median, gpy_dens_16P, gpy_dens_84P, ext_med_cube, ext_16_cube, ext_84_cube = rePredict_GP(repredict_gp, pred_chunk_size, pred_sample_size, l_bounds_pred, b_bounds_pred, d_bounds_pred, threeDGrid_pred, gp, pred_gpu, plot_gpu, cum_ext_past = cum_ext_past)

    print("GP predicted and/or loaded")

    
   

    #Plot densities and extinctions for analysis
    print("Begin Plotting Ext and Density")

    #Plot training performance plots
    plot_PerformanceMetrics()

    #Plot residuals
    plot_Res_ExtHist(condition_grid, threeDGrid_pred, ext_med_cube, ext_16_cube, ext_84_cube)
    plot_Res_Subtract_ExtHist_NormbyUnc(condition_grid, threeDGrid_pred, ext_med_cube, ext_16_cube, ext_84_cube)

    #Plot final cumilative extinction 
    plot_GP_Pred_ExtCumilative(l_bounds_pred, b_bounds_pred, ext_med_cube)



    #Plot selected slices along distance of predicted extinction and density 
    plot_GP_Pred_Dens_Slices_AlongDist(n_l_pred, n_b_pred, n_d_pred, l_bounds_pred, b_bounds_pred, d_bounds_pred, gpy_dens_median)
    plot_GP_Pred_Ext_Slices_AlongDist(n_l_pred, n_b_pred, n_d_pred, l_bounds_pred, b_bounds_pred, d_bounds_pred, ext_med_cube)

    


    
    print("Code Run Time --- %s seconds ---" % (time.time() - start_time))
    
    























