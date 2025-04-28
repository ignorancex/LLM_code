import numpy as np
import torch
import pyro
import tqdm
import time
from memory_profiler import profile
import pickle
try:
    from scipy.stats import median_absolute_deviation as mad
except ImportError:
    from scipy.stats import median_abs_deviation as mad
from glob import glob
import re
import os
import subprocess

import global_seed
import random
random.seed(global_seed.seed)
torch.manual_seed(global_seed.seed)
np.random.seed(global_seed.seed)

from dens_integTorch import integ_allLoS
from gpyClass_Latent import LatentDensityGPModel





# A quick helper function for getting smoothed percentile values from samples
#Taken from https://docs.gpytorch.ai/en/v1.1.1/examples/07_Pyro_Integration/Cox_Process_Example.html [11]
#Gives the 16th , 50th and 84th percentiles - these will be dens +/- uncertianties in the end
def percentiles_from_samples(samples, pred_gpu, percentiles=[0.16, 0.5, 0.84]):
    num_samples = samples.size(0)
    samples = samples.sort(dim=0)[0]

    # Get samples corresponding to percentile
    percentile_samples = [samples[int(num_samples * percentile)] for percentile in percentiles]

    # Smooth the samples
    kernel = torch.full((1, 1, 5), fill_value=0.2)

    if pred_gpu:
        kernel = kernel.double().cuda()

    percentiles_samples = [
        torch.nn.functional.conv1d(percentile_sample.view(1, 1, -1), kernel, padding=2).view(-1)
        for percentile_sample in percentile_samples
    ]

    return percentile_samples



@profile
#Train and Condition the GP grid on only the given subsample
#The GP itself must work in Cartesian Coords!
def GP_Train_andCondition(scale_length_x, scale_length_y, scale_length_z, mean_ext_dens, exp_scalefac, 
                            learning_rate, learning_eps, num_iter, num_particles, num_inducing, min_iter, 
                            stop_prcnt, stop_iter, snapshot_iter, resume_training,
                            l_bounds_train, b_bounds_train, d_bounds_train, threeDGrid_train, condition_grid, train_gpu, cum_ext_past = None):



    #Defining the varaible required for Conditioning (eval mode) and Training (in train mode which will use the hp boundaries given in the class)
    #The conditioning data - i.e the value at which the tranformed latent variables (i.e: extinctions) are condictioned on
    condition_coords = torch.tensor(condition_grid[["coords_cartx", "coords_carty", "coords_cartz"]].values, dtype=torch.float) #Define the coordinates for the independent varaible i.e coords
    condition_ext_mean = torch.tensor(condition_grid["Ext_p50"].values, dtype=torch.float) #We define the dependent vairiable i.e density mean
    condition_ext_unc = torch.tensor(condition_grid["Ext_p50_err"].values, dtype=torch.float) #We define the dependent vairiable i.e density unc (std.dev)

    condition_ext = torch.distributions.Normal(condition_ext_mean, condition_ext_unc).rsample() #We need to provide the density and the dens unc for the training densities


    #Parameters needed for the GP model input
    source_dists = condition_grid["dist_p50"].to_numpy()
    source_l = condition_grid["l"].to_numpy()
    source_b = condition_grid["b"].to_numpy()
    threeDGrid_train_l = threeDGrid_train["pol_l"].to_numpy()
    threeDGrid_train_b = threeDGrid_train["pol_b"].to_numpy()

    l_ind = condition_grid["l_ind"].to_numpy()
    print(l_ind, l_ind.dtype)
    b_ind = condition_grid["b_ind"].to_numpy()
    d_ind = condition_grid["d_ind"].to_numpy()

    if cum_ext_past is not None:
        l_ind_past = condition_grid["l_ind_past"].to_numpy()
        b_ind_past = condition_grid["b_ind_past"].to_numpy()
    else:
        l_ind_past = None
        b_ind_past = None

    #Selecting inducing points
    inducing_point_indeces = torch.randperm(condition_coords.size()[0])[:num_inducing] #Randomly select the number of inducing points from the entire table
    inducing_points = condition_coords[inducing_point_indeces,:]
    torch.set_printoptions(profile="full")
    print("inducing_points before training/learning==", inducing_points)
    torch.set_printoptions(profile="default")
    print("inducing_points size==", inducing_points.size())



    #Create the GP with latent variable
    gp = LatentDensityGPModel(source_dists, source_l, source_b, l_bounds_train, b_bounds_train, d_bounds_train, threeDGrid_train_l, threeDGrid_train_b, l_ind, b_ind, d_ind,
                                inducing_points, train_gpu, name_prefix="density_gp_model", cum_ext_past = cum_ext_past, l_ind_past = l_ind_past, b_ind_past = b_ind_past)





    #Defining the set of Hyperparameters to be used in GP (in this case we use scalelength = 2 and mean model = 1 same as we did in Celerite)
    hypers = {
        "mean_module.constant": torch.tensor(mean_ext_dens), #Mean of the GP --> Mean Density in log10(density)
        "covar_module.raw_outputscale": torch.tensor(exp_scalefac), #Scale factor HP for the exp kernel
        "covar_module.base_kernel.raw_lengthscale": torch.tensor([scale_length_x, scale_length_y, scale_length_z]), #For now we use the same scale length for all three exp kernels in the three axe
            }

    #Feeds the Hyperparaters defined above into GP as the initial HPs
    gp.initialize(**hypers) #** => tells python to unpack the dictionary and use each value in the dictionary as an input keyword argument 
    

    #Put GP into train mode for optimization
    gp.train()

    #The coordinates where the latent variables will be infered on - where the GP is trained on to learn the densities (at these given coords)
    train_coords = torch.tensor(threeDGrid_train[["cart_x", "cart_y", "cart_z"]].values, dtype=torch.float, requires_grad = True) #Convert the pandas df to a pytorch compatible data strutcure
    

    #Empty Arrays to hold iteration information for plotting performance plots later
    elbo_list = []
    lsx_list = []
    lsy_list =[]
    lsz_list = []
    scalefac_list = []
    meanDens_list = []

    if train_gpu: #Run on GPU (Don't need a cpu version as the varaible are already stored in CPU)

        print("Is a GPU available? ", torch.cuda.is_available())

        ### SEND EVERYTHING TO THE GPU ###
        train_coords = train_coords.double().cuda()
        condition_coords = condition_coords.double().cuda()
        condition_ext = condition_ext.double().cuda()
        condition_ext_mean = condition_ext_mean.double().cuda()
        condition_ext_unc = condition_ext_unc.double().cuda()
        gp = gp.double().cuda()

    # Use the adam+elbo (grad.descent) optimizer in pyro 
    # Here we use AdamW (instead of simple Adam): https://pytorch.org/docs/master/optim.html#torch.optim.Adagrad
    #CLASStorch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)   
    # Here we can train from the start or load a previous snapshot and train from the end of that snapshot - all the information required to restart the gp training from the previous snapshot is saved in the snapshot it self
    def train(elbo_list, lsx_list, lsy_list, lsz_list, scalefac_list, meanDens_list, gp, lr=learning_rate, resume_training=resume_training, min_iter = min_iter, num_iter=num_iter, learning_eps=learning_eps, num_particles=num_particles): 

        optimizer = pyro.optim.AdamW({"lr":lr, "eps":learning_eps})
        loss = pyro.infer.Trace_ELBO(num_particles=num_particles, vectorize_particles=False, retain_graph=True)
        infer = pyro.infer.SVI(gp.model, gp.guide, optimizer, loss=loss)
        loader = tqdm.tqdm(range(num_iter))

        #If we want to resume training from a snapshot it will overwrite the above setup parameters and keep going
        if resume_training:

            #load the most recent Snapshot
            snapshot = load_Snapshot()
            gp.load_state_dict(snapshot["gp_state_dict"])

            if train_gpu: 
                gp = gp.cuda() #For GPU - Making absolutely sure that the re-initialised GP is sent to the GPU

            optimizer.set_state(snapshot["optimizer_state_dict"])
            loss = snapshot["loss"]
            loader = tqdm.tqdm(range(snapshot["iteration"], num_iter))
            elbo_list = snapshot["elbo_list"]
            lsx_list = snapshot["lsx_list"]
            lsy_list = snapshot["lsy_list"]
            lsz_list = snapshot["lsz_list"]
            scalefac_list = snapshot["scalefac_list"]
            meanDens_list = snapshot["meanDens_list"]
            print("Resuming training from snapshot")
        
        #Regular training loop
        for i in loader:
            loss = infer.step(train_coords, condition_ext_mean, condition_ext_unc)
            loader.set_postfix(loss=loss)
            print("Iter %d/%d - Loss: %.3f   lengthscale_x: %.3f lengthscale_y: %.3f lengthscale_z: %.3f scalefactor: %.3f meanDens: %.3f " % (
                    i + 1, num_iter, loss, #ELBO==LOSS in our case
                    gp.covar_module.base_kernel.raw_lengthscale[:,0].item(),
                    gp.covar_module.base_kernel.raw_lengthscale[:,1].item(),
                    gp.covar_module.base_kernel.raw_lengthscale[:,2].item(),
                    gp.covar_module.raw_outputscale.item(),
                    gp.mean_module.constant.item()
                ))

            #Save iteration information for plotting performance plots later
            elbo_list.append(loss) #ELBO
            lsx_list.append(gp.covar_module.base_kernel.raw_lengthscale[:,0].item()) #Scale length X
            lsy_list.append(gp.covar_module.base_kernel.raw_lengthscale[:,1].item()) #Scale length Y
            lsz_list.append(gp.covar_module.base_kernel.raw_lengthscale[:,2].item()) ##Scale length X
            scalefac_list.append(gp.covar_module.raw_outputscale.item()) #Scale Factor
            meanDens_list.append(gp.mean_module.constant.item()) #Mean Dens

            #Imposing stop criteria - stop after ELBO stop changing by <1% over the last 10 iterations
            #if i > 100: to ensure it"s run for atleast the given minimum number of iterations and doesn"t stop right at the begining.
            #The min number of iterations needs to be larger than the number of iterations averaged over for the stopping criteria
            #To measure the change we use median absolute diviation/median is <1%
            if i > min_iter and (mad(elbo_list[-stop_iter:]))/np.median(elbo_list[-stop_iter:]) < stop_prcnt:
                print("ELBO converged, ending training now")
                snapshot_file = "Snapshot_" + str(i) + ".out" #Snapshot filename - snapshot file saved for the given iteration as a torch file and needs to be reloaded with torc
                save_Snapshot(i, gp, optimizer, loss, snapshot_file, elbo_list, lsx_list, lsy_list, lsz_list, scalefac_list, meanDens_list) #If the stopping criteria is reached the snapshot is saved at the end of training
                break #Abort training if stopping criteria is reached

            #Another possible case where we need to stop training: If the ELBO has become pathological
            if elbo_list[-1] < 0 or np.abs(elbo_list[-1])/elbo_list[0] > 1e3: #This tests for two cases: 1) the elbo is negative, or 2) the elbo has diverged by 3 orders of magnitude from the initial value, which probably means it has gone a bit crazy
                print("Error 74: ELBO diverging or jumping to negative values")
                print("Dustribution will now abort as fit has failed, restart")
                if not os.path.isdir("Old_Snapshots"):
                    subprocess.call("mkdir Old_Snapshots", shell=True)
                #subprocess.call("rm -rf Snapshot_*.out", shell=True) #delete current run snapshots
                subprocess.call("mv Snapshot_*.out Old_Snapshots/", shell=True)
                import sys
                sys.exit(74) #Use sys.exit to allow bash/slurm to access the exit status so it can be automatically restarted with a new seed


            #Making a Snapshot - Checks the current iteration number, finds the remainder compared to the number of iterations afterwhich a snapshot is taken and if the checkpoint is reached it makes a snapshot
            if i % snapshot_iter == 0: #% is the modulo or the remainder operator
                snapshot_file = "Snapshot_" + str(i) + ".out" #Snapshot filename - snapshot file saved for the given iteration as a torch file and needs to be reloaded with torch
                save_Snapshot(i, gp, optimizer, loss, snapshot_file, elbo_list, lsx_list, lsy_list, lsz_list, scalefac_list, meanDens_list) 

        return elbo_list, lsx_list, lsy_list, lsz_list, scalefac_list, meanDens_list

    elbo_list, lsx_list, lsy_list, lsz_list, scalefac_list, meanDens_list = train(elbo_list, lsx_list, lsy_list, lsz_list, scalefac_list, meanDens_list, gp)

    #Save iteration information for plotting performance plots later
    with open("Iteration_Info.pkl", "wb") as f:
        pickle.dump(elbo_list, f)
        pickle.dump(lsx_list, f)
        pickle.dump(lsy_list, f)
        pickle.dump(lsz_list, f)
        pickle.dump(scalefac_list, f)
        pickle.dump(meanDens_list, f)

    return gp




@profile
#Predict using the previously trained GP model
def GP_Predict(chunk_size, pred_sample_size, l_bounds_pred, b_bounds_pred, d_bounds_pred, threeDGrid_pred, gp, pred_gpu, cum_ext_past = None):

    #Puts the GP into the evaluate mode rather than training mode
    gp.eval()


    #This steps uses the trained gp (trained on only the subsample) to produce the prediction with the input test data sample which in this case is our full grid
    #If we want a new grid for prediction then we need to input it here! 
    pred_coords = torch.tensor(threeDGrid_pred[["cart_x", "cart_y", "cart_z"]].values, dtype=torch.float, requires_grad = True) #Convert the pandas df to a pytorch compatible data strutcure

    if pred_gpu: #Predict on GPU
        pred_coords = pred_coords.double().cuda()
        gp.cuda()
    else:
        gp.cpu() #To make gp CPU compatible if Pred in CPU. Only needed if training is done in GPU and Pred is in CPU. If training was already done on CPU this will have no effect. 

    gp_DensPred_start_time = time.time()
    print("GP Dens Pred start time = ", gp_DensPred_start_time)

    #We loop over chunks of the predicting grid to make sure there is no memory issues
    #This way we can even predict on signle coord points at a time
    with torch.no_grad(): #Removes the grad info since we don"t need it - our latent function now is the density so we don"t need grads
        
        i_start = 0
        i_end = i_start + chunk_size
        print("pred_coords size==", pred_coords.size())

        while i_start < pred_coords.size()[0]:

            try:
                inf_dens = True #Set this to true now so the loop will execute at least once
                while inf_dens: #we keep sampling until all samples are finite after exponentiation, as otherwise infs get propagated into nans when taking weighted quantiles in merging
                    function_dist = gp(pred_coords[i_start:i_end,:])  #Take a distribution of the latent varaibles/samples
                    dens_samples = 10**function_dist(torch.Size([pred_sample_size])) #sample from that distribution and transform to the function domain we want - in our case log10(dens) to dens
                    inf_dens = torch.any(torch.isinf(dens_samples))
                lowerP, median, upperP = percentiles_from_samples(dens_samples, pred_gpu) #Gives 16th, 50th, 84th percentiles
            except ValueError:
                inf_dens = True #Set this to true now so the loop will execute at least once
                while inf_dens: #we keep sampling until all samples are finite after exponentiation, as otherwise infs get propagated into nans when taking weighted quantiles in merging
                    function_dist = gp(pred_coords[i_start:,:])  #Take a distribution of the latent varaibles/samples
                    dens_samples = 10**function_dist(torch.Size([pred_sample_size])) #sample from that distribution and transform to the function domain we want - in our case log10(dens) to dens
                    inf_dens = torch.any(torch.isinf(dens_samples))
                lowerP, median, upperP = percentiles_from_samples(dens_samples, pred_gpu) 

            try:
                dens_samples_all = torch.cat((dens_samples_all, dens_samples), 1) 
                gpy_dens_median = torch.cat((gpy_dens_median, median), 0) #Median (50th Percentile)
                gpy_dens_16P = torch.cat((gpy_dens_16P, lowerP), 0) #16th Percentile
                gpy_dens_84P = torch.cat((gpy_dens_84P, upperP), 0) #84th Percentile
            except UnboundLocalError:
                dens_samples_all = dens_samples
                gpy_dens_median = median
                gpy_dens_16P = lowerP
                gpy_dens_84P = upperP

            i_start = i_end
            i_end += chunk_size

            print("dens_samples_all size",dens_samples_all.size())

    print("GP Dens Predict Run Time --- %s seconds ---" % (time.time() - gp_DensPred_start_time))
    print("Dens Pred completed Begin Ext Integration")


    gp_ExtPred_start_time = time.time()
    print("GP Ext Pred start time = ", gp_ExtPred_start_time)


    #Integrate all the density samples to get Extinctions percentiles
    extinction_hypercube = integ_allLoS(l_bounds_pred, b_bounds_pred, d_bounds_pred, threeDGrid_pred["pol_l"].to_numpy(), threeDGrid_pred["pol_b"].to_numpy(), dens_samples_all.cpu().numpy(), n_samples=pred_sample_size)


    #If this is anything above the first distance slice then we need to add the foreground extinction
    if cum_ext_past is not None:
        extinction_hypercube = extinction_hypercube + np.expand_dims(cum_ext_past, axis=(0, -1)) #unsqueeze(0).unsqueeze(-1)


    ext_med_cube = np.percentile(extinction_hypercube, 50, axis = 0)
    ext_16_cube = np.percentile(extinction_hypercube, 16, axis = 0)
    ext_84_cube = np.percentile(extinction_hypercube, 84, axis = 0)

    #### If extinction is larger than 20 mag reTrain the GP with new Seed ###
    ext_max = np.max(ext_med_cube)
    print("ext_max ==", ext_max)
    if ext_max > 20: #This tests for two cases: 1) the elbo is negative, or 2) the elbo has diverged by 3 orders of magnitude from the initial value, which probably means it has gone a bit crazy
            print("Error 84: Max extinction greater than 20 mag")
            print("Dustribution will now abort as fit has failed, restart")
            subprocess.call("rm -rf Snapshot_*.out", shell=True) #delete current run snapshots
            import sys
            sys.exit(84) #Use sys.exit to allow bash/slurm to access the exit status so it can be automatically restarted with a new seed

    print("GP Ext Predict Run Time --- %s seconds ---" % (time.time() - gp_ExtPred_start_time))
    print("Extinction Integration Completed")

    #Save the full gp samples for merging all the Milky Way chunks to get the full grids to get the entire Milky Way later.  
    torch.save(dens_samples_all, "gpy_dens_samples_all.out")
    np.save("ext_all_cube.pkl", extinction_hypercube, allow_pickle=True)

    return gpy_dens_median, gpy_dens_16P, gpy_dens_84P, ext_med_cube, ext_16_cube, ext_84_cube


#Function to Load a Saved GP Model
def load_GPmodel(num_inducing, condition_grid, threeDGrid_train, l_bounds_train, b_bounds_train, d_bounds_train, train_gpu, pred_gpu, gp_filename, cum_ext_past = None):
    ### Predict from the last run snapshot if the code failed to complete but we have several snapshots that can be used: e.g: run out of seeds ########
    try:
        if pred_gpu: #To load the pre-trained GP trained in a GPU which is saved as a GPU tensor, for predicting in a GPU we don't need to do anything 
            gp_dict = torch.load(gp_filename)
        else: #If we want to pre in a CPU from GP trained on a GPU where its saved as a GPU tensor we need to do the following
            gp_dict = torch.load(gp_filename, map_location=torch.device("cpu")) #to remap the GPU tensor saved GP into a CPU tensor
    except FileNotFoundError: #If the filename supplied doesn't exist, then we're probably trying to predict on a model which aborted part-way through training without resuming the training
        last_snapshot = load_Snapshot()
        #Save iteration information for plotting performance plots later
        with open("Iteration_Info.pkl", "wb") as f:
            pickle.dump(last_snapshot['elbo_list'], f)
            pickle.dump(last_snapshot['lsx_list'], f)
            pickle.dump(last_snapshot['lsy_list'], f)
            pickle.dump(last_snapshot['lsz_list'], f)
            pickle.dump(last_snapshot['scalefac_list'], f)
            pickle.dump(last_snapshot['meanDens_list'], f)
        gp_dict = last_snapshot["gp_state_dict"]

    source_dists = condition_grid["dist_p50"].to_numpy()
    source_l = condition_grid["l"].to_numpy()
    source_b = condition_grid["b"].to_numpy()
    condition_coords = torch.tensor(condition_grid[["coords_cartx", "coords_carty", "coords_cartz"]].values, dtype = torch.float) 
    condition_ext_mean = torch.tensor(condition_grid["Ext_p50"].values, dtype = torch.float) 
    condition_ext_unc = torch.tensor(condition_grid["Ext_p50_err"].values, dtype = torch.float)
    
    threeDGrid_train_l = threeDGrid_train["pol_l"].to_numpy()
    threeDGrid_train_b = threeDGrid_train["pol_b"].to_numpy()
    inducing_points = condition_coords[:num_inducing,:]

    l_ind = condition_grid["l_ind"].to_numpy()
    b_ind = condition_grid["b_ind"].to_numpy()
    d_ind = condition_grid["d_ind"].to_numpy()

    if cum_ext_past is not None:
        l_ind_past = condition_grid["l_ind_past"].to_numpy()
        b_ind_past = condition_grid["b_ind_past"].to_numpy()
    else:
        l_ind_past = None
        b_ind_past = None

    if pred_gpu: #To load the pre-trained GP trained in a GPU which is saved as a GPU tensor, for predicting in a GPU we don't need to do anything 
        gp = LatentDensityGPModel(source_dists, source_l, source_b, l_bounds_train, b_bounds_train, d_bounds_train, threeDGrid_train_l, threeDGrid_train_b, 
                                    l_ind, b_ind, d_ind, inducing_points, train_gpu, name_prefix="density_gp_model", cum_ext_past = cum_ext_past, l_ind_past = l_ind_past, b_ind_past = b_ind_past) 
    else: #If we want to pre in a CPU from GP trained on a GPU where its saved as a GPU tensor we need to do the following
        gp = LatentDensityGPModel(source_dists, source_l, source_b, l_bounds_train, b_bounds_train, d_bounds_train, threeDGrid_train_l, threeDGrid_train_b, 
                                    l_ind, b_ind, d_ind, inducing_points, train_gpu=pred_gpu, name_prefix="density_gp_model", cum_ext_past = cum_ext_past, l_ind_past = l_ind_past, b_ind_past = b_ind_past) 


    gp.load_state_dict(gp_dict)
    if pred_gpu:
        gp.double().cuda()
    print("Printing re-read GP:")
    print(gp)
    for k,v in gp.state_dict().items():
        print(k,'\n',v, '\t', v.type())

    return gp


#Saving snapshot at the given checkpoint 
def save_Snapshot(iteration, gp, optimizer, loss, snapshot_file, elbo_list, lsx_list, lsy_list, lsz_list, scalefac_list, meanDens_list):
    
    #Create and save the most recent snapshot
    torch.save({"iteration":iteration, "gp_state_dict":gp.state_dict(), "optimizer_state_dict": optimizer.get_state(), "loss":loss, "elbo_list":elbo_list, "lsx_list":lsx_list, "lsy_list":lsy_list, "lsz_list":lsz_list, "scalefac_list":scalefac_list, "meanDens_list":meanDens_list}, 
                    snapshot_file)


    # #Delete snapshots older than the last five saved (we can save more if we want to)
    # snapshots = glob("Snapshot_*.out") #Pickout all snapshot files to figure out which iteration they are from - gives a list of the snapshot file names

    # number_of_snapshots_tosave = 2 #Delete all snapshots except the five highest (last) iterations

    # if len(snapshots) > number_of_snapshots_tosave:
    #     snapshot_iters = np.array([ np.int(re.split("[_.]", snapshot)[1]) for snapshot in snapshots])
    #     snapshot_list_ascend = list(np.array(snapshots)[np.argsort(snapshot_iters)]) #Arrange the snapshot list by asending iteration number
    #     for f in snapshot_list_ascend[:-number_of_snapshots_tosave]:
    #         os.remove(f) #Delete all snapshots except the five highest (last) iterations


    return




#Loading snapshot at the given checkpoint 
def load_Snapshot():

    #Pickout all snapshot files to figure out which iteration they are from - gives a list of the snapshot file names
    snapshots = glob("Snapshot_*.out")

    #Identify the most recent snapshot in the directory which is the highest iteration numbered snapshot
    snapshot_iters = [int(re.split("[_.]", snapshot)[1]) for snapshot in snapshots] #split the file name to idenitfy the snapshot numbers to get the iteration number
    most_recent_file = snapshots[np.argmax(snapshot_iters)]
    
    return torch.load(most_recent_file) #, np.argmax(snapshot_iters), most_recent_file












