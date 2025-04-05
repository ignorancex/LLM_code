#!/usr/bin/env python
# coding: utf-8

# ## IMPORTS
import os
import sys
import os
import pytorch_lightning as pl
from tqdm import tqdm
import itertools as it


# ## Define paths
#every path should start from the project folder:
project_folder = "../"

#Data folder should contain raw and preprocessed data
data_folder = os.path.join(project_folder,"data")
raw_data_folder = os.path.join(data_folder,"raw")

#Source folder should contain all the (essential) source code
source_folder = os.path.join(project_folder,"src")
#Code can be tried in notebooks, then moved into the src folder to be imported in different notebooks

#The out folder should contain all outputs: models, results, plots, etc.
out_folder = os.path.join(project_folder,"out")
models_folder = os.path.join(out_folder,"models")
results_folder = os.path.join(out_folder,"results")


# ## Import own code

#To import from src:
#attach the source folder to the start of sys.path
sys.path.insert(0, project_folder)
print(sys.path) # view the path and verify

#import from src directory
from src import util_data
from src import util_torch


# # MAIN

# ## Params

random_seed = 42

val_split = 0.2 #test is already divided
batch_size = 128
num_epochs = 1000

k_list = range(0,2)

interesting_to_remove = ['3TC', 'ABC', 'ATV', 'AZT', 'BIC', 'CAB', 'D4T', 'DDI', 'DOR', 'DRV', 'DTG', 'EFV', 'ETR', 'EVG', 'FPV', 'FTC', 'IDV', 'LPV', 'NFV', 'NVP', 'RAL', 'RPV', 'SQV', 'TDF', 'TPV'] 

all_subsets_dict = {}
for k in k_list:
  all_subsets_dict[k] = [list(comb) for comb in it.combinations(interesting_to_remove, k)]


# ## Loop

fit_nn = False #True

use_history = True #False
# dataname = f'HIV_database_'+["No",""][use_history]+'HistoryMuts'
expname = f'all_hiv_'+["No",""][use_history]+'History'

dataname = "dataset_TCE_Stanford_FS"
# expname = "stanford"

for k in k_list:
    all_subsets = all_subsets_dict[k]
    for subset_cols in tqdm(all_subsets):
        ## Prepare data
        ### Load data
        filename = f'../data/raw/{dataname}.csv'
        data = util_data.load_data_FC(raw_data_folder, filename, subset_cols)
        ### Split data
        data = util_data.split_data_FC(data,val_split,random_seed)
        ### Prepare loaders
        loaders = util_torch.prepare_loaders_FC(data, batch_size)
        
        subset_str = util_data.subset_cols_to_str(subset_cols, False)
        num_features = data["x_train"].shape[1]
        num_classes = 2
        experiment_name = os.path.join(expname,'FC','subset_'+subset_str)
        checkpoint_path = os.path.join(models_folder,experiment_name)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        class_weight = util_data.compute_class_weights(data["train"]["label"])
        if fit_nn:
            ##Train
            ###Initialize model
            pl.seed_everything(seed=random_seed)
            model = util_torch.FullyConnectedNN(num_features=num_features, embedding_sizes=[64,256], num_classes=num_classes, loss_weight=class_weight)

            trainer = util_torch.prepare_trainer(num_epochs,checkpoint_path)

            ### Fit
            trainer.fit(model, loaders["train"], loaders["val"])

        ##Load
        model, trainer, loaders = util_torch.reload_FC(checkpoint_path, num_features, num_classes, data, loaders, batch_size, class_weight)

        ##Save results
        df_res,df_res_div = util_data.compute_metrics(num_classes,model,data,loaders,trainer)
        util_data.save_results(results_folder, "NEW"+experiment_name, df_res, df_res_div)

        print("DONE")



