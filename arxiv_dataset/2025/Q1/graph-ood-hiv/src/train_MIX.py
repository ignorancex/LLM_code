#!/usr/bin/env python
# coding: utf-8

# ## IMPORTS
import os
import sys
import pytorch_lightning as pl
import itertools as it
from tqdm import tqdm


# ## Define paths
#every path should start from the project folder:
project_folder = "../"

#Data folder should contain raw and preprocessed data
data_folder = os.path.join(project_folder,"data")

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
from src import util_graph


# # MAIN

# ## Params
random_seed = 42

val_split = 0.2 #test is already divided
batch_size = 128
num_epochs = 1000
embedding_size = 128

k_list = list(range(0, 2))

interesting_to_remove = ['3TC', 'ABC', 'ATV', 'AZT', 'BIC', 'CAB', 'D4T', 'DDI', 'DOR', 'DRV', 'DTG', 'EFV', 'ETR', 'EVG', 'FPV', 'FTC', 'IDV', 'LPV', 'NFV', 'NVP', 'RAL', 'RPV', 'SQV', 'TDF', 'TPV'] 

all_subsets_dict = {}
for k in k_list:
  all_subsets_dict[k] = [list(comb) for comb in it.combinations(interesting_to_remove, k)]


# ## Loop
fit_nn = False #True

for graph_type in ["all","subset"]:
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
            #filename = '../data/raw/HIV_database_'+["No",""][use_history]+'HistoryMuts.csv'
            filename = dataname

            ## Prepare data
            ### Load data
            dataset = util_graph.load_data_MIX(data_folder, filename, graph_type, subset_cols)
            ### Split data & Prepare loaders
            num_val, index_traindata, loaders = util_torch.prepare_loaders_GNN(dataset, batch_size, val_split, random_seed)

            subset_str = util_data.subset_cols_to_str(subset_cols, False)
            for batch in loaders["train"]: break
            num_features = batch[0].shape[1]
            num_graph_features = batch[1].x.shape[1] if len(batch[1].x.shape) !=1 else 1
            num_classes = 2
            experiment_name = os.path.join(expname,'MIX',graph_type,'subset_'+subset_str)
            checkpoint_path = os.path.join(models_folder,experiment_name)

            class_weight = util_data.compute_class_weights(dataset["train"].raw_data["label"])
            if fit_nn:
                ##Train
                ###Initialize model
                pl.seed_everything(seed=random_seed)
                model = util_torch.LightningMIX(num_features = num_features, num_graph_features = num_graph_features, num_classes = embedding_size, real_num_classes = num_classes, loss_weight=class_weight, embedding_sizes=[64,256])

                trainer = util_torch.prepare_trainer(num_epochs,checkpoint_path)
                ### Fit
                trainer.fit(model, loaders["train"], loaders["val"])

            ##Load
            model, trainer, loaders = util_torch.reload_MIX(checkpoint_path, dataset, index_traindata, num_val, loaders, batch_size, num_features = num_features, num_graph_features = num_graph_features, num_classes = embedding_size, real_num_classes = num_classes, loss_weight=class_weight, embedding_sizes=[64,256])
            data = util_graph.return_data_from_graph_datasets(dataset, index_traindata, num_val)
            df_res,df_res_div = util_data.compute_metrics(num_classes,model,data,loaders,trainer)

            ##Save results
            util_data.save_results(results_folder,"NEW"+experiment_name,df_res,df_res_div)

            print("DONE")




