from datasets import load_dataset
from pytorch_lightning import Trainer, seed_everything
from utils import *
import argparse
import warnings
import pickle
import os
import torch
from calibration_utils import *

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser(description='Data setup')
parser.add_argument('ds_name', type=str,
                    help='Dataset name (news/agnews/imdb)')
parser.add_argument('size', type=int,
                    help='Train data size')
parser.add_argument('data_dir', type=str,
                    help='Directory location to save data')
parser.add_argument('result_dir', type=str,
                    help='Directory location to save results')

args = parser.parse_args()

size = args.size
ds_name = args.ds_name
filename = ds_name + str(size) + ".hf"
data_dir = args.data_dir
result_dir = args.result_dir

# -------------------------------------------------------------------------------------------------------------------#
# load dataset
if ds_name == 'news':
    data = load_dataset("SetFit/20_newsgroups")
elif ds_name == 'agnews':
    data = load_dataset("ag_news")
elif ds_name == 'imdb':
    data = load_dataset("imdb")
    
# data = load_dataset("huggingface_dataset")  # To load other huggingface datasets

# saving dataset variation
create_data_split(data, size, ds_name, data_dir)

# setup input data
seed_everything(42)

dm = finetuning_data(model_name_or_path="bert-base-cased", data_name=ds_name, filename=filename,
                     data_dir=data_dir)
dm.setup("fit")

print("Train: {}".format(len(dm.dataset['train'])))
print("Test: {}".format(len(dm.dataset['test'])))

# -------------------------------------------------------------------------------------------------------------------#
# define model parameters
model = finetuner(model_name_or_path="bert-base-cased", num_labels=dm.num_labels)

# define training hyperparameters
trainer = Trainer(
    max_epochs=1,
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None)

# model training
trainer.fit(model, datamodule=dm)
print("Fine-tuning done!")

# model eval
metrs = trainer.test(model, datamodule=dm)
outputs = trainer.predict(model, datamodule=dm, return_predictions=True)

print(metrs)

soft_preds = [y for x in outputs for y in x[0]]
preds = [y for x in outputs for y in x[1]]
labels = [y for x in outputs for y in x[2]]

dir_path = result_dir
# Check whether the specified path exists or not
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

print("Saving prediction probabilities...")

filename = ds_name + str(size)
preds_file = 'predicted_labels' + filename + '.pickle'
soft_preds_file = 'predict_probs' + filename + '.pickle'
labels_file = 'labels' + filename + '.pickle'

preds_path = os.path.join(dir_path, preds_file)
soft_preds_path = os.path.join(dir_path, soft_preds_file)
labels_path = os.path.join(dir_path, labels_file)

# save
with open(soft_preds_path, 'wb') as handle:
    pickle.dump(soft_preds, handle)

with open(preds_path, 'wb') as handle:
    pickle.dump(preds, handle)

with open(labels_path, 'wb') as handle:
    pickle.dump(labels, handle)

print("Saved prediction probabilities!")
# -------------------------------------------------------------------------------------------------------------------#


