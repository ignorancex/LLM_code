import sys
import torch
import torch.utils.data as data_utils
from tqdm import tqdm

from metat2.trainer.meta_trainer import MetaTrainer
from metat2.data.dataloader import ProstateDataset
from metat2.model.dmt.options.train_options import TrainOptions
import numpy as np

import os
# parse options
config = TrainOptions().parse()
# print options to help debugging
print(' '.join(sys.argv))

test_set = ProstateDataset(config.test_image_dir, config.croot_modality, config.sroot_modality)
test_loader = data_utils.DataLoader(test_set, batch_size=config.batchSize, shuffle=False, num_workers=config.nThreads)

meta_trainer = MetaTrainer(config)

dscs = []
for i, test_data_i in enumerate(tqdm(test_loader, desc="Testing")):
    dsc = meta_trainer.inference(test_data_i, 60)
    dscs += dsc
print(f'DSC mean: ', np.mean(dscs))
print(f'DSC std: ', np.std(dscs))
print('Testing was successfully finished.')