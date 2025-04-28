import sys
import torch
import torch.utils.data as data_utils
from tqdm import tqdm

from metat2.trainer.meta_trainer import MetaTrainer
from metat2.data.dataloader import ProstateDataset
from metat2.model.dmt.options.train_options import TrainOptions
import numpy as np

import logging
import os
# parse options
config = TrainOptions().parse()
# print options to help debugging
print(' '.join(sys.argv))

logging_path = os.path.join(config.meta_model_save_path, f'{config.name}_train.log')
def setup_logging(logging_path=logging_path):
    logging.basicConfig(filename=logging_path, level=logging.INFO, 
                        format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

setup_logging()

train_set = ProstateDataset(config.train_image_dir, config.croot_modality, config.sroot_modality)
vali_set = ProstateDataset(config.vali_image_dir, config.croot_modality, config.sroot_modality)
test_set = ProstateDataset(config.test_image_dir, config.croot_modality, config.sroot_modality)

train_loader = data_utils.DataLoader(train_set, batch_size=config.batchSize, shuffle=True, num_workers=config.nThreads)
vali_loader = data_utils.DataLoader(vali_set, batch_size=config.batchSize, shuffle=True, num_workers=config.nThreads)
test_loader = data_utils.DataLoader(test_set, batch_size=config.batchSize, shuffle=False, num_workers=config.nThreads)

meta_trainer = MetaTrainer(config)

total_epoch = config.niter + config.niter_decay

for epoch in tqdm(range(config.start_epoch, total_epoch), desc="Epochs"):
    
    for i, (train_data_i, vali_data_i) in enumerate(zip(tqdm(train_loader, desc="Training"), tqdm(vali_loader, desc="Validation"))):
        seg_loss, trans_loss = meta_trainer.train_one_step(train_data_i, vali_data_i)
        logging.info(f'Epoch {epoch}, Iteration{i}, : seg_loss = {seg_loss:.4f}, trans_loss = {trans_loss:.4f}')
    meta_trainer.update_trans_learning_rate(epoch)

    if epoch % config.save_epoch_freq == 0 or \
       epoch == total_epoch-1:
        print('saving the model at the end of epoch %d' %
              (epoch))
        meta_trainer.save_checkpoint(config.name, epoch)

print('Training was successfully finished.')