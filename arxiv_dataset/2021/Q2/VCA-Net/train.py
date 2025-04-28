import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary

import argparse
import logging
import os
import sys
import h5py

from torch import optim as optim
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from dataloader import ATLASDataset
from eval import eval_net
from losses import EMLLoss
from model import VCANet


checkpoint_dir = 'checkpoints/'

def train_net(net, dataset, device, epochs=200, batch_size=8, lr=0.001, val_percent=0.1, save_checkpoints=True, restore_checkpoint=False, restore_path="INTERRUPTED_model.pt"):
	
	dataset = dataset
	num_val = int(len(dataset) * val_percent)
	num_train = len(dataset) - num_val
	print(f"num_train, val: {num_train}, {num_val}")
	train, val = random_split(dataset, [num_train, num_val])
	train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
	val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

	writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
	global_step = 0

	logging.info(f'''Starting training:
		Epochs:			 {epochs}
		Batch size:		 {batch_size}
		Learning rate:	 {lr}
		Training size:	 {num_train}
		Validation size: {num_val}
		Checkpoints:	 {save_checkpoints}
		Device:			 {device.type}
	''')

	optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
	criterion1 = EMLLoss().cuda()
	criterion2 = nn.BCEWithLogitsLoss().cuda()

	for epoch in range(epochs):
		net.train()

		epoch_loss = 0
		with tqdm(total=num_train, desc=f'Epoch {epoch+1}/{epochs}', unit='image') as pbar:
			for batch in train_loader:
				imgs = batch['image']
				masks = batch['mask']

				imgs = imgs.to(device=device, dtype=torch.float32)
				mask_type = torch.float32
				masks = masks.to(device=device, dtype=mask_type)

				masks_pred = net(imgs)

				loss_eml = criterion1(masks_pred, masks)
				loss_bce = criterion2(masks_pred, masks)
				loss = loss_eml + loss_bce
				epoch_loss += loss.item()
				writer.add_scalar('Loss/eml_loss', loss_eml.item(), global_step)
				writer.add_scalar('Loss/bce_loss', loss_bce.item(), global_step)
				writer.add_scalar('Loss/total_loss', loss.item(), global_step)

				pbar.set_postfix(**{'loss (batch)': loss.item()})

				optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_value_(net.parameters(), 0.1)
				optimizer.step()
	
				pbar.update(imgs.shape[0])

				writer.add_images('images', imgs, global_step)
				writer.add_images('masks/true', masks, global_step)
				writer.add_images('masks/pred', masks_pred > 0.5, global_step)
				writer.add_images('masks/pred_whole', masks_pred, global_step)

				global_step += 1
				if global_step % (num_train // (10 * batch_size)) == 0:
					for tag, value in net.named_parameters():
						tag = tag.replace('.', '/')
						writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
						if value.grad is not None:
							writer.add_histogram('grads/' + tag, value.grad.cpu().numpy(), global_step)
					
					val_score = eval_net(net, val_loader, device)
					scheduler.step(val_score)
					writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

					logging.info('Validation Dice Coeff: {}'.format(val_score))
					writer.add_scalar('Dice/Validate', val_score, global_step)

			if save_checkpoints:
				try:
					os.mkdir(checkpoint_dir)
					logging.info('Checkpoint directory created')
				except OSError:
					pass
				torch.save(net.state_dict(), checkpoint_dir + f'CP_epoch{epoch+1}.pth')
				logging.info(f'Checkpoint {epoch+1} saved!')

		writer.close()


if __name__ == '__main__':
	logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logging.info(f'Using device {device}')

	h5_train = h5py.File('../../../Datasets/ATLAS_R1.1/train_210.h5', 'r')
	train_dataset = ATLASDataset(h5_train, 189, 43281)

	model = VCANet(in_channels=1, out_channels=1)

	model.to(device=device)
	summary(model, (1, 224, 192))

	try:
		train_net(model, train_dataset, device=device)
	except KeyboardInterrupt:
		torch.save(model.state_dict(), 'INTERRUPTED_VCA.pth')
		logging.info('Interrupted model saved')
		try:
			sys.exit(0)
		except SystemExit:
			os._exit(0)
