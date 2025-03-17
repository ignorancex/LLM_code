import os
import time
import argparse
import json
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from env import AttrDict, build_env
from dataset import Dataset, get_dataset_filelist
from model import C2C
from utils import scan_checkpoint, load_checkpoint, save_checkpoint
import torch.nn as nn
import pkbar
from sklearn.metrics import roc_auc_score
from scheduler import CosineAnnealingWarmUpRestarts
import fairseq # Used UST branch from fairseq

torch.backends.cudnn.benchmark = True



def train(rank, a, h):

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))
    generator = C2C(C=128).to(device)
    scaler_g = torch.cuda.amp.GradScaler()
    # Finetuned checkpoint
    cp_path = "/home/woojinchung/codefile/BHI_2023_challenge/cp_w2v2/checkpoint.pt"
    w2v2, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
    w2v2 = w2v2[0].to(device)

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):  
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        

    steps = 0
    if cp_g is None :
        state_dict_g = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        generator.load_state_dict(state_dict_g['generator'])
        steps = state_dict_g['steps'] + 1
        last_epoch = state_dict_g['epoch']

    optim_g = torch.optim.Adam(generator.parameters(), lr=h.learning_rate, betas=(h.adam_b1, h.adam_b2))

    if state_dict_g is not None:
        optim_g.load_state_dict(state_dict_g['optim_g'])

    scheduler_g  = CosineAnnealingWarmUpRestarts(optim_g, 
                                                        T_0=1, 
                                                        T_mult=2, 
                                                        eta_max=3e-4, 
                                                        T_up=0, 
                                                        gamma=0.99)

    traindata, valdata = get_dataset_filelist(a)
    
    trainset = Dataset(traindata,
                            h.sampling_rate,
                            shuffle=True, device=device,
                            train=True)

# Reorder the trainset
    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=True,
                                sampler=None,
                                batch_size=h.batch_size,
                                pin_memory=True,
                                drop_last=True)

    if rank == 0:
        validset = Dataset(valdata,
                            h.sampling_rate,
                            False, False,
                            device=device, train=False)

        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                        sampler=None,
                                        batch_size=1,
                                        pin_memory=True,
                                        drop_last=True)
        
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    criterion = nn.BCEWithLogitsLoss()
    
    #################################### Training ####################################
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))
        accum_iter = 2
        for ii, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            cough_audio, covid_status, filename = batch

            cough_audio = torch.autograd.Variable(cough_audio.to(device, non_blocking=True))
            covid_status = torch.autograd.Variable(covid_status.to(device, non_blocking=True))
            with torch.no_grad():
                out = w2v2.feature_extractor(cough_audio)
                
            covid_stat_hat = generator(out)

            if ii == 0:
                optim_g.zero_grad()

            covid_stat_hat = torch.nan_to_num(covid_stat_hat)
            covid_stat_loss = criterion(covid_stat_hat, covid_status.float())

            loss_gen_all = covid_stat_loss
            loss_gen = loss_gen_all / accum_iter
            scaler_g.scale(loss_gen).backward()
            if ((ii+1) % accum_iter ==0) or (ii+1 == len(train_loader)):
                scaler_g.step(optim_g)
                scaler_g.update()
                optim_g.zero_grad()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Covid-stat-loss : {:4.3f}, s/b : {:4.3f}'\
                                                            .format(steps, loss_gen_all, covid_stat_loss, time.time() - start_b))
                                                
                # Checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': generator.state_dict(),
                                    'optim_g': optim_g.state_dict(),
                                    'steps': steps,
                                    'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_tot", loss_gen_all, steps)
                    sw.add_scalar("training/gen_covid_stat_aam_loss", covid_stat_loss, steps)

                #################################### Validation ####################################
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    covid_stat_error = 0
                    real_stat = torch.tensor([])
                    pred_stat = torch.tensor([])
                    with torch.no_grad():
                        pcount = 0
                        pbar = pkbar.Pbar('validation, epoch:{}'.format(epoch+1), len(validation_loader))
                        for jj, batch in enumerate(validation_loader):
                            pbar.update(pcount)
                            cough_audio, covid_status, filename = batch
                            with torch.no_grad():
                                out = w2v2.feature_extractor(cough_audio.to(device))
                            covid_stat_hat = generator(out,False)

                            filename = filename[0].split('.')[0]
                                
                            covid_stat_hat = torch.nan_to_num(covid_stat_hat)
                            loss = criterion(covid_stat_hat, covid_status.float().to(device)).item()
                            covid_stat_error += loss
                            
                            # ROC_AUC ready
                            real_stat = torch.cat((real_stat.to(device), covid_status.to(device)),dim=0)
                            pred_stat = torch.cat((pred_stat.to(device), covid_stat_hat.to(device)),dim=0)
                            
                            pcount += 1
                            
                        val_covid_stat_err = covid_stat_error / (jj+1)
                        AUC = roc_auc_score(real_stat.cpu(), pred_stat.cpu())
                        sw.add_scalar("validation/covid_stat_error", val_covid_stat_err, steps)
                        sw.add_scalar("validation/roc_auc", AUC, steps)

                    generator.train()
            steps += 1
        
        scheduler_g.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint_path', default='/home/woojinchung/codefile/BHI_2023_challenge/cp/cp_temp')
    parser.add_argument('--config', default='config_16batch.json')
    parser.add_argument('--training_epochs', default=10000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=2000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=2000, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    train(0, a, h)


if __name__ == '__main__':
    main()