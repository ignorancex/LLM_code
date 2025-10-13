import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import os
import pandas as pd
import sys
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import subprocess
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import numpy as np
import pickle
from tqdm import tqdm
import os
import glob
from sklearn.metrics import accuracy_score, mean_squared_error
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import FEDformer, Autoformer, Informer, Transformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import miopen
import types

# os.environ['MIOPEN_USER_DB_PATH'] = '/scratch/pawsey0993/katariak579/.miopen'
# os.environ['MIOPEN_CACHE_DIR'] = '/scratch/pawsey0993/katariak579/.cache/miopen'
# os.environ['XDG_CACHE_HOME'] = '/scratch/pawsey0993/katariak579/.cache'

class Attention(nn.Module):
    def __init__(self,dim,max_seq_len=197, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.,dp_rank = 8,mustInPos = None):
        
        super().__init__()
        
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        
        self.seq_len = max_seq_len
        self.dp_rank = dp_rank
        self.fold = 8
        self.fourierSampling = 16
        
        self.weights_fft = nn.Parameter(torch.empty(self.seq_len//2+1,dim,2))
        # self.weights_fft = nn.Parameter(torch.empty(32,dim,2))
        nn.init.kaiming_uniform_(self.weights_fft, mode='fan_in', nonlinearity='relu')
        
        self.tiny_conv_linear =  torch.nn.Conv1d(in_channels = dim*2 , out_channels = dim, kernel_size = 3,padding=  1, groups = 1)
    
        self.dropout1 = torch.nn.Dropout(p=attn_drop)
        self.bn_1 = nn.BatchNorm1d(self.seq_len)

         
        self.index_set_right =   torch.randperm(self.head_dim)
        if self.dp_rank <self.head_dim:
            self.index_set_right = self.index_set_right[:self.dp_rank]

        self.index_set_left =   torch.randperm(self.seq_len)
        if self.dp_rank <self.seq_len:
            self.index_set_left = self.index_set_left[:self.dp_rank]
            
            
            
        self.indx_set_Fourier =   torch.randperm(self.seq_len//2+1)
        if self.fourierSampling <self.seq_len:
            self.indx_set_Fourier = self.indx_set_Fourier[:self.fourierSampling]



        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.ln_post = nn.LayerNorm(dim)
        self.ln_pre = nn.LayerNorm(dim)
        
        
        
    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_heads*self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_heads, self.head_dim)
        X = X.transpose(1, 2)
        return X
        
    def forward(self, x):
        x0 = x
        B, N, C = x.shape
        
        if C//self.fold >0:
            u = x.reshape(B,N,self.fold,C//self.fold)
            u = torch.mean(u,dim = -1)
        #### Fourier Convolution ####
            fft_u = torch.fft.rfft(u, n = self.seq_len, axis = -2,norm = 'ortho')
            fft_u = torch.view_as_real(fft_u)
            fft_u = fft_u.repeat(1,1,C//self.fold,1)
        else:
            fft_u = torch.fft.rfft(x, n = self.seq_len, axis = -2,norm = 'ortho')
            fft_u = torch.view_as_real(fft_u)


        # weights_fft = torch.cat((self.weights_fft,torch.zeros(self.seq_len//2+1 - self.weights_fft.shape[0],self.weights_fft.shape[1],self.weights_fft.shape[2]).to('cuda')))
        weight_used =self.weights_fft.unsqueeze(0)

        '''we may also use low-rank fft matrix'''
        # weight_used = torch.einsum("lkt,kdt->ldt",self.weights_fft1,self.weights_fft2).unsqueeze(0)#self.weights_fft

        temp_real = fft_u[...,0]*weight_used[...,0] - fft_u[...,1]*weight_used[...,1]
        temp_imag = fft_u[...,0]*weight_used[...,1] + fft_u[...,1]*weight_used[...,0]


        out_ft1 = torch.cat([temp_real.unsqueeze(-1),temp_imag.unsqueeze(-1)],dim =  -1)
        # print(out_ft.shape,len(self.indx_set_Fourier),x.shape)
        # out_ft1 = torch.zeros(out_ft.shape).to('cuda')
        # out_ft1[:,self.indx_set_Fourier,:,:] = out_ft[:,self.indx_set_Fourier,:,:]
        # out_ft[:,self.indx_set_Fourier,:,:] = (out_ft[:,self.indx_set_Fourier,:,:]*0).detach()
        # out_ft = out_ft*0
        out_ft1 = torch.view_as_complex(out_ft1)
        
        
        m = torch.fft.irfft(out_ft1, n =  self.seq_len, axis = -2,norm = 'ortho')

        input_h = torch.cat((m, x), dim = -1)



        h =  self.tiny_conv_linear(input_h.permute(0,2,1)).permute(0,2,1)


        x = self.dropout1(F.elu(self.bn_1(h)))

        
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
    
        
        
        #### left part ####
        if self.dp_rank <= self.seq_len:
            k1 = k[:,:,self.index_set_left,:]
            v1 = v[:,:,self.index_set_left,:]
        else:
            k1 = k
            v1 = v



        dots = q @ k1.transpose(-1,-2)
        dots = dots / math.sqrt(self.head_dim)
        attn = nn.functional.softmax(dots,dim=-1)
        attn = self.attn_drop(attn)

        #### right part ####
        q2 = q.transpose(-1,-2)
        if self.dp_rank <= self.head_dim:

            k2 = k[:,:,:,self.index_set_right]
            v2 = v[:,:,:,self.index_set_right]
        else:
            k2 = k
            v2 = v
  
        dots_r = q2 @ k2
        dots_r = dots_r / math.sqrt(self.seq_len)
        attn_r = nn.functional.softmax(dots_r,dim=-1).transpose(-1,-2)
        attn_r = self.attn_drop(attn_r)

        X = self.split_heads(self.ln_1(self.combine_heads(torch.matmul(attn,v1))))/2 + self.split_heads(self.ln_2(self.combine_heads(torch.matmul(v2,attn_r))))/2
      
        x = X.transpose(1, 2).reshape(B, N, C)

        x = self.proj_drop(x)
        return x #+ x0
    
    



class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None
    
    def get_emb(self,sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        # print(tensor.shape)
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = self.get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc
        
    

    
class DataEmbedding(nn.Module):
    def __init__(self, enc_in, d_model, embed_type='fixed', freq='h', dropout=0.1, seq_len=96):
        super(DataEmbedding, self).__init__()
        self.tiny_conv_linear = torch.nn.Conv1d(in_channels=enc_in, 
                                              out_channels=d_model*1, 
                                              kernel_size=1,
                                              padding=0, 
                                              groups=1)
        self.dropout = nn.Dropout(p=dropout)        
        self.posembeding = PositionalEncoding1D(channels=d_model)

    def forward(self, x, x_mark=None):
        # Ensure x is on the correct device
        device = x.device
        x_mean = torch.mean(x, dim=-1, keepdim=True).to(device)
        x1 = self.dropout(self.tiny_conv_linear(x.permute(0,2,1)).permute(0,2,1))
        pos_encoding = self.posembeding(x1).to(device)
        return pos_encoding + x1
                        
                        
    
    
class SKTLinear(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    """
    def __init__(self, configs):
        super(SKTLinear, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.embd_dim = configs.enc_in
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len

        
        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,configs.dropout,configs.seq_len)
                
        
        self.encoder =  nn.ModuleList(
            [    
                Attention(dim = configs.d_model, max_seq_len=configs.seq_len, num_heads=configs.n_heads, qkv_bias=False, attn_drop=configs.dropout, proj_drop=configs.dropout,dp_rank = 8,mustInPos = None)  
                 for l in range(configs.e_layers)
            ],
        )

        self.Postmlp=  nn.ModuleList(
            [    
                nn.Linear(configs.d_model,self.embd_dim)
                 for l in range(configs.e_layers)
            ],
        )
        
        self.fourierExtrapolation = fourierExtrapolation(inputSize = configs.seq_len,n_harm = self.pred_len,n_predict = self.pred_len)
        

        
        

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        B1,H1,C1 = x_enc.shape
        # dec_out = []
        for i in range(len(self.encoder)):
            attn_layer,post = self.encoder[i],self.Postmlp[i]
            
            if i == 0:
                tmp_mean = torch.mean(x_enc[:,:,:],dim = 1,keepdim = True)#.detach()
                tmp_std = torch.sqrt(torch.var(x_enc[:,:,:],dim = 1,keepdim = True)+1e0)#.detach()
                x_enc = (x_enc - tmp_mean)/(tmp_std) 

                enc_out1 = self.enc_embedding(x_enc)
         
            enc_out1= attn_layer(enc_out1) + enc_out1 

             

     
        
        tmp1 = self.fourierExtrapolation.fourierExtrapolation(post(enc_out1[:,:self.seq_len,:]))
       
        # dec_out.append(tmp1[:,:,:].unsqueeze(-1))
        dec_out = tmp1[:,:,:].unsqueeze(-1)
 

        # dec_out = torch.mean(torch.cat(dec_out,axis = -1),axis = -1)
        # dec_out = dec_out[-1].squeeze(-1)
     
        output = (dec_out.reshape(B1,-1,C1))*(tmp_std)+tmp_mean 
        
        
        return output[:,-self.pred_len:,:], output[:,:self.seq_len,:]
    
    
    
class Exp_Main(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                print(f'Using GPU: {torch.cuda.get_device_name(0)}')
                return device
            else:
                print('CUDA is not available. Falling back to CPU.')
        print('Using CPU')
        return torch.device('cpu')

    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
        }
        model = SKTLinear(self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        try:    
            data_loaders = get_data_loaders(self.args)
            return None, data_loaders[flag]
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def _select_optimizer(self):
        model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        regression_criterion = nn.MSELoss()
        classification_criterion = nn.CrossEntropyLoss()
        return regression_criterion, classification_criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # outputs = outputs#/x_mean #+x_mean
                # batch_y = batch_y#/x_mean
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Create checkpoint directory
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion_reg, criterion_cls = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, group_emotions, group_emotion_categories, 
                class_emotions, class_valences, class_arousals, 
                event_emotions, events) in enumerate(train_loader):
                
                iter_count += 1
                model_optim.zero_grad()
                
                # Move data to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        loss = self.train_step(batch_x, batch_y, group_emotions, 
                                            group_emotion_categories, class_emotions,
                                            class_valences, class_arousals, 
                                            event_emotions, events,
                                            criterion_reg, criterion_cls)
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                else:
                    loss = self.train_step(batch_x, batch_y, group_emotions,
                                        group_emotion_categories, class_emotions,
                                        class_valences, class_arousals,
                                        event_emotions, events,
                                        criterion_reg, criterion_cls)
                    loss.backward()
                    model_optim.step()

                train_loss.append(loss.item())

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion_reg, criterion_cls)
            test_loss = self.vali(test_data, test_loader, criterion_reg, criterion_cls)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            
            # Save checkpoint for each epoch
            checkpoint_path = os.path.join(path, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': model_optim.state_dict(),
                'train_loss': train_loss,
                'val_loss': vali_loss,
                'test_loss': test_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs ,x_mean= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs ,x_mean= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs ,x_mean= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs ,x_mean= self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs= outputs#+x_mean
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs ,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs ,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs ,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs ,x_mean = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs#+x_mean
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
    
class fourierExtrapolation(nn.Module):
    def __init__(self, inputSize, n_harm=8, n_predict=96, device=None):
        super().__init__()
        
        # Set device
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize basic parameters
        self.n = inputSize
        self.n_harm = n_harm
        self.n_predict = n_predict
        
        # Create time array directly on the correct device
        t = torch.arange(0, self.n, device=self.device)
        self.register_buffer('t', t)
        self.register_buffer('t0', t.float())
        self.register_buffer('x_mean', torch.mean(self.t0))
        self.register_buffer('x_square', torch.mean((self.t0 - self.x_mean)*(self.t0 - self.x_mean)))
        
        # Create FFT frequencies on the correct device
        self.register_buffer('f', torch.fft.fftfreq(self.n, device=self.device))
        
        # Create indices
        f_abs = torch.abs(self.f)
        _, sorted_indices = torch.sort(f_abs)
        self.indexes = sorted_indices[:(1 + self.n_harm * 2)].tolist()
        
        # Create time vectors on the correct device
        t_extended = torch.arange(0, self.n + self.n_predict, device=self.device)
        self.register_buffer('t1', t_extended.unsqueeze(0).unsqueeze(-1).float())
        
        # Reshape tensors
        self.register_buffer('f_shaped', self.f.unsqueeze(0).unsqueeze(-1))
        self.register_buffer('t_shaped', t_extended.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        
        # Create g tensor
        f_indexed = self.f_shaped[:, self.indexes, :].permute(0, 2, 1)
        self.register_buffer('g', f_indexed.unsqueeze(1))
        
        # Create phase initialization
        self.register_buffer('phase_init', 2 * 3.1415 * self.g * self.t_shaped)
        
        # Regular layer
        self.decoder = nn.Linear(96*2, 96)

    def linearfit(self, y, Lambda=1e2):
        B, H, C = y.shape
        y_mean = torch.mean(y, dim=1, keepdim=True)
        b = torch.mean((self.t0 - self.x_mean)*(y-y_mean), dim=1, keepdim=True) / (self.x_square + Lambda)
        return b.detach()

    def fourierExtrapolation(self, x, notrend=True):
        x_freqdom = torch.fft.fft(x, dim=-2)
        x_freqdom = torch.view_as_real(x_freqdom)
        x_freqdom = x_freqdom[:, self.indexes, :, :]
        x_freqdom = torch.view_as_complex(x_freqdom)
        ampli = torch.absolute(x_freqdom) / self.n
        phase = torch.angle(x_freqdom)

        ampli = ampli.permute(0, 2, 1).unsqueeze(1)
        phase = phase.permute(0, 2, 1).unsqueeze(1)

        self.restored_sig = ampli * torch.cos(self.phase_init + phase)
        
        return torch.sum(self.restored_sig, dim=-1)
    
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of 300')
            if self.counter >= 300:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        
        # Ensure directory exists
        os.makedirs(path, exist_ok=True)
        
        # Create model save path
        save_path = os.path.join(path, f'checkpoint_{val_loss}.pth')
        
        # Save the model state dict
        torch.save(model.state_dict(), save_path)
        print(f'Model saved to {save_path}')
        
        self.val_loss_min = val_loss

class CustomEmotionDataset(Dataset):
    def __init__(self, face_features_path, text_latent_path1, text_latent_path2, text_latent_path3, labels_path, seq_len, pred_len,root_path="../Dataset"):

        self.root_path = root_path
        self.face_features_path = os.path.join(root_path, face_features_path)
        self.text_latent1_path = os.path.join(root_path, text_latent_path1)
        self.text_latent2_path = os.path.join(root_path, text_latent_path2)
        self.text_latent3_path = os.path.join(root_path, text_latent_path3)
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Load labels
        self.labels_df = pd.read_csv(os.path.join(root_path, labels_path))
        
        # Get valid subfolders (present in all input folders and labels)
        self.valid_subfolders = self._get_valid_subfolders()

    def _get_valid_subfolders(self):
        face_subfolders = set(os.listdir(self.face_features_path))
        text1_files = set(f.split('.')[0] for f in os.listdir(self.text_latent1_path))
        text2_files = set(f.split('.')[0] for f in os.listdir(self.text_latent2_path))
        text3_files = set(f.split('.')[0] for f in os.listdir(self.text_latent3_path))
        label_files = set(self.labels_df['video_file_name'])
        
        return list(face_subfolders & text1_files & text2_files & text3_files & label_files)

    def __len__(self):
        return len(self.valid_subfolders)

    def __getitem__(self, idx):
        subfolder = self.valid_subfolders[idx]
        
        # Load face features
        face_features = []
        face_subfolder_path = os.path.join(self.face_features_path, subfolder)
        face_files = sorted(os.listdir(face_subfolder_path))
        for file in face_files[:self.seq_len + self.pred_len]:
            feature = np.load(os.path.join(face_subfolder_path, file))
            face_features.append(feature)
        
        # Pad or truncate face features if necessary
        if len(face_features) < self.seq_len + self.pred_len:
            pad_length = self.seq_len + self.pred_len - len(face_features)
            face_features.extend([np.zeros((1024,))] * pad_length)
        elif len(face_features) > self.seq_len + self.pred_len:
            face_features = face_features[:self.seq_len + self.pred_len]
        
        face_features = np.array(face_features)
        
        # Load text latent features
        text_latent1 = np.load(os.path.join(self.text_latent1_path, f"{subfolder}.npy"))
        text_latent2 = np.load(os.path.join(self.text_latent2_path, f"{subfolder}.npy"))
        text_latent3 = np.load(os.path.join(self.text_latent3_path, f"{subfolder}.npy"))
        
        # Reshape face features and text latent features
        face_features = face_features.reshape(self.seq_len + self.pred_len, -1)  # reshape to (seq_len + pred_len, 1024)
        text_latent1 = np.tile(text_latent1, (self.seq_len + self.pred_len, 1))  # repeat to match face_features shape
        text_latent2 = np.tile(text_latent2, (self.seq_len + self.pred_len, 1))
        text_latent3 = np.tile(text_latent3, (self.seq_len + self.pred_len, 1))
        
        # Combine all features
        combined_features = np.concatenate([face_features, text_latent1, text_latent2, text_latent3], axis=1)
        
        # Split into input and target sequences
        x = combined_features[:self.seq_len]
        y = combined_features[self.seq_len:self.seq_len + self.pred_len]
        
        # Get labels
        labels = self.labels_df[self.labels_df['video_file_name'] == subfolder].iloc[0]
        
        return (
        torch.FloatTensor(x),
        torch.FloatTensor(y),
        torch.LongTensor([labels['group_emotion']]),
        torch.LongTensor([labels['group_emotion_category']]),
        torch.LongTensor([labels['class_emotion']]),
        torch.FloatTensor([labels['class_valence']]),
        torch.FloatTensor([labels['class_arousal']]),
        torch.LongTensor([labels['event_emotion']]),
        torch.LongTensor([labels['event']])
    )


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def plot_label_distributions(labels_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Read the labels CSV file
    df = pd.read_csv(labels_path)
    
    # List of columns to plot
    columns_to_plot = ['group_emotion', 'group_emotion_category', 'class_emotion', 'event_emotion', 'event']
    
    for column in columns_to_plot:
        plt.figure(figsize=(10, 6))
        df[column].value_counts().sort_index().plot(kind='bar')
        plt.title(f'{column.replace("_", " ").title()} Distribution')
        plt.xlabel(column.replace("_", " ").title())
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{column}_distribution.jpg'))
        plt.close()
    
    # Plot valence and arousal distributions
    for column in ['class_valence', 'class_arousal']:
        plt.figure(figsize=(10, 6))
        plt.hist(df[column], bins=50, range=(0, 999))
        plt.title(f'{column.replace("_", " ").title()} Distribution\nRange: [{df[column].min():.2f}, {df[column].max():.2f}]')
        plt.xlabel(column.replace("_", " ").title())
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{column}_distribution.jpg'))
        plt.close()
    
    print(f"Label distribution plots saved in {save_dir}")


def custom_collate_fn(batch):
    x = torch.stack([item[0] for item in batch])
    y = torch.stack([item[1] for item in batch])
    group_emotions = torch.cat([item[2] for item in batch])
    group_emotion_categories = torch.cat([item[3] for item in batch])
    class_emotions = torch.cat([item[4] for item in batch])
    class_valences = torch.cat([item[5] for item in batch])
    class_arousals = torch.cat([item[6] for item in batch])
    event_emotions = torch.cat([item[7] for item in batch])
    event = torch.cat([item[8] for item in batch])
    return x, y, group_emotions, group_emotion_categories, class_emotions, class_valences, class_arousals, event_emotions, event


def get_data_loaders(args):
    train_dataset = CustomEmotionDataset(
        face_features_path=os.path.join(args.root_path, 'train', 'face_features'),
        text_latent_path1=os.path.join(args.root_path, 'train', 'text_latent1'),
        text_latent_path2=os.path.join(args.root_path, 'train', 'text_latent2'),
        text_latent_path3=os.path.join(args.root_path, 'train', 'text_latent3'),
        labels_path=os.path.join(args.root_path, 'train', 'labels.csv'),
        seq_len=args.seq_len,
        pred_len=args.pred_len
    )
    
    val_dataset = CustomEmotionDataset(
        face_features_path=os.path.join(args.root_path, 'val', 'face_features'),
        text_latent_path1=os.path.join(args.root_path, 'val', 'text_latent1'),
        text_latent_path2=os.path.join(args.root_path, 'val', 'text_latent2'),
        text_latent_path3=os.path.join(args.root_path, 'val', 'text_latent3'),
        labels_path=os.path.join(args.root_path, 'val', 'labels.csv'),
        seq_len=args.seq_len,
        pred_len=args.pred_len
    )
    
    test_dataset = CustomEmotionDataset(
        face_features_path=os.path.join(args.root_path, 'test', 'face_features'),
        text_latent_path1=os.path.join(args.root_path, 'test', 'text_latent1'),
        text_latent_path2=os.path.join(args.root_path, 'test', 'text_latent2'),
        text_latent_path3=os.path.join(args.root_path, 'test', 'text_latent3'),
        labels_path=os.path.join(args.root_path, 'test', 'labels.csv'),
        seq_len=args.seq_len,
        pred_len=args.pred_len
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def create_and_save_plots(predictions, metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Group Emotion
    group_emotion_pred = np.argmax(np.concatenate([p['group_emotion'] for p in predictions]), axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(group_emotion_pred, bins=9, range=(0, 9), align='left', rwidth=0.8)
    plt.title('Group Emotion Distribution')
    plt.xlabel('Group Emotion')
    plt.ylabel('Count')
    plt.xticks(range(9))
    plt.savefig(os.path.join(save_dir, 'group_emotion.jpg'))
    plt.close()

    # Group Emotion Category
    group_emotion_category_pred = np.argmax(np.concatenate([p['group_emotion_category'] for p in predictions]), axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(group_emotion_category_pred, bins=3, range=(0, 3), align='left', rwidth=0.8)
    plt.title('Group Emotion Category Distribution')
    plt.xlabel('Group Emotion Category')
    plt.ylabel('Count')
    plt.xticks(range(3))
    plt.savefig(os.path.join(save_dir, 'group_emotion_category.jpg'))
    plt.close()

    # Class Emotion
    class_emotion_pred = np.argmax(np.concatenate([p['class_emotion'] for p in predictions]), axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(class_emotion_pred, bins=9, range=(0, 9), align='left', rwidth=0.8)
    plt.title('Class Emotion Distribution')
    plt.xlabel('Class Emotion')
    plt.ylabel('Count')
    plt.xticks(range(9))
    plt.savefig(os.path.join(save_dir, 'class_emotion.jpg'))
    plt.close()

    # Class Valence
    class_valence_pred = np.concatenate([p['class_valence'] for p in predictions]).squeeze()
    plt.figure(figsize=(10, 6))
    plt.hist(class_valence_pred, bins=50, range=(0, 999))
    plt.title(f'Class Valence Distribution (0-999)\nRange: [{class_valence_pred.min():.2f}, {class_valence_pred.max():.2f}]')
    plt.xlabel('Class Valence')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, 'class_valence.jpg'))
    plt.close()

    # Class Arousal
    class_arousal_pred = np.concatenate([p['class_arousal'] for p in predictions]).squeeze()
    plt.figure(figsize=(10, 6))
    plt.hist(class_arousal_pred, bins=50, range=(0, 999))
    plt.title(f'Class Arousal Distribution (0-999)\nRange: [{class_arousal_pred.min():.2f}, {class_arousal_pred.max():.2f}]')
    plt.xlabel('Class Arousal')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, 'class_arousal.jpg'))
    plt.close()

    # Event Emotion
    event_emotion_pred = np.argmax(np.concatenate([p['event_emotion'] for p in predictions]), axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(event_emotion_pred, bins=10, range=(0, 10), align='left', rwidth=0.8)
    plt.title('Event Emotion Distribution')
    plt.xlabel('Event Emotion')
    plt.ylabel('Count')
    plt.xticks(range(10))
    plt.savefig(os.path.join(save_dir, 'event_emotion.jpg'))
    plt.close()

    # Event
    event_pred = np.argmax(np.concatenate([p['event'] for p in predictions]), axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(event_pred, bins=9, range=(0, 9), align='left', rwidth=0.8)
    plt.title('Event Distribution')
    plt.xlabel('Event')
    plt.ylabel('Count')
    plt.xticks(range(9))
    plt.savefig(os.path.join(save_dir, 'event.jpg'))
    plt.close()

    print(f"Plots saved in {save_dir}")



def custom_collate_fn(batch):
    x = torch.stack([item[0] for item in batch])
    y = torch.stack([item[1] for item in batch])
    group_emotions = torch.cat([item[2] for item in batch])
    group_emotion_categories = torch.cat([item[3] for item in batch])
    class_emotions = torch.cat([item[4] for item in batch])
    class_valences = torch.cat([item[5] for item in batch])
    class_arousals = torch.cat([item[6] for item in batch])
    event_emotions = torch.cat([item[7] for item in batch])
    event = torch.cat([item[8] for item in batch])
    return x, y, group_emotions, group_emotion_categories, class_emotions, class_valences, class_arousals, event_emotions, event




class CustomS3AttentionModel(nn.Module):
    def __init__(self, configs):
        super(CustomS3AttentionModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        
        # Create all components
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, 
                                         configs.freq, configs.dropout).to(self.device)
        
        self.encoder = nn.ModuleList([
            Attention(dim=configs.d_model, 
                     max_seq_len=configs.seq_len, 
                     num_heads=configs.n_heads,
                     qkv_bias=False,
                     attn_drop=configs.dropout,
                     proj_drop=configs.dropout,
                     dp_rank=8).to(self.device)
            for _ in range(configs.e_layers)
        ])
        
        # Output layers
        self.projection = nn.Linear(configs.d_model, configs.enc_in).to(self.device)
        self.group_emotion_projection = nn.Linear(configs.d_model, 9).to(self.device)
        self.group_emotion_category_projection = nn.Linear(configs.d_model, 3).to(self.device)
        self.class_emotion_projection = nn.Linear(configs.d_model, 9).to(self.device)
        self.class_valence_projection = nn.Linear(configs.d_model, 1).to(self.device)
        self.class_arousal_projection = nn.Linear(configs.d_model, 1).to(self.device)
        self.event_emotion_projection = nn.Linear(configs.d_model, 10).to(self.device)
        # Change this line in CustomS3AttentionModel class
        self.event_projection = nn.Linear(configs.d_model, 10).to(self.device)  # Changed from 9 to 10
        # self.event_projection = nn.Linear(configs.d_model, 9).to(self.device)
        
        # Create Fourier extrapolation with device
        self.fourier_extrapolation = fourierExtrapolation(
            inputSize=configs.seq_len,
            n_harm=self.pred_len,
            n_predict=self.pred_len,
            device=self.device
        ).to(self.device)

    def forward(self, x_enc, x_mark_enc=None):
        x_enc = x_enc.to(self.device)
        if x_mark_enc is not None:
            x_mark_enc = x_mark_enc.to(self.device)
            
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # Encoder
        for attn_layer in self.encoder:
            enc_out = attn_layer(enc_out) + enc_out
            
        # Ensure all operations happen on the same device
        enc_out = enc_out.to(self.device)
        
        # Predictions
        time_series_pred = self.fourier_extrapolation.fourierExtrapolation(self.projection(enc_out))
        
        last_hidden = enc_out[:, -1, :]
        group_emotion = self.group_emotion_projection(last_hidden)
        group_emotion_category = self.group_emotion_category_projection(last_hidden)
        class_emotion = self.class_emotion_projection(last_hidden)
        class_valence = self.class_valence_projection(last_hidden)
        class_arousal = self.class_arousal_projection(last_hidden)
        event_emotion = self.event_emotion_projection(last_hidden)
        event = self.event_projection(last_hidden)
        
        return time_series_pred, group_emotion, group_emotion_category, class_emotion, class_valence, class_arousal, event_emotion, event
    
def normalize_to_range(values, new_min=0.35, new_max=0.675):
    min_val = np.min(values)
    max_val = np.max(values)
    normalized = (values - min_val) / (max_val - min_val)
    return normalized * (new_max - new_min) + new_min
    

import os
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error
from utils.tools import EarlyStopping, adjust_learning_rate

class CustomExpMain:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = CustomS3AttentionModel(self.args)
        return model

    def _get_data(self, flag):
        try:
            data_loaders = get_data_loaders(self.args)
            return None, data_loaders[flag]
        except Exception as e:
            print(f"Error loading data: {e}")
            raise


    def _select_optimizer(self):
        model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion_reg = nn.MSELoss()
        criterion_cls = nn.CrossEntropyLoss()
        return criterion_reg, criterion_cls

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Create checkpoint directory
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion_reg, criterion_cls = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, group_emotions, group_emotion_categories, 
                class_emotions, class_valences, class_arousals, 
                event_emotions, events) in enumerate(train_loader):
                
                iter_count += 1
                model_optim.zero_grad()
                
                # Move data to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        loss = self.train_step(batch_x, batch_y, group_emotions, 
                                            group_emotion_categories, class_emotions,
                                            class_valences, class_arousals, 
                                            event_emotions, events,
                                            criterion_reg, criterion_cls)
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                else:
                    loss = self.train_step(batch_x, batch_y, group_emotions,
                                        group_emotion_categories, class_emotions,
                                        class_valences, class_arousals,
                                        event_emotions, events,
                                        criterion_reg, criterion_cls)
                    loss.backward()
                    model_optim.step()

                train_loss.append(loss.item())

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion_reg, criterion_cls)
            test_loss = self.vali(test_data, test_loader, criterion_reg, criterion_cls)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            
            # Save checkpoint for each epoch
            checkpoint_path = os.path.join(path, f'checkpoint_{epoch+1}_tl_{train_loss}_vl_{vali_loss}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': model_optim.state_dict(),
                'train_loss': train_loss,
                'val_loss': vali_loss,
                'test_loss': test_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        return self.model

    def save_epoch_metrics(self, path, epoch, train_loss, val_loss):
        """Save epoch metrics to a separate file"""
        metrics_file = os.path.join(path, 'training_metrics.txt')
        with open(metrics_file, 'a') as f:
            f.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}\n")

    def vali(self, vali_data, vali_loader, criterion_reg, criterion_cls):
        self.model.eval()
        total_loss = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y, group_emotions, group_emotion_categories, 
                    class_emotions, class_valences, class_arousals, 
                    event_emotions, events) in enumerate(vali_loader):
                    
                # Move data to device and ensure correct types
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                group_emotions = group_emotions.long().to(self.device)
                group_emotion_categories = group_emotion_categories.long().to(self.device)
                class_emotions = class_emotions.long().to(self.device)
                class_valences = class_valences.float().to(self.device)
                class_arousals = class_arousals.float().to(self.device)
                event_emotions = event_emotions.long().to(self.device)
                events = events.long().to(self.device)

                # Get model outputs
                outputs = self.model(batch_x)
                time_series_pred, group_emotion_pred, group_emotion_category_pred, \
                class_emotion_pred, class_valence_pred, class_arousal_pred, \
                event_emotion_pred, event_pred = outputs

                # Calculate losses with proper type handling
                loss_ts = criterion_reg(time_series_pred, batch_y)
                loss_group = criterion_cls(group_emotion_pred, group_emotions.squeeze())
                loss_group_category = criterion_cls(group_emotion_category_pred, group_emotion_categories.squeeze())
                loss_class = criterion_cls(class_emotion_pred, class_emotions.squeeze())
                loss_valence = criterion_reg(class_valence_pred.squeeze(), class_valences.squeeze())
                loss_arousal = criterion_reg(class_arousal_pred.squeeze(), class_arousals.squeeze())
                loss_event_emotion = criterion_cls(event_emotion_pred, event_emotions.squeeze())
                loss_event = criterion_cls(event_pred, events.squeeze())

                # Combine all losses
                loss = (loss_ts + loss_group + loss_group_category + loss_class + 
                    loss_valence + loss_arousal + loss_event_emotion + loss_event)
                
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train_step(self, batch_x, batch_y, group_emotions, group_emotion_categories, 
                class_emotions, class_valences, class_arousals, 
                event_emotions, events, criterion_reg, criterion_cls):
        # Move data to device and ensure correct types
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        group_emotions = group_emotions.long().to(self.device)
        group_emotion_categories = group_emotion_categories.long().to(self.device)
        class_emotions = class_emotions.long().to(self.device)
        class_valences = class_valences.float().to(self.device)
        class_arousals = class_arousals.float().to(self.device)
        event_emotions = event_emotions.long().to(self.device)
        events = events.long().to(self.device)

        # Get model outputs
        outputs = self.model(batch_x)
        time_series_pred, group_emotion_pred, group_emotion_category_pred, \
        class_emotion_pred, class_valence_pred, class_arousal_pred, \
        event_emotion_pred, event_pred = outputs

        # Calculate losses with proper type handling
        loss_ts = criterion_reg(time_series_pred, batch_y)
        loss_group = criterion_cls(group_emotion_pred, group_emotions.squeeze())
        loss_group_category = criterion_cls(group_emotion_category_pred, group_emotion_categories.squeeze())
        loss_class = criterion_cls(class_emotion_pred, class_emotions.squeeze())
        loss_valence = criterion_reg(class_valence_pred.squeeze(), class_valences.squeeze())
        loss_arousal = criterion_reg(class_arousal_pred.squeeze(), class_arousals.squeeze())
        loss_event_emotion = criterion_cls(event_emotion_pred, event_emotions.squeeze())
        loss_event = criterion_cls(event_pred, events.squeeze())

        # Combine all losses
        loss = (loss_ts + loss_group + loss_group_category + loss_class + 
                loss_valence + loss_arousal + loss_event_emotion + loss_event)
        
        return loss

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

        self.model.eval()
        
        preds = []
        trues = []
        group_emotion_preds = []
        group_emotion_category_preds = []
        class_emotion_preds = []
        class_valence_preds = []
        class_arousal_preds = []
        event_emotion_preds = []
        event_preds= []
        
        with torch.no_grad():
            for i, (batch_x, batch_y, group_emotions, group_emotion_categories, class_emotions, class_valences, class_arousals, event_emotions, event) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                time_series_pred, group_emotion, group_emotion_category, class_emotion, class_valence, class_arousal, event_emotion, event = outputs

                preds.append(time_series_pred.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())
                group_emotion_preds.append(group_emotion.detach().cpu().numpy())
                group_emotion_category_preds.append(group_emotion_category.detach().cpu().numpy())
                class_emotion_preds.append(class_emotion.detach().cpu().numpy())
                class_valence_preds.append(class_valence.detach().cpu().numpy())
                class_arousal_preds.append(class_arousal.detach().cpu().numpy())
                event_emotion_preds.append(event_emotion.detach().cpu().numpy())
                event_preds.append(event.detach().cpu().numpy())
                # event.append(event.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        group_emotion_preds = np.concatenate(group_emotion_preds, axis=0)
        group_emotion_category_preds = np.concatenate(group_emotion_category_preds, axis=0)
        class_emotion_preds = np.concatenate(class_emotion_preds, axis=0)
        class_valence_preds = np.concatenate(class_valence_preds, axis=0)
        class_arousal_preds = np.concatenate(class_arousal_preds, axis=0)
        event_emotion_preds = np.concatenate(event_emotion_preds, axis=0)
        event_preds = np.concatenate(event_preds, axis=0)

        # Calculate metrics
        mse = mean_squared_error(trues, preds)
        group_emotion_acc = accuracy_score(np.argmax(group_emotion_preds, axis=1), group_emotions.numpy())
        group_emotion_category_acc = accuracy_score(np.argmax(group_emotion_category_preds, axis=1), group_emotion_categories.numpy())
        class_emotion_acc = accuracy_score(np.argmax(class_emotion_preds, axis=1), class_emotions.numpy())
        class_valence_mse = mean_squared_error(class_valence_preds, class_valences.numpy())
        class_arousal_mse = mean_squared_error(class_arousal_preds, class_arousals.numpy())
        event_emotion_acc = accuracy_score(np.argmax(event_emotion_preds, axis=1), event_emotions.numpy())
        event_acc = accuracy_score(np.argmax(event_preds, axis=1), event.numpy())

        # Print metrics
        print('MSE: {:.4f}'.format(mse))
        print('Group Emotion Accuracy: {:.4f}'.format(group_emotion_acc))
        print('Group Emotion Category Accuracy: {:.4f}'.format(group_emotion_category_acc))
        print('Class Emotion Accuracy: {:.4f}'.format(class_emotion_acc))
        print('Class Valence MSE: {:.4f}'.format(class_valence_mse))
        print('Class Arousal MSE: {:.4f}'.format(class_arousal_mse))
        print('Event Emotion Accuracy: {:.4f}'.format(event_emotion_acc))
        print('Event Accuracy: {:.4f}'.format(event_acc))


        # Save results
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'metrics.npy', np.array([mse, group_emotion_acc, group_emotion_category_acc, class_emotion_acc, class_valence_mse, class_arousal_mse, event_emotion_acc, event_acc]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
    
    def predict_and_evaluate_saved_model(self, model_path, data_loader):
        # Load the full saved model
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        
        predictions = []
        true_values = {
            'group_emotion': [],
            'group_emotion_category': [],
            'class_emotion': [],
            'class_valence': [],
            'class_arousal': [],
            'event_emotion': [],
            'event': []
        }
        
        with torch.no_grad():
            for i, (batch_x, batch_y, group_emotions, group_emotion_categories, class_emotions, class_valences, class_arousals, event_emotions, event) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                
                # Normalize valence and arousal to 0-999 range
                normalized_valence = normalize_to_range(outputs[4].cpu().numpy())
                normalized_arousal = normalize_to_range(outputs[5].cpu().numpy())
                
                # Append predictions
                predictions.append({
                    'time_series': outputs[0].cpu().numpy(),
                    'group_emotion': outputs[1].cpu().numpy(),
                    'group_emotion_category': outputs[2].cpu().numpy(),
                    'class_emotion': outputs[3].cpu().numpy(),
                    'class_valence': normalized_valence,
                    'class_arousal': normalized_arousal,
                    'event_emotion': outputs[6].cpu().numpy(),
                    'event': outputs[7].cpu().numpy()
                })
                
                # Append true values
                true_values['group_emotion'].extend(group_emotions.numpy())
                true_values['group_emotion_category'].extend(group_emotion_categories.numpy())
                true_values['class_emotion'].extend(class_emotions.numpy())
                true_values['class_valence'].extend(normalize_to_range(class_valences.numpy()))
                true_values['class_arousal'].extend(normalize_to_range(class_arousals.numpy()))
                true_values['event_emotion'].extend(event_emotions.numpy())
                true_values['event'].extend(event.numpy())
        
        # Calculate metrics
        metrics = {}
        
        for key in ['group_emotion', 'group_emotion_category', 'class_emotion', 'event_emotion', 'event']:
            pred = np.argmax(np.concatenate([p[key] for p in predictions]), axis=1)
            true = np.array(true_values[key])
            metrics[f'{key}_accuracy'] = accuracy_score(true, pred)
        
        for key in ['class_valence', 'class_arousal']:
            pred = np.concatenate([p[key] for p in predictions]).squeeze()
            true = np.array(true_values[key])
            metrics[f'{key}_mse'] = mean_squared_error(true, pred)
            metrics[f'{key}_mae'] = mean_absolute_error(true, pred)
        
        # Generate and save plots
        plot_dir = os.path.join(os.path.dirname(model_path), 'plots')
        create_and_save_plots(predictions, metrics, plot_dir)
        
        return predictions, metrics


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        group_emotion_preds = []
        group_emotion_category_preds = []
        class_emotion_preds = []
        class_valence_preds = []
        class_arousal_preds = []
        event_emotion_preds = []
        event_preds = []
        
        with torch.no_grad():
            for i, (batch_x, _, _, _, _, _, _, _) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                time_series_pred, group_emotion, group_emotion_category, class_emotion, class_valence, class_arousal, event_emotion, event = outputs

                preds.append(time_series_pred.detach().cpu().numpy())
                group_emotion_preds.append(group_emotion.detach().cpu().numpy())
                group_emotion_category_preds.append(group_emotion_category.detach().cpu().numpy())
                class_emotion_preds.append(class_emotion.detach().cpu().numpy())
                class_valence_preds.append(class_valence.detach().cpu().numpy())
                class_arousal_preds.append(class_arousal.detach().cpu().numpy())
                event_emotion_preds.append(event_emotion.detach().cpu().numpy())
                event_preds.append(event.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        group_emotion_preds = np.concatenate(group_emotion_preds, axis=0)
        group_emotion_category_preds = np.concatenate(group_emotion_category_preds, axis=0)
        class_emotion_preds = np.concatenate(class_emotion_preds, axis=0)
        class_valence_preds = np.concatenate(class_valence_preds, axis=0)
        class_arousal_preds = np.concatenate(class_arousal_preds, axis=0)
        event_emotion_preds = np.concatenate(event_emotion_preds, axis=0)
        event_preds = np.concatenate(event_preds, axis=0)

        # Save predictions
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'time_series_prediction.npy', preds)
        np.save(folder_path + 'group_emotion_prediction.npy', group_emotion_preds)
        np.save(folder_path + 'group_emotion_category_prediction.npy', group_emotion_category_preds)
        np.save(folder_path + 'class_emotion_prediction.npy', class_emotion_preds)
        np.save(folder_path + 'class_valence_prediction.npy', class_valence_preds)
        np.save(folder_path + 'class_arousal_prediction.npy', class_arousal_preds)
        np.save(folder_path + 'event_emotion_prediction.npy', event_emotion_preds)
        np.save(folder_path + 'event_prediction.npy', event_preds)

        return

    def _process_one_batch(self, batch_x):
        batch_x = batch_x.float().to(self.device)
        outputs = self.model(batch_x)
        return outputs

    @staticmethod
    def save_model(model, path):
        torch.save(model.state_dict(), path)

    @staticmethod
    def load_model(model, path):
        model.load_state_dict(torch.load(path))

    def evaluate_one_batch(self, batch_x, batch_y, group_emotions, group_emotion_categories, class_emotions, class_valences, class_arousals, event_emotions, event):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        group_emotions = group_emotions.long().to(self.device)
        group_emotion_categories = group_emotion_categories.long().to(self.device)
        class_emotions = class_emotions.long().to(self.device)
        class_valences = class_valences.float().to(self.device)
        class_arousals = class_arousals.float().to(self.device)
        event_emotions = event_emotions.long().to(self.device)
        event = event.long().to(self.device)

        outputs = self.model(batch_x)
        time_series_pred, group_emotion, group_emotion_category, class_emotion, class_valence, class_arousal, event_emotion, event = outputs

        criterion_reg, criterion_cls = self._select_criterion()

        loss_ts = criterion_reg(time_series_pred, batch_y)
        loss_group = criterion_cls(group_emotion, group_emotions.squeeze())
        loss_group_category = criterion_cls(group_emotion_category, group_emotion_categories.squeeze())
        loss_class = criterion_cls(class_emotion, class_emotions.squeeze())
        loss_valence = criterion_reg(class_valence.squeeze(), class_valences.squeeze())
        loss_arousal = criterion_reg(class_arousal.squeeze(), class_arousals.squeeze())
        loss_event = criterion_cls(event_emotion, event_emotions.squeeze())
        loss_event_2  = criterion_cls(event, event.squeeze())
        total_loss = loss_ts + loss_group + loss_group_category + loss_event

        # total_loss = loss_ts + loss_group + loss_group_category + loss_class + loss_valence + loss_arousal + loss_event

        return total_loss, outputs

    def get_model_size(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        model_size = sum([np.prod(p.size()) for p in model_parameters])
        return model_size

    def train_and_evaluate(self, setting):
        model = self.train(setting)
        self.test(setting)
        return model

    # def save_checkpoint(self, state, filename='checkpoint.pth'):
    #     torch.save(state, filename)
    #     print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer = self._select_optimizer()
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f"Loaded checkpoint from {filename}. Epoch: {epoch}, Loss: {loss}")
            return optimizer, epoch, loss
        else:
            print(f"No checkpoint found at {filename}")
            return None

    def get_learning_rate(self):
        for param_group in self._select_optimizer().param_groups:
            return param_group['lr']

    def reset_model(self):
        self.model = self._build_model().to(self.device)

    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def export_onnx(self, file_path, input_shape):
        dummy_input = torch.randn(input_shape, device=self.device)
        torch.onnx.export(self.model, dummy_input, file_path, verbose=True)
        print(f"Model exported to ONNX format at {file_path}")

    def log_metrics(self, metrics, step):
        # Implement logging logic here (e.g., using tensorboard or wandb)
        pass

    def set_model_mode(self, mode):
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()
        else:
            raise ValueError("Mode must be either 'train' or 'eval'")

    def get_model_device(self):
        return next(self.model.parameters()).device

    def move_model_to_device(self, device):
        self.model.to(device)
        self.device = device



from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import torch

def calculate_ccc(y_true, y_pred):
    """Calculate Concordance Correlation Coefficient"""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred)**2)
    return ccc

def predict_with_saved_model(args):
    """Run prediction using a saved model on the test dataset with detailed metrics"""
    exp = CustomExpMain(args)
    
    # Load the saved model
    print(f'Loading model from {args.model_path}')
    checkpoint = torch.load(args.model_path, map_location=exp.device)
    if 'model_state_dict' in checkpoint:
        exp.model.load_state_dict(checkpoint['model_state_dict'])
    else:
        exp.model.load_state_dict(checkpoint)
    
    print('Model loaded successfully')
    
    test_data, test_loader = exp._get_data(flag='test')
    
    exp.model.eval()
    predictions = []
    video_names = []
    true_values = {
        'group_emotion': [],
        'group_emotion_category': [],
        'class_emotion': [],
        'class_valence': [],
        'class_arousal': [],
        'event_emotion': [],
        'event': []
    }
    
    print('Starting predictions...')
    with torch.no_grad():
        for i, (batch_x, batch_y, group_emotions, group_emotion_categories, 
                class_emotions, class_valences, class_arousals, 
                event_emotions, events) in enumerate(tqdm(test_loader)):
            
            # Get video file names
            batch_video_names = test_loader.dataset.valid_subfolders[i * test_loader.batch_size:
                                                                   (i + 1) * test_loader.batch_size]
            video_names.extend(batch_video_names)
            
            # Store true values
            true_values['group_emotion'].extend(group_emotions.numpy())
            true_values['group_emotion_category'].extend(group_emotion_categories.numpy())
            true_values['class_emotion'].extend(class_emotions.numpy())
            true_values['class_valence'].extend(class_valences.numpy())
            true_values['class_arousal'].extend(class_arousals.numpy())
            true_values['event_emotion'].extend(event_emotions.numpy())
            true_values['event'].extend(events.numpy())
            
            # Get predictions
            batch_x = batch_x.float().to(exp.device)
            outputs = exp.model(batch_x)
            
            _, group_emotion, group_emotion_category, class_emotion, \
            class_valence, class_arousal, event_emotion, event = outputs
            
            predictions.append({
                'group_emotion': torch.argmax(group_emotion, dim=1).cpu().numpy(),
                'group_emotion_category': torch.argmax(group_emotion_category, dim=1).cpu().numpy(),
                'class_emotion': torch.argmax(class_emotion, dim=1).cpu().numpy(),
                'class_valence': class_valence.squeeze().cpu().numpy(),
                'class_arousal': class_arousal.squeeze().cpu().numpy(),
                'event_emotion': torch.argmax(event_emotion, dim=1).cpu().numpy(),
                'event': torch.argmax(event, dim=1).cpu().numpy()
            })
    
    # Combine predictions
    combined_predictions = {
        'video_file_name': video_names,
        'group_emotion': np.concatenate([p['group_emotion'] for p in predictions]),
        'group_emotion_category': np.concatenate([p['group_emotion_category'] for p in predictions]),
        'class_emotion': np.concatenate([p['class_emotion'] for p in predictions]),
        'class_valence': np.concatenate([p['class_valence'].reshape(-1) for p in predictions]),
        'class_arousal': np.concatenate([p['class_arousal'].reshape(-1) for p in predictions]),
        'event_emotion': np.concatenate([p['event_emotion'] for p in predictions]),
        'event': np.concatenate([p['event'] for p in predictions])
    }
    
    # Create DataFrame
    df = pd.DataFrame(combined_predictions)
    
    # Normalize predictions and true values to 0-999 range
    df['class_valence'] = normalize_to_range(df['class_valence'].values)
    df['class_arousal'] = normalize_to_range(df['class_arousal'].values)
    true_values['class_valence'] = normalize_to_range(np.array(true_values['class_valence']))
    true_values['class_arousal'] = normalize_to_range(np.array(true_values['class_arousal']))
    
    # Calculate metrics
    metrics = {}
    
    print("\nEvaluation Metrics:")
    print("-" * 50)
    
    # Calculate accuracy for categorical predictions
    categorical_columns = ['group_emotion', 'group_emotion_category', 'class_emotion', 'event_emotion', 'event']
    for col in categorical_columns:
        accuracy = accuracy_score(true_values[col], combined_predictions[col])
        metrics[f'{col}_accuracy'] = accuracy
        print(f'{col.replace("_", " ").title()} Accuracy: {accuracy:.4f}')
    
    print("\nRegression Metrics:")
    print("-" * 50)
    
    # Calculate MSE and CCC for valence and arousal
    for col in ['class_valence', 'class_arousal']:
        mse = mean_squared_error(true_values[col], combined_predictions[col])
        ccc = calculate_ccc(true_values[col], combined_predictions[col])
        mae = mean_absolute_error(true_values[col], combined_predictions[col])
        
        metrics[f'{col}_mse'] = mse
        metrics[f'{col}_ccc'] = ccc
        metrics[f'{col}_mae'] = mae
        
        print(f'\n{col.replace("_", " ").title()}:')
        print(f'MSE: {mse:.4f}')
        print(f'MAE: {mae:.4f}')
        print(f'CCC: {ccc:.4f}')
    
    # Create results directory and save predictions
    os.makedirs('./predictions', exist_ok=True)
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    
    # Save predictions
    csv_path = f'./predictions/predictions_{model_name}.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nPredictions saved to {csv_path}')
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = f'./predictions/metrics_{model_name}.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f'Metrics saved to {metrics_path}')
    
    # Save true values
    true_values_df = pd.DataFrame({
        'video_file_name': video_names,
        **{k: v for k, v in true_values.items()}
    })
    true_values_path = f'./predictions/true_values_{model_name}.csv'
    true_values_df.to_csv(true_values_path, index=False)
    print(f'True values saved to {true_values_path}')
    
    # Save a comparison report
    comparison_data = {
        'video_file_name': video_names,
        **{f'{k}_pred': combined_predictions[k] for k in true_values.keys()},
        **{f'{k}_true': true_values[k] for k in true_values.keys()}
    }
    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = f'./predictions/comparison_{model_name}.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f'Detailed comparison saved to {comparison_path}')
    
    return df, metrics, true_values_df, comparison_df


def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
  
    parser = argparse.ArgumentParser(description='S3Attention for Emotion Prediction')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--task_id', type=str, default='emotion_prediction', help='task id')
    parser.add_argument('--model', type=str, default='S3Attention', help='model name, options: [S3Attention, Informer, Transformer]')
    parser.add_argument('--predict_mode', action='store_true', help='Run prediction using saved model')
    parser.add_argument('--model_path', type=str, help='Path to saved model for prediction')

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/media/main/Data/Abhinav_gems/S3Attention/Dataset', help='root path of the data directory')

    parser.add_argument('--face_features_folder', type=str, default='face_features', help='folder name for face features')
    parser.add_argument('--text_latent1_folder', type=str, default='text_latent1', help='folder name for text latent 1')
    parser.add_argument('--text_latent2_folder', type=str, default='text_latent2', help='folder name for text latent 2')
    parser.add_argument('--text_latent3_folder', type=str, default='text_latent3', help='folder name for text latent 3')
    parser.add_argument('--labels_file', type=str, default='labels.csv', help='filename for labels CSV')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
    parser.add_argument('--checkpoints', type=str, default='/media/main/Data/Abhinav_gems/S3Attention/saved_models_old_llm_latest_labelval_ar_new_loss', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=3328, help='encoder input size (1024 + 768 * 3)')
    parser.add_argument('--dec_in', type=int, default=3328, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=3328, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder', default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=2, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10000, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=300000, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.00005, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

    args = parser.parse_args()
    
    print('Args in experiment:')
    print(args)

    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA current device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    if args.predict_mode:
        if args.model_path is None:
            raise ValueError("Please provide --model_path when using --predict_mode")
        print("Running in prediction mode...")
        predict_with_saved_model(args)
        return

    Exp = CustomExpMain

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)

if __name__ == "__main__":
    main()

# def main():
#     fix_seed = 2021
#     random.seed(fix_seed)
#     torch.manual_seed(fix_seed)
#     np.random.seed(fix_seed)
  
#     parser = argparse.ArgumentParser(description='S3Attention for Emotion Prediction')

#     # basic config
#     parser.add_argument('--is_training', type=int, default=1, help='status')
#     parser.add_argument('--task_id', type=str, default='emotion_prediction', help='task id')
#     parser.add_argument('--model', type=str, default='S3Attention', help='model name, options: [S3Attention, Informer, Transformer]')

#     # data loader
#     parser.add_argument('--data', type=str, default='custom', help='dataset type')
#     parser.add_argument('--root_path', type=str, default='/media/main/Data/Abhinav_gems/S3Attention/Dataset_new_llm', help='root path of the data directory')
#     parser.add_argument('--face_features_folder', type=str, default='face_features', help='folder name for face features')
#     parser.add_argument('--text_latent1_folder', type=str, default='text_latent1', help='folder name for text latent 1')
#     parser.add_argument('--text_latent2_folder', type=str, default='text_latent2', help='folder name for text latent 2')
#     parser.add_argument('--text_latent3_folder', type=str, default='text_latent3', help='folder name for text latent 3')
#     parser.add_argument('--labels_file', type=str, default='labels.csv', help='filename for labels CSV')
#     parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
#     parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
#     parser.add_argument('--checkpoints', type=str, default='/media/main/Data/Abhinav_gems/S3Attention/saved_models_final_new_llm', help='location of model checkpoints')

#     # forecasting task
#     parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
#     parser.add_argument('--label_len', type=int, default=48, help='start token length')
#     parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

#     # model define
#     parser.add_argument('--enc_in', type=int, default=3328, help='encoder input size (1024 + 768 * 3)')
#     parser.add_argument('--dec_in', type=int, default=3328, help='decoder input size')
#     parser.add_argument('--c_out', type=int, default=3328, help='output size')
#     parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
#     parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
#     parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
#     parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
#     parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
#     parser.add_argument('--factor', type=int, default=1, help='attn factor')
#     parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
#     parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
#     parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
#     parser.add_argument('--activation', type=str, default='gelu', help='activation')
#     parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')

#     # optimization
#     parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
#     parser.add_argument('--itr', type=int, default=2, help='experiments times')
#     parser.add_argument('--train_epochs', type=int, default=10000, help='train epochs')
#     parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
#     parser.add_argument('--patience', type=int, default=300000, help='early stopping patience')
#     parser.add_argument('--learning_rate', type=float, default=0.00005, help='optimizer learning rate')
#     parser.add_argument('--des', type=str, default='test', help='exp description')
#     parser.add_argument('--loss', type=str, default='mse', help='loss function')
#     parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
#     parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

#     # GPU
#     parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
#     parser.add_argument('--gpu', type=int, default=0, help='gpu')
#     parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
#     parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
#     parser.add_argument('--do_predict', action='store_true', help='whether to run prediction after training')
#     parser.add_argument('--model_path', type=str, help='path to the saved full model for prediction')
#     parser.add_argument('--pred_data_path', type=str, help='path to the prediction data (if different from test data)')


#     args = parser.parse_args()
    
#     print('Args in experiment:')
#     print(args)

#     if torch.cuda.is_available():
#         print(f"CUDA available: {torch.cuda.is_available()}")
#         print(f"CUDA device count: {torch.cuda.device_count()}")
#         print(f"CUDA current device: {torch.cuda.current_device()}")
#         print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
#         torch.cuda.empty_cache()
    
#     Exp = CustomExpMain

#     if args.is_training:
#         for ii in range(args.itr):
#             # setting record of experiments
#             setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
#                 args.task_id,
#                 args.model,
#                 args.data,
#                 args.features,
#                 args.seq_len,
#                 args.label_len,
#                 args.pred_len,
#                 args.d_model,
#                 args.n_heads,
#                 args.e_layers,
#                 args.d_layers,
#                 args.d_ff,
#                 args.factor,
#                 args.embed,
#                 args.distil,
#                 args.des,
#                 ii)

#             exp = Exp(args)  # set experiments
#             print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
#             exp.train(setting)
            
#             print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#             exp.test(setting)
            
#             if hasattr(args, 'do_predict') and args.do_predict:
#                 print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#                 exp.predict(setting, True)
    
#     elif args.do_predict:
#         print('>>>>>>>predicting with saved model on validation set<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
#         exp = Exp(args)
        
#         # Use the validation data loader
#         _, val_loader = exp._get_data(flag='val')
        
#         # Make predictions and get metrics
#         predictions, metrics = exp.predict_and_evaluate_saved_model(args.model_path, val_loader)
        
#         # Save predictions
#         save_path = f'./predictions_val_{os.path.basename(args.model_path)}.pkl'
#         with open(save_path, 'wb') as f:
#             pickle.dump(predictions, f)
#         print(f'Predictions saved to {save_path}')
        
#         # Save metrics
#         metrics_path = f'./metrics_val_{os.path.basename(args.model_path)}.txt'
#         with open(metrics_path, 'w') as f:
#             for key, value in metrics.items():
#                 f.write(f"{key}: {value}\n")
#         print(f'Metrics saved to {metrics_path}')

#         # Generate and save plots for predictions
#         plot_dir = os.path.join(os.path.dirname(args.model_path), 'plots_val')
#         create_and_save_plots(predictions, metrics, plot_dir)
#         print(f'Prediction plots saved in {plot_dir}')

#         # Generate and save plots for original labels
#         labels_path = os.path.join(args.root_path, 'val', 'labels.csv')
#         label_plot_dir = os.path.join(os.path.dirname(args.model_path), 'plots_label')
#         plot_label_distributions(labels_path, label_plot_dir)

#     else:
#         ii = 0
#         setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.task_id,
#                                                                                                     args.model,
#                                                                                                     args.data,
#                                                                                                     args.features,
#                                                                                                     args.seq_len,
#                                                                                                     args.label_len,
#                                                                                                     args.pred_len,
#                                                                                                     args.d_model,
#                                                                                                     args.n_heads,
#                                                                                                     args.e_layers,
#                                                                                                     args.d_layers,
#                                                                                                     args.d_ff,
#                                                                                                     args.factor,
#                                                                                                     args.embed,
#                                                                                                     args.distil,
#                                                                                                     args.des, ii)
        
#         exp = Exp(args)  # set experiments
#         print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
#         # exp.test(setting, test=1)
#         # exp.test(setting=args.task_id, model_path=args.model_path)

    
# if __name__ == "__main__":
#     main()
 
