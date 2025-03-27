import numpy as np
import pandas as pd
import torch
from pathlib import Path
import pickle
from pytorch_pae import AE
from pytorch_pae import conditional_GIS
import os
from scipy import stats


### possible improvements
# Early stopping of training is only implemented for density estimator, not for Autoencoder. Add early stopping criterion for AE.
# Print out plot of training loss and validation loss for each model


class Spectra_PAE():
    """
    SpectraPAE class
    -----------------
    data_dir:  str, data location
    model_dir: str, model location (if models don't exist, models will be saved there)
    input_dim: tupel, data dimensionality in format (w,h,c) or (w,c)
    AE_network_type: one of 'fc' or 'conv'.'conv' is not tested for this class. (It is tested in the AE class, and will probably work here.)
    general_params: dict, general parameters for Autoencoder class. Please refer to the AE class. If none, default dictsionary used for publication will be loaded.
    fc_network_params: dict, parameters defining the network architecture. Please refer to the AE class. If none, default dictionary used for publication will be loaded.
    conv_network_params: dic, same as fc_network_params, just for AE_network_type='conv' 
    training_params: dict, training parameters for AE training. Please refer to the AE class. If none, default dictionary used for publication will be loaded.
    seed: random seed
    prefixes: dict, prefixes of model names, change prefixes to change the name under which the models are saved
    """
    def __init__(self, data_dir, model_dir, dataset_name, input_dim=None,  AE_network_type='fc', general_params=None, fc_network_params=None,
                 conv_network_params=None, training_params=None, seed=1234, prefixes={'AE1':'AE1', 'AE2':'AE2','NF1':'NF1','NF2':'NF2'}):
        
        
        self.device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir      = model_dir
        self.data_dir       = data_dir
        
        self.data_params    = {'dataset':dataset_name, 'loc': data_dir}
        
        if not general_params:
            self.general_params = pickle.load(open('/global/u2/v/vboehm/codes/Spectra_PAE/params/general_params.pkl','rb'))

        self.general_params['input_c']   = input_dim[-1]
        self.general_params['input_dim'] = input_dim[0]
        
        if not training_params:
            self.training_params = pickle.load(open('./params/training_params.pkl','rb'))
            
        if AE_network_type=='fc':
            if not fc_network_params:
                self.network_params = pickle.load(open('./params/fc_network_params.pkl','rb'))
                
        elif AE_network_type=='conv':
            if not conv_network_params:
                self.network_params = pickle.load(open('./params/conv_network_params.pkl','rb'))
        else:
            raise ValueError('network type not supported')
            
        self.seed = seed
        
        self.prefixes  = prefixes

    def set_seeds(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)        
        
    def train_complete_model(self, nepochs=100, retrain=False, use_prior=False, niter=500):
        """
        trains and saves complete model
        --------
        nepochs: number of epochs to train for (you could add an early stopping criteria on validation loss)
        retrain: whether to retrain even if model if files exist (better to change the name under which the models are saved by chanin the prefixes)
        use_prior: whether to use a prior when evaluating the class probability
        niter: number of training steps in the normalizing flow
        """
        
        self.set_seeds()
        self.setup_AE1()
        
        # train AE1
        if os.path.isfile(os.path.join(self.AE1.save_dir,'{}.ckpt'.format(self.AE1.name))) and (retrain==False):
            self.AE1.load_model(self.model_dir)
        self.train_AE1(nepochs=nepochs)
        
        # evaluate AE1 and save results
        for split in ['train', 'valid', 'test']:
            recons = self.evaluate_AE1(split)
            pickle.dump(recons,open(os.path.join(self.data_dir,'{}_recons_{}.pkl'.format(self.prefixes['AE1'],split)),'wb'))
            
        # setup and train AE2
        self.setup_AE2()
        if os.path.isfile(os.path.join(self.AE1.save_dir,'{}.ckpt'.format(self.AE2.name))) and (retrain==False):
            self.AE2.load_model(self.model_dir)
        self.train_AE2(nepochs=nepochs)
        
        # evaluate AE2
        for split in ['train', 'valid', 'test']:
            embeddings = self.get_AE2_encoded(split)
            pickle.dump(embeddings,open(os.path.join(self.data_dir,'{}_encoded_{}.pkl'.format(self.prefixes['AE2'],split)),'wb'))
            
        # train first conditional normalizing flow
        self.setup_NF1_input_data()
        if os.path.isfile(os.path.join(self.model_dir,self.prefixes['NF1'])) and (retrain==False):
            print('loading trained NF1...')
            self.NF1 = torch.load(os.path.join(self.model_dir,self.prefixes['NF1']))   
        else:
            self.train_conditional_density_stage1(max_iter=niter)
            
        if use_prior:
            prior=self.get_prior(self.labels['train'])
        else: 
            prior=None
        # reclassify spectra based on conditional density estimation (NF1)
        
        print('reclassify by identifying labels with highest probability under NF1...')
        self.new_labels = {}
        for split in ['train', 'valid', 'test']:
            self.new_labels[split] = self.NF1_classify(torch.squeeze(self.NF1_data[split]),prior=prior)
            
        # train second conditional normalzing flow with new labels
        if os.path.isfile(os.path.join(self.model_dir,self.prefixes['NF2'])) and (retrain==False):
            print('loading trained NF2...')
            self.NF2 = torch.load(os.path.join(self.model_dir,self.prefixes['NF2']))   
        else:
            self.train_conditional_density_stage2(max_iter=niter)
        
    def setup_AE1(self):
        """
        initializes first autoencoder
        """
        self.training_params['criterion1'] = 'masked_chi2'
        self.training_params['criterion2'] = 'masked_chi2'
        self.data_params
        self.AE1            = AE.Autoencoder(self.general_params, self.data_params, self.network_params, self.network_params, 
                                self.training_params, self.device, transforms=None, name=self.prefixes['AE1'], save_dir=self.model_dir)
        return True
    
    def train_AE1(self, nepochs):
        """
        trains first autoencoder
        nepochs: int, number of epochs to train
        """
        self.AE1.train()
        print('training AE stage 1...')
        self.AE1.train_model(nepochs)
        print('AE stage 1 training completed.')
        return True
    
    def evaluate_AE1(self, split):
        """
        reconstructs the data with the first autoencoder
        this step denoises and impaints the data
        split: str, data split. one of 'train', 'valid', 'test'
        """
        print('evaluating AE stage 1 {}...'.format(split))
        recons = []
        self.AE1.eval()
        if split == 'train':
            loader = self.AE1.train_loader
        elif split == 'valid':
            loader = self.AE1.valid_loader
        else:
            loader = self.AE1.test_loader
            
        for ii, data in enumerate(loader,0):
            data     = data['features']
            with torch.no_grad():
                data     = data.to(self.AE1.device).float()
                recon    = self.AE1.forward(data)
            recons.append(recon)
        recons = torch.cat(recons,axis=0)
        recons = np.asarray(recons.detach().cpu().numpy())
        recons = np.reshape(recons, (-1,self.general_params['input_dim'], self.general_params['input_c']))
        return recons
    
    def setup_AE2(self):
        """
        initializes second autoencoder
        """
        training_params = self.training_params
        training_params['criterion1'] = 'MSELoss'
        training_params['criterion2'] = 'MSELoss'
        data_params     = self.data_params
        data_params['dataset_name']   = 'AE1_encoded_spectra'
        self.AE2 = AE.Autoencoder(self.general_params, data_params, self.network_params, self.network_params, 
                              training_params, self.device, transforms=None, name=self.prefixes['AE2'], save_dir=self.model_dir)
        
        loc        = os.path.join(self.AE1.save_dir,'{}.ckpt'.format(self.AE1.name))
        checkpoint = torch.load(loc)
        self.AE2.load_state_dict(checkpoint['model_state_dict'])
        return True
    
    def train_AE2(self, nepochs):
        """
        trains second autoencoder
        nepochs: int, number of epochs to train
        """
        print('training AE stage 2...')
        self.AE2.train()
        self.AE2.train_model(nepochs)
        print('AE stage 2 training completed.')
        return True
    
    def get_AE2_encoded(self, split):
        """
        encodes data that was reconstructed with the first autoencoder with the second autoencoder
        split: str, data split. one of 'train', 'valid', 'test'
        """
        
        print('evaluating AE stage 2 {}...'.format(split))
        latent_vars = []
        self.AE2.eval()
        if split == 'train':
            loader = self.AE2.train_loader
        elif split == 'valid':
            loader = self.AE2.valid_loader
        else:
            loader = self.AE2.test_loader
        for ii, data in enumerate(loader,0):
            data     = data['features']
            with torch.no_grad():
                data     = data.to(self.AE2.device).float()
                latent   = self.AE2.encoder.forward(data)
            latent_vars.append(latent)
        latent_vars = torch.cat(latent_vars,axis=0)
        latent_vars = np.asarray(latent_vars.detach().cpu().numpy())
        latent_vars = latent_vars.reshape((-1,self.general_params['latent_dim'], self.general_params['input_c']))
        return latent_vars
    
    def setup_NF1_input_data(self):
        """
        converts encoded data (encoded with second autoencoder) to tensors and adds labels
        """
        
        self.NF1_data = {}
        self.labels = {}
        for split in ['train', 'valid', 'test']:
            self.NF1_data[split] = torch.Tensor(pickle.load(open(os.path.join(self.data_dir,'{}_encoded_{}.pkl'.format(self.prefixes['AE2'],split)),'rb'))).float().to(self.device)
            self.labels[split]   = torch.Tensor(pickle.load(open(os.path.join(self.data_dir,'{}_labels.pkl'.format(split)),'rb'))).long().to(self.device)
        self.labels['valid']     = torch.cat([self.labels['test'], self.labels['valid']])
        
        return True

        
    def train_conditional_density_stage1(self, max_iter=500):
        """
        trains the conditional density estimator on the AE2 encoded data
        max_iter: int, maximal number of training steps (note that this algorithm has an early stopping criterion!)
        """
        
        
        print('training Normalizing Flow Stage 1...')
        self.NF1= conditional_GIS.train_ConditionalGIS(torch.squeeze(self.NF1_data['train']),self.labels['train'],
                                                       torch.squeeze(self.NF1_data['valid']),self.labels['valid'], max_iter=max_iter)
        torch.save(self.NF1, os.path.join(self.model_dir,'NF1'))
        
        return True
        
    def train_conditional_density_stage2(self, max_iter=500):
        """
        trains the second conditional density estimator, after the data points have been reclassified
        """
        print('train Normalizing Flow Stage 2...')
        self.NF2 = conditional_GIS.train_ConditionalGIS(torch.squeeze(self.NF1_data['train']),torch.Tensor(self.new_labels['train']).to(self.device).long(),
                                                       torch.squeeze(self.NF1_data['valid']),torch.Tensor(self.new_labels['valid']).to(self.device).long(), max_iter=max_iter)
        
        torch.save(self.NF2, os.path.join(self.model_dir,'NF2'))
        
        return True
                  
    def NF1_classify(self, data, prior=None):
        """
        classifies data according to its highest probability label under the first conditional density estimator
        data: torch tensor of encoded data
        prior: numpy array of size number of classes (16). prior probability for each class. if given, the algorithms computes argmax[p(data|label)*p(label)]  
        """
        
        print('classifying...')
        px = np.zeros((16,len(data)))
        for ii in range(16):
            px[ii] = self.NF1.evaluate_density(data,ii*torch.ones((len(data))).to(self.device)).cpu().numpy()
        if np.any(prior):
            px*=prior[:,None]
        new_class = np.argmax(px,axis=0)
        
        return new_class
    
    def evaluate_NF1(self,data,labels):
        """
        evaluates the first conditional density estimator
        """
        if not torch.is_tensor(data):
            data   = torch.Tensor(data)
        data.to(self.device)
        if not torch.is_tensor(labels):
            labels = torch.Tensor(labels)
        labels.to(self.device).long()
        
        logp   = self.NF1.evaluate_density(torch.squeeze(data),torch.squeeze(labels)).cpu().numpy()
        return logp
        
        
    def evaluate_NF2(self,data,labels):
        """
        evaluates the second and final conditional density estimator
        """
        if not torch.is_tensor(data):
            data   = torch.Tensor(data)
        data.to(self.device)
        if not torch.is_tensor(labels):
            labels = torch.Tensor(labels)
        labels.to(self.device).long()
        
        logp   = self.NF2.evaluate_density(torch.squeeze(data),torch.squeeze(labels)).cpu().numpy()
        
        return logp
    
    
    def evaluate_logp_percentile(self,data, labels, datum,label):
        """
        compares logp to training dataset with same label
        returns percentile of data points in data with logp lover than datum (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.percentileofscore.html)
        data: reference data
        labels: labels for reference data
        
        datum: data point to rank, must be of shape (1,latent_dim)
        label: label for data point, must be of shape (1,1)
        
        """        
        
        if not torch.is_tensor(data):
            data   = torch.Tensor(data)
        data.to(self.device)
        
        if not torch.is_tensor(labels):
            labels = torch.Tensor(labels)
        labels.to(self.device).long()
        
        if not torch.is_tensor(datum):
            datum   = torch.Tensor(datum)
        datum.to(self.device)
        
        if not torch.is_tensor(label):
            label = torch.Tensor(label)
        label.to(self.device).long()
        
        logp    = self.NF2.evaluate_density(torch.squeeze(torch.cat((datum,datum),axis=0)),torch.squeeze(torch.cat((label,label),axis=0))).cpu().numpy()
        logps   = self.NF2.evaluate_density(torch.squeeze(data),torch.squeeze(labels)).cpu().numpy()
        
        precentile = stats.percentileofscore(logps[labels.cpu()==label.cpu()],logp[0], kind='rank')
        
        return precentile
        
    def get_prior(self,labels):
        """
        computes prior probability in a frequentist fashion from given labels
        """
        print('computing prior probabilities...')
        _, counts = np.unique(labels.detach().cpu().numpy(), return_counts=True)
        
        return counts/np.sum(counts)