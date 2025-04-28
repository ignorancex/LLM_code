import torch
from torch import nn
from torch.nn import functional as F
#from .types_ import *
import torch
from torch import nn
from torch.nn import functional as F
from utils.utils_eval import _test_step, get_eval_dictionary, _test_end
from models.losses import L1_AE, L1_VAE, kld_gauss
import numpy as np
from pytorch_lightning.core.lightning import LightningModule
import torchvision.models as models
import torch.optim as optim
from typing import Any, List
from imageio import imwrite
import torchio as tio



class BaseVAE_3D(nn.Module):


    def __init__(self,
                 cfg,
                 **kwargs) -> None:
        super(BaseVAE_3D, self).__init__()

        self.latent_dim = cfg["latent_size"]

        modules = []
        kernel_size = []
        self.filters = cfg["filters"]
        self.batch_size = cfg["batch_size"]
        filter_multiplier = cfg["filter_multiplier"]
        self.dimension = cfg["dimension"]
        in_channels = self.filters[0]

        # Build Encoder
        for h_dim in self.filters[1:]:
            modules.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels=h_dim,
                              kernel_size= 5, stride= 2, padding  = 2),
                    nn.BatchNorm3d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        
        conv_1x1 = nn.Conv3d(self.filters[-1], out_channels=self.filters[-1]//16,
                              kernel_size= 1, stride= 1, padding  = 0)
        modules.append(conv_1x1)
        self.encoder = nn.Sequential(*modules)
        
        self.fc_mu = nn.Linear(self.filters[-1]*filter_multiplier, self.latent_dim)
        self.fc_var = nn.Linear(self.filters[-1]*filter_multiplier, self.latent_dim)

        # Build Decoder
        modules = []
        modules.append(nn.Conv3d(self.filters[-1]//16, out_channels=self.filters[-1],
                              kernel_size= 1, stride= 1, padding  = 0))

        self.decoder_input = nn.Linear(self.latent_dim, self.filters[-1] * filter_multiplier)

        self.filters.reverse()
        """
        #Creates checkerboard patterns
        for i in range(len(self.filters[:-1]) ):
            modules.append(

                nn.Sequential(
                    nn.ConvTranspose3d(self.filters[i],
                                       self.filters[i + 1],
                                       kernel_size=2,
                                       stride = 2,
                                       padding=0),
                    nn.BatchNorm3d(self.filters[i + 1]),
                    nn.LeakyReLU(),
                   )

            )
        """

        for i in range(len(self.filters[:-1]) ):
            modules.append(
                nn.Sequential(
                    nn.Conv3d(self.filters[i],
                                       self.filters[i + 1],
                                       kernel_size=5,
                                       stride = 1,
                                       padding=2),
                    nn.BatchNorm3d(self.filters[i + 1]),
                    nn.LeakyReLU(),
                    nn.Upsample(scale_factor= 2,mode='trilinear')
                   )
            )



        self.decoder = nn.Sequential(*modules)

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        
        result = self.decoder_input(z)
        result = result.view(result.shape[0], 4, 8, 8, 8)
        result = self.decoder(result)
        
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        x_mu, x_logvar = self.encode(input)
        z_sampled = self.reparameterize(x_mu, x_logvar)
        x_hat = self.decode(z_sampled)

        outputs = {}
        outputs['x_hat'] = x_hat 
        outputs['mu'] = x_mu 
        outputs['logvar'] = x_logvar
        outputs['z'] = z_sampled
        return  outputs


class VariationalAutoencoder_3D(LightningModule):

    def __init__(self,cfg,prefix=None):
        
        super(VariationalAutoencoder_3D, self).__init__()
        self.cfg = cfg
        # Model 
        self.VAE = BaseVAE_3D(cfg)  

        # Loss function
        self.criterion = L1_VAE(cfg)
        self.prefix = prefix
        self.save_hyperparameters()

    def forward(self, x):
        output = self.VAE(x)
        
        return output

    def prepare_batch(self, batch):

        return batch['image'][tio.DATA],batch['label'],batch['patient_disease_id']

    def training_step(self, batch, batch_idx: int):
        # process batch
        inputs,y,_ = self.prepare_batch(batch)
        outputs = self.forward(inputs)
        
        # calculate loss
        loss = self.criterion(outputs,inputs)
        loss_kld = loss['reg']
        loss_reco = loss['recon_error']
        loss_combined= loss['combined_loss']
        loss = loss_combined
 
        z = outputs['z'].detach()
        self.log(f'{self.prefix}train/Loss_KLD', loss_kld, prog_bar=False, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)
        self.log(f'{self.prefix}train/Loss_Reco', loss_reco, prog_bar=False, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)
        self.log(f'{self.prefix}train/loss', loss_combined, prog_bar=False, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)
        
        
        return {"loss": loss, 'latent_space': z}
    

    def training_epoch_end(self, outputs ) -> None:
        zs = []
        [zs.append(x['latent_space']) for x in outputs]
        z_mean = torch.mean(torch.cat(zs),0)
        z_std = torch.std(torch.cat(zs),0)
        self.log(f'{self.prefix}train/latent_space_mean', z_mean, prog_bar=False, on_step=False, on_epoch=True,sync_dist=True)
        self.log(f'{self.prefix}train/latent_space_std', z_std, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.z_m = z_mean.cpu()
        self.z_std = z_std.cpu()
        return super().training_epoch_end(outputs)

    def validation_step(self, batch: Any, batch_idx: int):

        # process batch
        inputs,y,_ = self.prepare_batch(batch)
        outputs = self.forward(inputs)
        
        # calculate loss
        loss = self.criterion(outputs,inputs)
        loss_kld = loss['reg']
        loss_reco = loss['recon_error']
        loss_combined = loss['combined_loss']

        
        
        loss = loss_combined
 
        z = outputs['z'].detach()
        self.log(f'{self.prefix}val/Loss_KLD', loss_kld, prog_bar=False, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)
        self.log(f'{self.prefix}val/Loss_Reco', loss_reco, prog_bar=False, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)
        self.log(f'{self.prefix}val/loss', loss_combined, prog_bar=False, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)
        
        return {"loss": loss}

    def on_test_start(self):
        self.eval_dict = get_eval_dictionary()
        

    def test_step(self, batch: Any, batch_idx: int):

        inputs,y,patient_disease_id = self.prepare_batch(batch)
        outputs = self.forward(inputs)

        # Store latent space
        latent_vector = outputs['z'].cpu().squeeze()
        z_mu_sample = outputs['z'].cpu().squeeze()
        z_sd_sample = 1
        KLD_to_learned_prior = kld_gauss(z_mu_sample,z_sd_sample,self.z_m,self.z_std).mean()
        self.eval_dict['KLD_to_learned_prior'].append(KLD_to_learned_prior)
        # calculate loss and Anomalyscores
        loss = self.criterion(outputs,inputs)
        AnomalyScoreComb_vol = loss['combined_loss'].item()
        AnomalyScoreReg_vol = loss['reg'].item()
        AnomalyScoreReco_vol = loss['recon_error'].item()

        AnomalyScoreReco_volL2 = loss['recon_error_L2'].item()
        AnomalyScoreComb_volL2 = loss['combined_loss_L2'].item()

        
        self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
        self.eval_dict['AnomalyScoreRecoPerVolL2'].append(AnomalyScoreReco_volL2)
        self.eval_dict['AnomalyScoreRegPerVol'].append(AnomalyScoreReg_vol)
        self.eval_dict['AnomalyScoreCombiPerVol'].append(AnomalyScoreComb_vol)
        self.eval_dict['AnomalyScoreCombiPerVolL2'].append(AnomalyScoreComb_volL2)
        self.eval_dict['AnomalyScoreCombPriorPerVol'].append(AnomalyScoreReco_vol + self.cfg.beta * KLD_to_learned_prior)
        self.eval_dict['AnomalyScoreCombPriorPerVolL2'].append(AnomalyScoreReco_volL2 + self.cfg.beta * KLD_to_learned_prior)


        final_volume = outputs['x_hat']
        z = outputs['z']

        

        # calculate metrics 
        # everything that is independent of the model choice
        _test_step(self, final_volume,z,inputs,y,patient_disease_id[0],batch_idx) 

           
    def on_test_end(self) :
        # calculate metrics
        # everything that is independent of the model choice 
        _test_end(self) 


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)
    
    def update_prefix(self, prefix):
        self.prefix = prefix 
