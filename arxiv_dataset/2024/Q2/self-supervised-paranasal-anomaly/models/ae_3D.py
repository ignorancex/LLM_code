import torch
from torch import nn
from torch.nn import functional as F
from utils.utils_eval import _test_step, get_eval_dictionary, _test_end,_save_predicted_volume
from models.losses import L1_AE
import numpy as np
import pytorch_lightning as pl
import torchvision.models as models
import torch.optim as optim
from typing import Any, List
from imageio import imwrite
import torchio as tio

class BaseAE_3D(nn.Module):


    def __init__(self,
                 cfg,
                 **kwargs) -> None:
        super(BaseAE_3D, self).__init__()

        self.latent_dim = cfg["latent_size"]

        modules = []
        kernel_size = []
        self.batch_size = cfg["batch_size"]
        self.filters = cfg["filters"]
        filter_multiplier = cfg["filter_multiplier"]
        self.dimension = cfg["dimension"]
        self.fc_input_dim = cfg["fc_input_dim"]
        self.spatial_ae = cfg["spatial_ae"]

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

        if not self.spatial_ae:
            conv_1x1 = nn.Conv3d(self.filters[-1], out_channels=self.filters[-1]//16,
                                kernel_size= 1, stride= 1, padding  = 0)
            modules.append(conv_1x1)
            self.fc = nn.Linear(self.fc_input_dim, self.latent_dim)
        self.encoder = nn.Sequential(*modules)
        #self.fc = nn.Linear(self.filters[-1]*filter_multiplier, self.latent_dim)
        


        # Build Decoder
        modules = []
        if not self.spatial_ae:
            modules.append(nn.Conv3d(self.filters[-1]//16, out_channels=self.filters[-1],
                                kernel_size= 1, stride= 1, padding  = 0))

        #self.decoder_input = nn.Linear(self.latent_dim, self.filters[-1] * filter_multiplier)
            self.decoder_input = nn.Linear(self.latent_dim, self.fc_input_dim)

        self.filters.reverse()

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
        preflattened_result = self.encoder(input)

        
        if not self.spatial_ae:
            result = torch.flatten(preflattened_result, start_dim=1)
            z = self.fc(result)
        else: 
            
            z =  preflattened_result
        
        return [z,preflattened_result.shape,input.shape]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        if not self.spatial_ae:
            result = self.decoder_input(z[0])
        
            result = torch.reshape(result, (z[1][0],z[1][1],z[1][2],z[1][3],z[1][4]))
            result = self.decoder(result)

        else: 

            result = self.decoder(z[0])

        #Matches the resolution of the input image. Otherwise loss calculation not possible!
        result = torch.nn.functional.interpolate(result, size=(z[2][2],z[2][3],z[2][4]), mode='nearest')
        
        return result

    
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        
        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss 
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach()}





class Autoencoder_3D(pl.LightningModule):

    def __init__(self,cfg,prefix=None):
        
        super(Autoencoder_3D, self).__init__()
        self.cfg = cfg
        # Model 
        self.AE = BaseAE_3D(cfg)  
        
        print(self.AE)
        # Loss function
        self.criterion = L1_AE(cfg)
        self.prefix = prefix
        self.save_hyperparameters()

    def forward(self, x):
        z = self.AE.encode(x)
        
        y = self.AE.decode(z)
        return {"image":y,"z":z[0]}

    def prepare_batch(self, batch):

        return {"image": batch['one_image'][tio.DATA],
                "org_image": batch['org_image'][tio.DATA],
                "label": batch['label'],
                "disease_label": batch['disease_label'],
                "patient_disease_id":batch['patient_disease_id'],
                "image_path":batch['image_path'],
                "smax":batch['smax'],
                "crop_size":batch['crop_size'],} 
    
    def training_step(self, batch, batch_idx: int):


        return_object = self.prepare_batch(batch)

        inputs = return_object['image']
        outputs  = self.forward(inputs)
        # print(inputs.shape)
        target = return_object['org_image']
        #print(outputs["image"].shape,inputs.shape)
        # calculate loss
        loss = self.criterion(outputs["image"],target)
        z = outputs['z'].detach()

        self.log(f'train/loss',loss["recon_error"], prog_bar=False, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)
        return {"loss": loss["recon_error"], 'latent_space': z}
    

        
    

    def validation_step(self, batch: Any, batch_idx: int):

        # process batch
        return_object = self.prepare_batch(batch)
        inputs = return_object['image']
        outputs  = self.forward(inputs)
        # print(inputs.shape)
        target = return_object['org_image']
        #print(outputs["image"].shape,inputs.shape)
        # calculate loss
        loss = self.criterion(outputs["image"],target)

        
        # log val metrics
        self.log(f'val/loss',loss["recon_error"], prog_bar=True, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)
        return {"loss": loss["recon_error"]}

        

    def on_test_start(self):
        self.eval_dict = get_eval_dictionary()
        

    def test_step(self, batch: Any, batch_idx: int):

        
        return_object = self.prepare_batch(batch)
        
        inputs = return_object['image']
        outputs  = self.forward(inputs)
        # print(inputs.shape)
        target = return_object['org_image']
        label = return_object['label']
        #Latent vector
        z = torch.squeeze(outputs['z'].detach())
        if self.cfg.spatial_ae: 
            z = None
        patient_disease_id = return_object['patient_disease_id']
        smax = return_object['smax']
        crop_size = return_object['crop_size']
        disease_target = return_object['disease_label']
        #print(outputs["image"].shape,inputs.shape)
        # calculate loss
        loss = self.criterion(outputs["image"],target)

        
        AnomalyScoreReco_vol = loss['recon_error'].item()
        AnomalyScoreReco_volL2 = loss['recon_error_L2'].item()

        AnomalyScoreComb_vol = loss['combined_loss'].item()


        self.eval_dict['smax'].append(smax[0])
        self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
        self.eval_dict['AnomalyScoreCombiPerVol'].append(AnomalyScoreComb_vol)
        self.eval_dict['AnomalyScoreRecoPerVolL2'].append(AnomalyScoreReco_volL2)


        self.eval_dict['AnomalyScoreCombiPerVolL2'].append(0)
        #Dummy values put in place to prevent code from breaking
        self.eval_dict['AnomalyScoreRegPerVol'].append(0)
        self.eval_dict['AnomalyScoreCombPriorPerVol'].append(0)
        self.eval_dict['AnomalyScoreCombPriorPerVolL2'].append(0)
        self.eval_dict['KLD_to_learned_prior'].append(0)
        # calculate metrics
        _save_predicted_volume(self, self.cfg["save_folder"], outputs["image"],target,return_object['image_path'])
        #_test_step(self, outputs["image"],z,target,label[0],disease_target[0],patient_disease_id[0],crop_size[0],batch_idx,smax=smax[0],noisy_img=inputs) # everything that is independent of the model choice

        

        
           
    def on_test_end(self) :
        # calculate metrics
        _test_end(self) # everything that is independent of the model choice 


    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)
    
    def update_prefix(self, prefix):
        self.prefix = prefix 
