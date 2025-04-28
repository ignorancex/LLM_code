import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from gvp.models import ThreeD_Protein_Model

def gaussian_analytical_kl(mu1, logsigma1, mu2, logsigma2):
    return -0.5 + logsigma2 - logsigma1 + 0.5 * (logsigma1.exp() ** 2 + (mu1 - mu2) ** 2) / (logsigma2.exp() ** 2)


class ThreeD_Conditional_VAE(pl.LightningModule):
    def __init__(self, latent_dim, checkpoint_path = None, freeze = False):
        super().__init__()
        

        save_model = './checkpoint/selfies_vae_model_020.pt'
        self.vae_model = torch.load(save_model)
        self.vae_model = self.vae_model

        self.protein_model = ThreeD_Protein_Model(node_in_dim = (6,3), node_h_dim = (128, 32), edge_in_dim = (32, 1), edge_h_dim=(32, 1), 
                                                  num_layers = 3, drop_rate=0.1)
    
        self.prior_network = nn.Sequential(
            nn.Linear(128 + 1280, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, latent_dim * 2)
        )
        self.latent_dim = latent_dim
        
        self.checkpoint_path = checkpoint_path
        self.freeze = freeze

    def encode(self, batch):
        cond = batch[0]
        input_ids = batch[1]
        seq = batch[2]
        with torch.no_grad():
            mu, log_var = self.vae_model.forward_encoder(input_ids)

        x_prot = self.protein_model((cond.node_s, cond.node_v), 
                                    cond.edge_index, (cond.edge_s, cond.edge_v), seq, cond.batch)
        prior = self.prior_network(x_prot).view(-1, 2, self.latent_dim)
        prior_mu, prior_log_var = prior[:, 0, :], prior[:, 1, :]
        
        log_std = torch.exp(0.5 * log_var)
        prior_std = torch.exp(0.5 * prior_log_var)
        eps = torch.randn_like(prior_std)
        z = mu + eps * log_std
        return z, mu, log_var, prior_mu, prior_log_var

    def decode(self, z, cond):
        return self.vae_model.sample(n_batch = z.shape[0], z = z)
    
    def forward(self, batch):
        input_ids = batch[1]
        z,  mu, log_var, prior_mu, prior_log_var = self.encode(batch)
        return self.vae_model.forward_decoder(input_ids, z), z, mu, log_var, prior_mu, prior_log_var
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        return {'optimizer': optimizer}
    
    def loss_function(self, pred, target, mu, log_var, prior_mu, prior_log_var, batch_size, p):
        #kld = gaussian_analytical_kl(mu, log_var, prior_mu, prior_log_var).sum() / batch_size
        kld = F.mse_loss(prior_mu, mu) + F.mse_loss(prior_log_var, log_var)
        return kld
    
    def training_step(self, train_batch, batch_idx):
        recon_loss, z, mu, log_var, prior_mu, prior_log_var = self(train_batch)
        input_ids = train_batch[0].view(-1)
        kl_loss = self.loss_function(None, input_ids,mu, log_var, prior_mu, prior_log_var, len(train_batch), None)
        loss = recon_loss + kl_loss
        self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
        
    def validation_step(self, val_batch, batch_idx):
        out, z, mu, log_var, prior_mu, prior_log_var = self(val_batch)
        input_ids = val_batch[0].view(-1)
        loss = self.loss_function(out, input_ids,mu, log_var, prior_mu, prior_log_var, len(val_batch), None)
        self.log('val_loss', loss)
        return loss