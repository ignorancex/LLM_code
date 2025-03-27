from attrdict import AttrDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

from third_party.genesis_v2.modules.unet import UNet
import third_party.genesis_v2.modules.attention as attention
import third_party.genesis_v2.modules.blocks as B
from third_party.genesis_v2.monet_config import MONet
from sri.utils import init_weights, mvn, std_mvn


class AutoregressivePrior(nn.Module):
    def __init__(self, scene_latent_dim, feat_dim):
        super(AutoregressivePrior, self).__init__()
        self.scene_encoder = nn.Sequential(
            nn.Linear(scene_latent_dim, feat_dim),
            nn.GELU()
        )
        self.lstm = nn.LSTM(feat_dim, 4*feat_dim)
        self.sp = nn.Linear(4*feat_dim, feat_dim)
        self.mu = nn.Linear(4*feat_dim, feat_dim)
        init_weights(self.scene_encoder, 'xavier')
        init_weights(self.lstm, 'xavier')
        init_weights(self.sp, 'xavier')
        init_weights(self.mu, 'xavier')


    def KL_loss(self, s, z, q_z):
        """
        s ~ q(s | x)
        z ~ q(z_o1:K | s, x) is a List 
        q_z is a List of mvn, the autoregressive posteriors
        """
        s = self.scene_encoder(s)
        # Throw out the last mean for autoregressive processing
        z = torch.stack(z[:-1],0)  # [num_slots-1,batch_size,z_size]
        z = torch.cat((s.unsqueeze(0), z), 0) # [num_slots, batch_size, z_size]
        state = None
        p_zs = []
        for step in range(z.shape[0]):
            lstm_out, state = self.lstm(z[step].unsqueeze(0), state)
            mu = self.mu(lstm_out[0])
            sp = self.sp(lstm_out[0])
            p_zs += [mvn(mu, sp)]
        kl = 0
        for q,p in zip(q_z, p_zs):
            kl += torch.distributions.kl.kl_divergence(q, p)
        return kl

    def forward(self, S, num_slots, t=1.0):
        """
        S is a scene latent sample, [batch_size, scene_latent_dim]
        num_slots is int 
        Xs are for computing ordered posterior X --> permuted means
        """
        input = self.scene_encoder(S)
        input = input.unsqueeze(0)  # [1,batch_size,z_size]
        state = None
        zs, p_zs = [], []

        for _ in range(num_slots):
            lstm_out, state = self.lstm(input, state)
            mu = self.mu(lstm_out[0])
            sp = self.sp(lstm_out[0])
            p_z = mvn(mu, sp, temperature=t)
            p_zs += [p_z]
            z = p_z.rsample()
            zs += [z]
            input = z.unsqueeze(0)  # [1, batch_size, z_size]

        return zs, p_zs


class AutoregressivePosterior(nn.Module):
    def __init__(self, scene_latent_dim, feat_dim):
        super(AutoregressivePosterior, self).__init__()
        self.scene_encoder = nn.Sequential(
            nn.Linear(scene_latent_dim, feat_dim),
            nn.GELU()
        )
        self.lstm = nn.LSTM(feat_dim, 4*feat_dim)
        self.sp = nn.Linear(4*feat_dim, feat_dim)
        init_weights(self.scene_encoder, 'xavier')
        init_weights(self.lstm, 'xavier')
        init_weights(self.sp, 'xavier')

    
    def forward(self, S, X):
        """
        S is a scene latent sample, [batch_size, scene_latent_dim]
        Xs are for computing ordered posterior X --> permuted means
        """
        S = self.scene_encoder(S)
        # Throw out the last mean for autoregressive processing
        X = X[:,:-1] 
        X = torch.cat((S.unsqueeze(1), X), 1) # [batch_size, num_slots, z_size]
        X = X.permute(1,0,2).contiguous()  # [num_slots, batch_size, z_size]
        X, _ = self.lstm(X)
        X = X.permute(1,0,2).contiguous()  # [batch_size, num_slots, 4*z_size]
        return self.sp(X)


class SceneEncoder(nn.Module):
    """
    Encode an orderless set of object slots into a Gaussian distribution
    for a single distributed scene representations using self-attention
    """
    def __init__(self, feat_dim, scene_latent_dim, scene_encoder_layers):
        super(SceneEncoder, self).__init__()
        self.scene_latent_dim = scene_latent_dim
        self.scale = feat_dim ** -0.5
        self.blocks = nn.ModuleList()
        for _ in range(scene_encoder_layers):
            self.blocks += [RelationAttentionNet(in_dim=feat_dim, out_dim=feat_dim)]

        self.project_to_scene_dim = nn.Sequential(
            nn.Linear(feat_dim, scene_latent_dim),
            nn.GELU()
        )
        self.mu_s = nn.Linear(scene_latent_dim, scene_latent_dim)
        self.sp_s = nn.Linear(scene_latent_dim, scene_latent_dim)

        init_weights(self.project_to_scene_dim, 'xavier')
        init_weights(self.mu_s, 'xavier')
        init_weights(self.sp_s, 'xavier')

    def forward(self, X):
        """
        X are slots [batch_size, K, z_size]
        """  
        for block in self.blocks:
            X = block(X)

        X = self.project_to_scene_dim(torch.sum(X,1)) # [N,K,D] --> [N,scene_dim] 
        q_s_mu = self.mu_s(X)
        q_s_sp = self.sp_s(X)
        q_scene = mvn(q_s_mu, q_s_sp)
        return q_scene


class RelationAttentionNet(nn.Module):
    """
    Assumes a fully connected graph.
    """
    def __init__(self, in_dim, out_dim):
        super(RelationAttentionNet, self).__init__()
        self.scale = out_dim ** -.5
        self.k = nn.Linear(in_dim, out_dim, bias = False)
        self.q = nn.Linear(in_dim, out_dim, bias = False)
        # relational v 
        self.relational_v = nn.Sequential(
            nn.Linear(2*in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
        init_weights(self.k, 'xavier')
        init_weights(self.q, 'xavier')
        init_weights(self.relational_v, 'xavier')
        self.norm_pre_ff = nn.LayerNorm(out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
        init_weights(self.mlp, 'xavier')
        
        
    def forward(self, x):
        """
        x is a FloatTensor of shape [N, K, D]
 
        """
        K = x.shape[1]
        x_ = x.clone()
        r = x.unsqueeze(2).repeat([1,1,K,1]) - x.unsqueeze(1).repeat([1,K,1,1])  # [N,K,K,D]
        v = self.relational_v(torch.cat((x.unsqueeze(2).repeat([1,1,K,1]), r), -1))  # [N,K,K,D]
        k, q = self.k(x), self.q(x)
        q *= self.scale     
        dots = torch.einsum('bid,bjd->bij', q, k)  # [batch_size, K, K]
        attn = dots.softmax(dim=2)  # attn is [batch_size, K, K]
        #out = torch.einsum('bijd,bij->bjd', v, attn)  # [batch_size, K, D]
        x = torch.einsum('bij,bijd->bid', attn, v)
        x = x_ + self.mlp(self.norm_pre_ff(x_ + x))  # [batch_size, K, D]
        return x


class GenesisV2(nn.Module):
    def __init__(self, K, img_size, feat_dim, semiconv, kernel, dynamic_K, debug,
                 pixel_bound, pixel_std, image_likelihood):
        super(GenesisV2, self).__init__()
        # Encoder
        self.K_steps = K
        self.dynamic_K = dynamic_K
        self.debug = debug
        self.pixel_bound = pixel_bound
        self.img_size = img_size
        self.image_likelihood = image_likelihood
        self.encoder = UNet(
            num_blocks=int(np.log2(img_size)-1),
            img_size=img_size,
            filter_start=min(feat_dim, 64),
            in_chnls=3,
            out_chnls=feat_dim,
            norm='gn')
        self.encoder.final_conv = nn.Identity()
        self.att_process = attention.InstanceColouringSBP(
            img_size=img_size,
            kernel=kernel,
            colour_dim=8,
            K_steps=self.K_steps,
            feat_dim=feat_dim,
            semiconv=semiconv)
        self.seg_head = B.ConvGNReLU(feat_dim, feat_dim, 3, 1, 1)
        self.feat_head = nn.Sequential(
            B.ConvGNReLU(feat_dim, feat_dim, 3, 1, 1),
            nn.Conv2d(feat_dim, 2*feat_dim, 1))
        self.z_head = nn.Sequential(
            nn.LayerNorm(2*feat_dim),
            nn.Linear(2*feat_dim, 2*feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2*feat_dim, 2*feat_dim))
        # Decoder
        c = feat_dim
        self.decoder_module = nn.Sequential(
            B.BroadcastLayer(img_size // 16),
            nn.ConvTranspose2d(feat_dim+2, c, 5, 2, 2, 1),
            nn.GroupNorm(8, c), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, c, 5, 2, 2, 1),
            nn.GroupNorm(8, c), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, min(c, 64), 5, 2, 2, 1),
            nn.GroupNorm(8, min(c, 64)), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(min(c, 64), min(c, 64), 5, 2, 2, 1),
            nn.GroupNorm(8, min(c, 64)), nn.ReLU(inplace=True),
            nn.Conv2d(min(c, 64), 4, 1))
        self.std = pixel_std

    def forward(self, x, use_normalized_nll=False):
        batch_size, _, H, W = x.shape

        # --- Extract features ---
        enc_feat, _ = self.encoder(x)
        enc_feat = F.relu(enc_feat)

        # --- Predict attention masks ---
        if self.dynamic_K:
            if batch_size > 1:
                # Iterate over individual elements in batch
                log_m_k = [[] for _ in range(self.K_steps)]
                att_stats, log_s_k = None, None
                for f in torch.split(enc_feat, 1, dim=0):
                    log_m_k_b, _, _ = self.att_process(
                        self.seg_head(f), self.K_steps-1, dynamic_K=True)
                    for step in range(self.K_steps):
                        if step < len(log_m_k_b):
                            log_m_k[step].append(log_m_k_b[step])
                        else:
                            log_m_k[step].append(-1e10*torch.ones([1, 1, H, W]))
                for step in range(self.K_steps):
                    log_m_k[step] = torch.cat(log_m_k[step], dim=0)
                if self.debug:
                    assert len(log_m_k) == self.K_steps
            else:
                log_m_k, log_s_k, att_stats = self.att_process(
                    self.seg_head(enc_feat), self.K_steps-1, dynamic_K=True)
        else:
            
            log_m_k, log_s_k, att_stats = self.att_process(
                self.seg_head(enc_feat), self.K_steps-1, dynamic_K=False)
            if self.debug:
                assert len(log_m_k) == self.K_steps
            
        # -- Object features, latents, and KL
        comp_stats = AttrDict(mu_k=[], sigma_k=[], z_k=[], kl_l_k=[], q_z_k=[])
        
        for log_m in log_m_k:
            mask = log_m.exp()
            # Masked sum
            obj_feat = mask * self.feat_head(enc_feat)
            obj_feat = obj_feat.sum((2, 3))
            # Normalise
            obj_feat = obj_feat / (mask.sum((2, 3)) + 1e-5)
            # Posterior
            mu, sigma_ps = self.z_head(obj_feat).chunk(2, dim=1)
            sigma = B.to_sigma(sigma_ps)
            q_z = Normal(mu, sigma)
            z = q_z.rsample()
            comp_stats['mu_k'].append(mu)
            comp_stats['sigma_k'].append(sigma)
            comp_stats['z_k'].append(z)
            comp_stats['q_z_k'].append(q_z)

        # --- Decode latents ---
        recon, x_r_k, log_m_r_k = self.decode_latents(comp_stats.z_k)

        mx_r_k = [x*logm.exp() for x, logm in zip(x_r_k, log_m_r_k)]
        
        # Track quantities of interest
        stats = AttrDict(
            recon=recon, log_m_k=log_m_k, log_s_k=log_s_k, x_r_k=x_r_k,
            log_m_r_k=log_m_r_k, mx_r_k=mx_r_k,
            instance_seg=torch.argmax(torch.cat(log_m_k, dim=1), dim=1),
            instance_seg_r=torch.argmax(torch.cat(log_m_r_k, dim=1), dim=1))

        
        losses = AttrDict()
        # -- Reconstruction loss
        if self.image_likelihood == 'MoG':
            genv2_nll = SRI.gmm_negative_loglikelihood(x,
                            log_m_r_k, x_r_k, self.std, use_normalized_nll)
        elif self.image_likelihood == 'MoG-Normed':
            genv2_nll = SRI.gmm_negative_loglikelihood(x,
                            log_m_r_k, x_r_k, self.std, use_normalized_nll=True)
        
        # -- Component KL
        losses['kl_l_k'], _ = GenesisV2.mask_latent_loss(
            comp_stats.q_z_k, comp_stats.z_k,
            debug=self.debug)

                
        # Store losses
        losses['genv2_err'] = genv2_nll

        return recon, losses, stats, comp_stats, att_stats


    def decode_latents(self, z_k, the_decoder=None):
        if the_decoder is None:
            the_decoder = self.decoder_module
        # --- Reconstruct components and image ---
        x_r_k, m_r_logits_k = [], []
        z_k_batched = torch.stack(z_k)  # [K, batch_size, z_dim]
        batch_size = z_k_batched.shape[1]
        z_k_batched = z_k_batched.view(self.K_steps * batch_size, -1)
        #for z in z_k:
        #    dec = the_decoder(z)
        #    x_r_k.append(dec[:, :3, :, :])
        #    m_r_logits_k.append(dec[:, 3: , :, :])
        dec = the_decoder(z_k_batched)
        x_r = dec[:,:3].view(self.K_steps, batch_size, 3, self.img_size, self.img_size)
        m_r_logits = dec[:,3:].view(self.K_steps, batch_size, 1, self.img_size, self.img_size)
        x_r_k = torch.tensor_split(x_r, self.K_steps, 0)
        m_r_logits_k = torch.tensor_split(m_r_logits, self.K_steps, 0)
        x_r_k = [_[0] for _ in x_r_k]
        m_r_logits_k = [_[0] for _ in m_r_logits_k]

        # Optional: Apply pixelbound
        if self.pixel_bound:
            x_r_k = [torch.sigmoid(item) for item in x_r_k]
        # --- Reconstruct masks ---
        log_m_r_stack = MONet.get_mask_recon_stack(
            m_r_logits_k, 'softmax', log=True)
        log_m_r_k = torch.split(log_m_r_stack, 1, dim=4)
        log_m_r_k = [m[:, :, :, :, 0] for m in log_m_r_k]
        # --- Reconstruct input image by marginalising (aka summing) ---
        x_r_stack = torch.stack(x_r_k, dim=4)
        m_r_stack = torch.stack(log_m_r_k, dim=4).exp()
        recon = (m_r_stack * x_r_stack).sum(dim=4)

        return recon, x_r_k, log_m_r_k


    @staticmethod
    def mask_latent_loss(q_zm_0_k, zm_0_k, zm_k_k=None, ldj_k=None, debug=False):
        num_steps = len(zm_0_k)
        if zm_k_k is None:
            zm_k_k = zm_0_k

        # -- Determine prior --
        p_zm_k = num_steps*[Normal(0, 1)]

        # -- Compute KL using Monte Carlo samples for every step k --
        kl_m_k = []
        for step, p_zm in enumerate(p_zm_k):
            log_q = q_zm_0_k[step].log_prob(zm_0_k[step]).sum(dim=1)
            log_p = p_zm.log_prob(zm_k_k[step]).sum(dim=1)
            kld = log_q - log_p
            if ldj_k is not None:
                ldj = ldj_k[step].sum(dim=1)
                kld = kld - ldj
            kl_m_k.append(kld)

        # -- Sanity check --
        if debug:
            assert len(p_zm_k) == num_steps
            assert len(kl_m_k) == num_steps

        return kl_m_k, p_zm_k


class SRI(nn.Module):
    def __init__(self, K_steps, pixel_bound, feat_dim,  semiconv, kernel,
                 pixel_std, dynamic_K, image_likelihood, debug,
                 img_size, scene_latent_dim, scene_encoder_layers):
        super(SRI, self).__init__()
        # Configuration
        self.K_steps = K_steps
        self.feat_dim = feat_dim
        self.image_likelihood = image_likelihood
        self.debug = debug
        self.img_size = img_size
        self.scene_latent_dim = scene_latent_dim

        self.genesisv2 = GenesisV2(K_steps, img_size, feat_dim, semiconv, kernel,
                                   dynamic_K, debug, pixel_bound, pixel_std, image_likelihood)
        self.scene_encoder = SceneEncoder(feat_dim, scene_latent_dim, scene_encoder_layers)
        self.prior_lstm = AutoregressivePrior(scene_latent_dim, feat_dim)
        self.posterior_lstm = AutoregressivePosterior(scene_latent_dim, feat_dim)
        c = feat_dim

        self.decoder = nn.Sequential(
            B.BroadcastLayer(img_size // 16),
            nn.ConvTranspose2d(feat_dim+2, c, 5, 2, 2, 1),
            nn.GroupNorm(8, c), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, c, 5, 2, 2, 1),
            nn.GroupNorm(8, c), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c, min(c, 64), 5, 2, 2, 1),
            nn.GroupNorm(8, min(c, 64)), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(min(c, 64), min(c, 64), 5, 2, 2, 1),
            nn.GroupNorm(8, min(c, 64)), nn.ReLU(inplace=True),
            nn.Conv2d(min(c, 64), 4, 1))

        # --- Output pixel distribution ---
        self.std = pixel_std


    @staticmethod
    def greedy_matching(q_z_mean, p_z_mean, q_z_sp=None):
        """
        Compute an O(K) greedy matching using O(K^2) space
        between the means of q_z and p_z
        from p-->q. 

        Applies permutation to q_means and 

        Args:
            q_z_mean is [batch_size, K, z_size]
            p_z_mean is [batch_size, K, z_size]
        Returns:
            reordered q means [batch_size, K, z_size]
        """
        q_z_mean_ = q_z_mean.clone()
        batch_size, K, _ = q_z_mean.shape
        batch_idxs_shuffling = torch.arange(0,batch_size,device=q_z_mean.device).unsqueeze(1).repeat(1, K)
        batch_idxs = torch.arange(0,batch_size,device=q_z_mean.device)  # [batch_size]
        with torch.no_grad():
            p_z_mean = p_z_mean.unsqueeze(2).repeat(1, 1, K, 1)  # [batch_size, K, K, z_size]
            q_z_mean = q_z_mean.unsqueeze(1).repeat(1, K, 1, 1)  # [batch_size, K, K, z_size]
            cost = torch.linalg.vector_norm(p_z_mean - q_z_mean, ord=2, dim=-1)  # [batch_size, K, K]
            idxs = []
            
            for step in range(K):
                col_idx = torch.argmin(cost[:,step], dim=1)  # [batch_size]
                cost[batch_idxs,:,col_idx] = np.inf  # set all values in this column to inf
                idxs += [col_idx]
            # Re-order
            idxs = torch.transpose(torch.stack(idxs), 0,1).long()  # [batch_size,K]
            q_z_mean_ = q_z_mean_[batch_idxs_shuffling, idxs]
            if q_z_sp is not None:
                q_z_sp = q_z_sp[batch_idxs_shuffling, idxs]
                return q_z_mean_, q_z_sp
            else:
                return q_z_mean_

    def forward(self, x, use_normalized_nll=False):
        batch_size = x.size(0)

        ######################## SRI Set-to-sequence Inference
        # 1. Segregate
        recon, losses, genv2_stats, comp_stats, att_stats = self.genesisv2(x)
    
        sri_comp_stats = AttrDict(mu_k=[], sigma_k=[], z_scene=[], z_ordered_k=[], kl_l_k=[])
        mu_randomly_ordered_no_grad = torch.stack(comp_stats['mu_k']).detach()
        mu_randomly_ordered_no_grad = mu_randomly_ordered_no_grad.permute(1,0,2).contiguous()
        sigma_randomly_ordered_no_grad = torch.stack(comp_stats['sigma_k']).detach()
        sigma_randomly_ordered_no_grad = sigma_randomly_ordered_no_grad.permute(1,0,2).contiguous()
        z_randomly_ordered_no_grad = torch.stack(comp_stats['z_k']).detach()
        z_randomly_ordered_no_grad = z_randomly_ordered_no_grad.permute(1,0,2).contiguous()

        # 2. Relate
        q_scene = self.scene_encoder(z_randomly_ordered_no_grad)
        
        # 2. Imagination step: slot prior rollout
        scene_latent = q_scene.rsample()
        rolled_out_slots, rollout_prior = self.prior_lstm(scene_latent, self.K_steps)
    
        # 3. Match and Permute
        q_ordered_mu = SRI.greedy_matching(
                        mu_randomly_ordered_no_grad.view(batch_size, self.K_steps, -1),
                        torch.stack([rp.mean for rp in rollout_prior], 1))
        
        # 4. Correlated variances
        q_z_sp_ordered = self.posterior_lstm(scene_latent, q_ordered_mu) 


        q_ordered_mu = torch.tensor_split(q_ordered_mu.view(batch_size, self.K_steps, -1), self.K_steps, 1)
        q_ordered_mu = [_.squeeze(1) for _ in q_ordered_mu]
        q_z_sp_ordered = torch.tensor_split(q_z_sp_ordered.view(batch_size, self.K_steps, -1), self.K_steps, 1)
        q_z_sp_ordered = [_.squeeze(1) for _ in q_z_sp_ordered]
        q_z_ordered = [mvn(mu, sp) for mu,sp in zip(q_ordered_mu, q_z_sp_ordered)]

        ######################## Compute L_SRI
                
        # L_rolloutKL
        rollout_kl = 0
        for q,p in zip(q_z_ordered, rollout_prior):
            rollout_kl += torch.distributions.kl.kl_divergence(q, p)
        
        # Store samples and decode
        sri_comp_stats['mu_k'] = q_ordered_mu
        sri_comp_stats['z_imagination'] = rolled_out_slots
        sri_comp_stats['z_ordered_k'] = [q_z.rsample() for q_z in q_z_ordered]

        scene_prior = std_mvn(shape=[batch_size, self.scene_latent_dim], device=x.device)
        # compute scene distribution kl
        scene_kl = torch.distributions.kl.kl_divergence(q_scene, scene_prior)  # [batch_size]
        slot_kl = self.prior_lstm.KL_loss(scene_latent, sri_comp_stats.z_ordered_k, q_z_ordered)
                    
        # --- Decode latents ---
        recon_ordered, x_r_k_ordered, log_m_r_k_ordered = \
            self.genesisv2.decode_latents(sri_comp_stats.z_ordered_k,
                                the_decoder=self.decoder)
        mx_r_k_ordered = [x*logm.exp() for x, logm in zip(x_r_k_ordered, log_m_r_k_ordered)]
    
        # -- Reconstruction loss
        if self.image_likelihood == 'MoG':
            sri_nll = SRI.gmm_negative_loglikelihood(x,
                                log_m_r_k_ordered, x_r_k_ordered, self.std, use_normalized_nll)
        elif self.image_likelihood == 'MoG-Normed':
            sri_nll = SRI.gmm_negative_loglikelihood(x,
                                log_m_r_k_ordered, x_r_k_ordered, self.std, use_normalized_nll=True)

        losses['sri_err'] = sri_nll
        losses['sri_kl_scene'] = scene_kl
        losses['sri_kl_z_ordered'] = rollout_kl
        losses['sri_kl_slot'] = slot_kl

        sri_stats = AttrDict(
            recon=recon_ordered, x_r_k=x_r_k_ordered,
            log_m_r_k=log_m_r_k_ordered, mx_r_k=mx_r_k_ordered,
            instance_seg_r=torch.argmax(torch.cat(log_m_r_k_ordered, dim=1), dim=1))
    
        return (recon, recon_ordered), losses, (genv2_stats, sri_stats), att_stats, (comp_stats, sri_comp_stats)


    @staticmethod
    def gmm_negative_loglikelihood(x, log_m_k, x_r_k, std, use_normalized_nll=False):
        # NLL [batch_size, 1, H, W]
        x_loc = torch.stack(x_r_k, dim=4)
        sq_err = (x.unsqueeze(4) - x_loc).pow(2)
        # log N(x; x_loc, log_var): [N, K, C, H, W]
        if not use_normalized_nll:
            log_prob_x = -0.5 * (2. * math.log(std)) - 0.5 * (sq_err / std ** 2)
        else:
            log_prob_x = -0.5 * (2 * math.log(std) + math.log(2 * math.pi)) - 0.5 * (sq_err / (std ** 2))
        # [N, C, H, W, K]
        mask_logprobs = torch.stack(log_m_k, dim=4)
        log_p_k = (mask_logprobs + log_prob_x)
        # logsumexp over slots [N, C, H, W]
        log_p = torch.logsumexp(log_p_k, dim=4)
        # [N]
        nll = -torch.sum(log_p, dim=[1,2,3])
        return nll


    def sample(self, batch_size, temp=1.0, device='cuda', K_steps=None, scene_latent=None):
        K_steps = self.K_steps if K_steps is None else K_steps
        
        if scene_latent is None:
            scene_prior = std_mvn(shape=[batch_size, self.scene_latent_dim], device=device, temperature=temp)
            s = scene_prior.sample()
        else:
            s = scene_latent
        # slots
        z_k, _ = self.prior_lstm(s, K_steps, t=temp)
      
        # Decode latents
        recon, x_r_k, log_m_r_k = self.genesisv2.decode_latents(z_k, the_decoder=self.decoder)

        stats = AttrDict(x_k=x_r_k, log_m_k=log_m_r_k,
                         mx_k=[x*m.exp() for x, m in zip(x_r_k, log_m_r_k)],
                         z_k=z_k)
        return recon, stats