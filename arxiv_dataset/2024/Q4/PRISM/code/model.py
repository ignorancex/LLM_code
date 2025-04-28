### ====== PRISM MODEL ======

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from networks import AnatomyUNet, StyleEncoder, Patchifier
from loss import KLDivergenceLoss, PatchNCELoss, PerceptualLoss, BilateralLoss, TotalVariationLoss
from dataset import PRISM_MRI_Dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm

class PRISM:
    def __init__(self, intensity_levels, latent_dim, num_sites=3, gpu_id=0, modality='T2', modalities = ['T2', 'PD']):
        self.n_sites = num_sites
        # mod_dict = {'T1': 0, 'T2': 1, 'PD': 2}
        mod_dict = {'T2': 0, 'PD': 1}
        self.modality = mod_dict[modality]
        # self.modalities = mod_dict.values()
        self.other_modalities = [mod_dict[mod] for mod in modalities if mod != modality]
        self.modalities = [self.modality] + self.other_modalities
        self.intensity_levels = intensity_levels
        self.latent_dim = latent_dim
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

        self.train_loader, self.valid_loader = None, None
        self.out_dir = None
        self.optimizer = None
        self.scheduler = None

        self.l1_loss, self.kld_loss, self.contrastive_loss, self.perceptual_loss, self.bilateral_loss, self.tv_loss = None, None, None, None, None, None

        # define networks
        self.anatomy_encoder = AnatomyUNet(in_ch=1, out_ch=self.intensity_levels, base_ch=8, final_act='none')
        self.style_encoder = StyleEncoder(in_ch=1, out_ch=self.latent_dim)
        self.decoder = AnatomyUNet(in_ch=1 + self.latent_dim, out_ch=1, base_ch=16, final_act='relu')
        self.patchifier = Patchifier(in_ch=1, out_ch=128)

        self.anatomy_encoder.to(self.device)
        self.style_encoder.to(self.device)
        self.decoder.to(self.device)
        self.patchifier.to(self.device)
        self.start_epoch = 0


    def init_training(self, out_dir, lr, vgg_path='/kaggle/input/vgg16_imagenet/pytorch/default/1/vgg16_imagenet.pth'):
        # define loss functions
        self.l1_loss = nn.L1Loss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.kld_loss = KLDivergenceLoss()

        # # Initialize the VGG-16 model without weights
        vgg = models.vgg16(weights=None)
        # Load the saved state dictionary
        vgg.load_state_dict(torch.load(vgg_path))
        # Use the .features and move to the desired device
        vgg = vgg.features.to(self.device)

        # If vgg model not available, use the following line to download the model
#         vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(self.device)
        self.perceptual_loss = PerceptualLoss(vgg)
        self.contrastive_loss = PatchNCELoss()
        self.bilateral_loss = BilateralLoss(spatial_sigma=3.0, intensity_sigma=0.1)
        self.tv_loss = TotalVariationLoss()

        # define optimizer and learning rate scheduler
        self.optimizer = Adam(list(self.anatomy_encoder.parameters()) +
                              list(self.style_encoder.parameters()) +
                              list(self.decoder.parameters()) +
                              list(self.patchifier.parameters()), lr=lr)
        self.scheduler = CyclicLR(self.optimizer, base_lr=4e-4, max_lr=7e-4, cycle_momentum=False)
        self.start_epoch = self.start_epoch + 1

        self.out_dir = out_dir


    def load_dataset_from_pt(self, batch_size, train_path='/kaggle/input/ixi-guys-train/ixi-guys-ds.pth', test_path='/kaggle/input/ixi-guys-test/ixi-guys-test.pth'):
        train_dataset = torch.load(train_path)
        test_dataset = torch.load(test_path)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    def get_style_code(self, src_imgs):
        if isinstance(src_imgs, list):
            style_codes, mus, logvars = [], [], []
            for modality_stack in src_imgs:
                style_codes.append([])
                mus.append([])
                logvars.append([])
                for image in modality_stack:
                    mu, logvar = self.style_encoder(image)
                    style_code = torch.randn(mu.size()).to(self.device) * torch.sqrt(torch.exp(logvar)) + mu
                    style_codes[-1].append(style_code)
                    mus[-1].append(mu)
                    logvars[-1].append(logvar)
            return style_codes, mus, logvars
       
        else:
            mu, logvar = self.style_encoder(src_imgs)
            style_code = torch.randn(mu.size()).to(self.device) * torch.sqrt(torch.exp(logvar)) + mu
            return style_code, mu, logvar

    
    def get_anatomy_representations(self, src_imgs, mask):
        if isinstance(src_imgs, list):
            logits, anatomies = [], []
            for modality_stack in src_imgs:
                logits.append([])
                anatomies.append([])
                for image in modality_stack:
                    logit = self.anatomy_encoder(image)
                    anatomy = self.channel_aggregation(F.gumbel_softmax(logit, tau=1.0, dim=1, hard=True)) * mask
                    logits[-1].append(logit)
                    anatomies[-1].append(anatomy)
            return logits, anatomies
        
        else:
            logit = self.anatomy_encoder(src_imgs)
            anatomy = self.channel_aggregation(F.gumbel_softmax(logit, tau=1.0, dim=1, hard=True)) * mask
            return logit, anatomy

    
    def get_src_images(self, subject):
        images = []
        for modality in self.modalities:
            if subject[modality]['exists'][0]:
                images.append([])
                image = subject[modality]['image'].to(self.device)
                images[modality].append(image)
                # for aug in subject[modality]['aug']:
                #     images.append(aug.to(self.device))
                images[modality].append(subject[modality]['aug'][0].to(self.device))
                images[modality].append(subject[modality]['aug'][1].to(self.device))
            else:
                raise ValueError(f"Modality {modality} does not exist for subject {subject[modality]['subject_id'][0]}")

        return images
    

    def channel_aggregation(self, onehot_encoded_anatomy):
        """
        Combine multi-channel one-hot encoded anatomy representations into one channel (label-encoding).

        ===INPUTS===
        * onehot_encoded_anatomy: torch.Tensor (batch_size, self.intensity_levels, image_dim, image_dim)
            One-hot encoded anatomy variable. At each pixel location, only one channel will take value of 1,
            and other channels will be 0.
        ===OUTPUTS===
        * label_encode_anatomy: torch.Tensor (batch_size, 1, image_dim, image_dim)
            The intensity value of each pixel will be determined by the channel index with value of 1.
        """
        batch_size = onehot_encoded_anatomy.shape[0]
        image_dim = onehot_encoded_anatomy.shape[3]
        value_tensor = (torch.arange(0, self.intensity_levels) * 1.0).to(self.device)
        value_tensor = value_tensor.view(1, self.intensity_levels, 1, 1).repeat(batch_size, 1, image_dim, image_dim)
        label_encode_anatomy = onehot_encoded_anatomy * value_tensor.detach()
        return label_encode_anatomy.sum(1, keepdim=True) / self.intensity_levels
    
    
    def decode(self, anatomy, style_code, mask):
        image_dim = mask.size(-1)
        combined_map = torch.cat([anatomy, style_code.repeat(1, 1, image_dim, image_dim)], dim=1)
        rec_image = self.decoder(combined_map) * mask
        return rec_image

    
    def calculate_features_for_contrastive_loss(self, anatomies, source_images):
        '''
        Inputs:
        - source_images: nested list corresponding to a patient where each sublist corresponds to an Mri modality and contains tensors of the mri slice and its augmentations. eg. [[t1_original_batch, t1_gamma1_batch, t1_gamma2_batch, ...], [t2_original_batch, t2_gamma1_batch, t2_gamma2_batch, ...], [pd_original_batch, pd_gamma1_batch, pd_gamma2_batch, ...]]
        - anatomies: nested list corresponding to a patient where each sublist corresponds to an Mri modality and contains tensors of the anatomy representations of the mri slice and its augmentations. eg. [[t1_original_batch, t1_gamma1_batch, t1_gamma2_batch, ...], [t2_original_batch, t2_gamma1_batch, t2_gamma2_batch, ...], [pd_original_batch, pd_gamma1_batch, pd_gamma2_batch, ...]]
        
        Description:
        - This function calculates the features for the contrastive loss function.
        - query_feature: feature patch extracted by patchifier from the query anatomy: anatomies[self.modality][0]
        - positive_features: feature patches extracted by patchifier from same location as query patch, from positive anatomies: anatomies[<other modalities>][<all augmentations>]
        - negative_features: 
            - feature patches extracted by patchifier from same location as query patch, from source images: source_images[<all modalities>][<all augmentations>]
            - feature patches extracted by patchifier from other random locations (wrt query patch) from: anatomies[<all modalities>][<all augmentations>]
            - feature patches extracted by patchifier from random locations of anatomies of other batch samples (ie. shuffled across batch dimension)

        Output:
        - query_feature: torch.Tensor (batch_size, 128, num_patches)
        - positive_features: torch.Tensor (batch_size, 128, num_patches)
        - negative_features: torch.Tensor (batch_size, 128, num_patches)
        '''

        batch_size = anatomies[0][0].shape[0]

        # Query patch is selected from anatomies of the self.modality (T2 in this case)
        query_anatomy = anatomies[self.modality][0]  # Only original T2 image (index 0) is the query
        query_feature = self.patchifier(query_anatomy).view(batch_size, 128, -1)

        # Positive patches from all augmentations of other modalities (T1 and PD in this case)
        # Total 3*m-1 (TO DO: add augs from query mod)
        positive_features = torch.cat(
            [self.patchifier(anatomy).view(batch_size, 128, -1) 
#             for modality in self.other_modalities 
            # for anatomy in anatomies[modality]], dim=-1) # All augmentations are considered as positive
            for anatomy in anatomies[self.modality][1:3]], dim=-1) # Only original, gamma1 and gamma2 images are considered as positive
                
        num_positive_patches = positive_features.shape[-1]
#         print(f"num_positive_patches: {num_positive_patches}")

        # Negative features:
        # 1. Extract patches from source images of all modalities (including all augmentations)
        negative_from_source = torch.cat(
            [self.patchifier(image).view(batch_size, 128, -1) 
#             for modality in range(len(source_images))
            # for image in source_images[modality]], dim=-1) # All patches are considered as negative
            for image in source_images[self.modality][:3]], dim=-1) # Only original, gamma1 and gamma2 patches are considered as negative
        
        num_src_neg_patches = negative_from_source.shape[-1]
#         print(f"num_src_neg_patches: {num_src_neg_patches}")
        
        # 2. Extract patches from random locations in anatomies (all modalities and augmentations)
        negative_random_patches = torch.cat(
            [self.patchifier(anatomy).view(batch_size, 128, -1)#[:, :, torch.randperm(num_negative_patches)] 
#             for modality in range(len(anatomies)) 
            # for anatomy in anatomies[modality]], dim=-1) # All patches are considered as negative
            for anatomy in anatomies[self.modality][:3]], dim=-1) # Only original, gamma1 and gamma2 patches are considered as negative
        
        num_anatomy_neg_patches = negative_random_patches.shape[-1]
#         print(f"num_anatomy_patches: {num_anatomy_neg_patches}")
        
        negative_random_patches = negative_random_patches[:, :, torch.randperm(num_anatomy_neg_patches)]

        # 3. Extract patches from shuffled patches of other batch samples
        negative_shuffled = torch.cat(
            [self.patchifier(anatomy).view(batch_size, 128, -1)[torch.randperm(batch_size), :, :] 
#             for modality in range(len(anatomies)) 
            # for anatomy in anatomies[modality]], dim=-1) # All patches are considered as negative
            for anatomy in anatomies[self.modality][:3]], dim=-1) # Only original, gamma1 and gamma2 patches are considered as negative

        # Combine all negative features
        negative_features = torch.cat([negative_from_source, negative_random_patches, negative_shuffled], dim=-1)
        
#         print(f'Query feature shape: {query_feature.shape}')
#         print(f'Positive features shape: {positive_features.shape}')
#         print(f'Negative features shape: {negative_features.shape}')

        return query_feature, positive_features, negative_features
    

    def calculate_loss(self, rec_image, ref_image, mask, mu, logvar, anatomies, source_images):

        # 1. reconstruction loss
        rec_loss = self.l1_loss(rec_image[mask], ref_image[mask]).mean() + 2*self.mse_loss(rec_image[mask], ref_image[mask]).mean()
        perceptual_loss = self.perceptual_loss(rec_image, ref_image).mean()

        # 2. KLD loss
        kld_loss = self.kld_loss(mu, logvar).mean()

        # 3. anatomical contrastive loss
        query_feature, \
            positive_feature, \
            negative_feature = self.calculate_features_for_contrastive_loss(anatomies, source_images)
        anatomy_contrastive_loss = self.contrastive_loss(query_feature, positive_feature.detach(), negative_feature.detach())
        
        # COMBINE LOSSES
        total_loss = 10 * rec_loss + 5e-1 * perceptual_loss + 1e-5 * kld_loss + anatomy_contrastive_loss
#         self.optimizer.zero_grad()
#         total_loss.backward()
#         self.optimizer.step()
#         self.scheduler.step()
        loss_dict = {'rec_loss': rec_loss.item(),
                'percep_loss': perceptual_loss.item(),
                'kld_loss': kld_loss.item(),
                'anatomy_contrastive': anatomy_contrastive_loss.item(),
                'total_loss': total_loss.item()}
        return total_loss, loss_dict


    def calculate_cycle_consistency_loss(self, style_rec, style_src, anatomy_rec, anatomy_src):
        style_cyc_loss = self.l1_loss(style_rec, style_src).mean()
        anatomy_cyc_loss = self.l1_loss(anatomy_rec, anatomy_src).mean()

        cycle_loss = style_cyc_loss + 5e-2 * anatomy_cyc_loss
#         self.optimizer.zero_grad()
#         (5e-2 * cycle_loss).backward()
#         self.optimizer.step()
#         self.scheduler.step()
        loss_dict = {'style_cyc': style_cyc_loss.item(),
                'anatomy_cyc': anatomy_cyc_loss.item()}
        return 5e-2 * cycle_loss, loss_dict
        
    def train_batch(self, batch, epoch, batch_id):
        source_images = self.get_src_images(batch) # nested list of source images + augmentations, for each modality eg. [[mod1],[mod2],[mod2]]
        source_image = source_images[self.modality][0] # original image of the concerned modality
#         print(f"source_images len: {len(source_images)}") # len: 2 (T2, PD)
#         print(f"source image shape: {source_image.shape}") # shape: torch.Size([8, 1, 256, 256])
        mask = batch[self.modality]['mask'].to(self.device)     # potential for error?
#         print(f"mask shape: {mask.shape}")  # shape: torch.Size([8, 1, 256, 256])
        _, anatomy_representations = self.get_anatomy_representations(source_images, mask) # nested list of anatomies of images + augmentations, for each modality eg. [[mod1],[mod2],[mod2]]
        src_anatomy = anatomy_representations[self.modality][0] # original anatomy of the concerned modality
        src_anatomy_clone = src_anatomy.clone()
        style_code, mu, logvar = self.get_style_code(source_image) # nested list ...    # PFE?
        style_code_clone = style_code.clone()

        rec_image = self.decode(src_anatomy, style_code, mask)
        rec_image_clone = rec_image.clone()
#         print(f"rec_img shape: {rec_image.shape}")

        loss, loss_dict = self.calculate_loss(rec_image, source_image, mask, mu, logvar,
                                   anatomy_representations, source_images)
        
        style_recon, _ = self.style_encoder(rec_image_clone)
        _, anatomy_recon =  self.get_anatomy_representations(rec_image_clone, mask)
#         print(f"anatomy_recon shape: {anatomy_recon.shape}")
        
        # 4. cycle loss
        cycle_loss, cyc_loss_dict = self.calculate_cycle_consistency_loss(style_recon, style_code_clone.detach(), 
                                                           anatomy_recon, src_anatomy_clone.detach())
        
#         bl_loss = self.bilateral_loss(src_anatomy)
        # tv_loss = self.tv_loss(src_anatomy)
        # denoise_loss = 0.05 * tv_loss # + 0.03 * bl_loss
        total_loss = loss + cycle_loss# + denoise_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.train_loader.set_description((f'epoch: {epoch}; '
                                           f'rec: {loss_dict["rec_loss"]:.3f}; '
                                           f'percep: {loss_dict["percep_loss"]:.3f}; '
                                           f'kld: {loss_dict["kld_loss"]:.3f}; '
                                           f'anatomy_contrastive: {loss_dict["anatomy_contrastive"]:.3f}; '
#                                            f'bl: {bl_loss:.3f}'
                                           # f'tv: {tv_loss:.3f}'
                                           f'style_cyc: {cyc_loss_dict["style_cyc"]:.3f}; '
                                           f'anatomy_cyc: {cyc_loss_dict["anatomy_cyc"]:.3f}; '))

        

    def train(self, epochs):
        for epoch in range(self.start_epoch, epochs+1):
            # ====== TRAINING ======
            self.train_loader = tqdm(self.train_loader)
            self.style_encoder.train()
            self.anatomy_encoder.train()
            self.decoder.train()
            self.patchifier.train()
            for batch_id, image_dicts in enumerate(self.train_loader):
                self.train_batch(image_dicts, epoch, batch_id)
   

    def save_model(self, epoch):
        torch.save(self.anatomy_encoder.state_dict(), f'{self.out_dir}/anatomy_encoder_epoch{epoch}.pth')
        torch.save(self.style_encoder.state_dict(), f'{self.out_dir}/style_encoder_epoch{epoch}.pth')
        torch.save(self.decoder.state_dict(), f'{self.out_dir}/decoder_epoch{epoch}.pth')


    def load_model(self, epoch):
        self.anatomy_encoder.load_state_dict(torch.load(f'{self.out_dir}/anatomy_encoder_epoch{epoch}.pth'))
        self.style_encoder.load_state_dict(torch.load(f'{self.out_dir}/style_encoder_epoch{epoch}.pth'))
        self.decoder.load_state_dict(torch.load(f'{self.out_dir}/decoder_epoch{epoch}.pth'))