import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from utils.utils_eval import _test_step, get_eval_dictionary, _test_end,get_eval_dictionary_classification
import hydra

import torch.optim as optim
import torchio as tio
from utils.utils_eval import  redFlagEvaluation_einscanner, calc_thresh, calc_thresh_classification,evaluateModel



import torch
from torch import nn
from .resnet import ResNetClassifer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_sizes, frames, num_classes, dim, depth, heads, mlp_dim, pool = 'mean', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,conv_model= None,enable_maxpool= False,conv_dimension= 64):
        super().__init__()


        image_height, image_width = pair(image_size)
        self.image_patch_sizes = [x for x in image_patch_sizes if x > 0] 
        
        self.total_num_volume_patches_per_context =  [] 
        self.total_voxels_per_context =  [] 
        for block_size in self.image_patch_sizes:

            assert image_height % block_size == 0 and image_width % block_size == 0 and frames % block_size == 0 #Image dimensions must be divisible by the block size
            
            #Stores the number of patches per block size. Block sizes 8x8x8,16x16x16,32x32,32x64x64x64
            total_num_volume_patches = (image_height // block_size) * (image_width // block_size) * (frames // block_size)
            self.total_num_volume_patches_per_context.append(total_num_volume_patches)
            
            #Stores total voxels(dimensions) per block size. Block sizes 8x8x8,16x16x16,32x32,32x64x64x64
            self.total_voxels_per_context.append(channels * block_size * block_size * block_size)

        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = []

        self.avgpool =  torch.nn.AdaptiveAvgPool3d(1)
        self.enable_maxpool = enable_maxpool
        if self.enable_maxpool:
            self.maxpool  =  torch.nn.AdaptiveMaxPool3d(1)
        self.factor_pool = 2 if self.enable_maxpool else 1 #the dimension of the patch dimension increases 2x if max pool features are also used 

        dim = conv_dimension * self.factor_pool
        for patch_dim in self.total_voxels_per_context:
            self.to_patch_embedding.append( nn.Sequential(
                #Rearrange('b c (f h w) p1 p2 pf -> b (f h w) (p1 p2 pf c)', p1 = patch_dim, p2 = patch_dim, pf = patch_dim),
                nn.Linear(dim, dim),
            ).cuda())

        self.rearrange_patches = []

        for block_size in self.image_patch_sizes:
        
            self.rearrange_patches.append( nn.Sequential(
                Rearrange('b c (f pf) (h p1) (w p2) -> b c (f h w) pf p1 p2', p1 = block_size, p2 = block_size, pf = block_size),
            ))
        
        
        self.conv_model = conv_model.eval()
        self.pos_patch_embeddings =  [] 

        embedding_dim = dim #* len(self.image_patch_sizes)
        for num_patches in self.total_num_volume_patches_per_context: 

            pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim)).cuda()
            self.pos_patch_embeddings.append(pos_embedding)

        total_separator_tokens  = 2 * len(self.total_num_volume_patches_per_context) if len(self.total_num_volume_patches_per_context) > 1 else 0
        self.pos_embedding = nn.Parameter(torch.randn(1, sum(self.total_num_volume_patches_per_context) + total_separator_tokens + 1, dim)).cuda()
        

        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.sep_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(embedding_dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, video):
        
        embeddings = None
        #print("Input",video.shape)
        for i,rearrange in enumerate(self.rearrange_patches):
            context_embedding = None

            #B C D W H -> B C (W*H*D/block_size) block_size block_size
            x = rearrange(video)
            #print("Rearrange 1", x.shape)
            #print(self.total_num_volume_patches_per_context,"ttal patches")


            #Performing 3D Convolution on multi-context level
            

            for index in range(x.shape[2]): #iterating through each block 

                
                _,_,x_conv =  self.conv_model.model(x[:,:,index,:,:]) #Convolving each block
                x_conv_pool = self.avgpool(x_conv) #Average pooling to reduce the extrated features to a single scalar (B C 64)
                #print("Rearrange 2", x_conv_pool.shape,"Rearrange 1", x.shape)
                if self.enable_maxpool:
                    x_conv_max = self.maxpool(x_conv)
                    x_conv_pool = torch.cat((x_conv_pool,x_conv_max),dim=1)

                    #print("Rearrange 3", x_conv_pool.shape)
                if context_embedding is None: 
                    
                    
                    context_embedding = torch.squeeze(x_conv_pool)
                    if len(context_embedding.shape) == 1: 

                        context_embedding = torch.unsqueeze(context_embedding, dim=0)
                        context_embedding = torch.unsqueeze(context_embedding, dim=0)
                    else: 
                        context_embedding = torch.unsqueeze(context_embedding, dim=1)
                    
                   

                    #print("Rearrange 4", context_embedding.shape)
                
                else: 

                    x_conv_pool = torch.squeeze(x_conv_pool)

                    if len(x_conv_pool.shape) == 1: 
                        x_conv_pool = torch.unsqueeze(x_conv_pool, dim=0)
                        x_conv_pool = torch.unsqueeze(x_conv_pool, dim=0)
                    else: 
                        x_conv_pool = torch.unsqueeze(x_conv_pool, dim=1)
                    

                    context_embedding = torch.cat((context_embedding,x_conv_pool), dim=1)
                    #print("Rearrange 5", context_embedding.shape)
            
            if embeddings is None:
                
                if len(context_embedding.shape) == 2:
                    context_embedding = torch.unsqueeze(context_embedding, dim=0)

                embeddings = context_embedding

                #MLP 
                embeddings = self.to_patch_embedding[i](embeddings)

                #Learnable positonal embedding 
                pos_embedding = self.pos_patch_embeddings[i][:, :(self.total_num_volume_patches_per_context[i])]
                #Add learnable positional embedding  
                embeddings += pos_embedding

            else: 

                #print(embeddings.shape,context_embedding.shape,"Contexgt 2")
                #MLP
                context_embedding = self.to_patch_embedding[i](context_embedding)
                
                #Learnable positonal embedding
                pos_embedding = self.pos_patch_embeddings[i][:, :(self.total_num_volume_patches_per_context[i])]
                
                #Add learnable positional embedding
                context_embedding += pos_embedding

                b, _, _ = context_embedding.shape
                sep_tokens = repeat(self.sep_token, '1 1 d -> b 1 d', b = b)
                print("separate token",sep_tokens.shape)
                #concatanating multi context embeddings
                embeddings = torch.cat((embeddings,sep_tokens,context_embedding,sep_tokens), dim=1)


        #print("Final Embedding",embeddings.shape)
            
        #print(embeddings.shape)
        b, n, _ = embeddings.shape
        print("Embedding shape",embeddings.shape,"positional embedding shape",self.pos_embedding.shape)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        
        x = torch.cat((cls_tokens, embeddings), dim=1)
        print("class + embedding shape",x.shape)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class ViTClassifier(LightningModule):

    def __init__(self,cfg,prefix=None):
        super(ViTClassifier, self).__init__()
        
        self.cfg = cfg
        self.conv_model: LightningModule = None
        #Load conv model
        #self.conv_model.load_state_dict(torch.load(self.cfg.convnet_model_ckpt_path)['state_dict'])
        # Model 
        
                
        self.prefix =prefix
        self.softmax = torch.nn.Softmax(dim=1)
        # Loss function
        cls_weights = torch.tensor(list(self.cfg.class_weights)).float()
        self.criterion = torch.nn.CrossEntropyLoss(weight=cls_weights).cuda()
        
        
    
    def create_ViTmodel(self): 
        
        assert self.conv_model is not None 

        self.model = ViT(
            image_size = self.cfg['image_size'],          # image size
            frames = self.cfg['image_size'],               # number of frames
            image_patch_sizes = [self.cfg['image_patch_size_1'],self.cfg['image_patch_size_2'],self.cfg['image_patch_size_3'],self.cfg['image_patch_size_4']],     # image patch size
            num_classes = self.cfg['n_classes'],
            dim = self.cfg['dim'],
            depth = self.cfg['depth'],
            heads = self.cfg['heads'],
            mlp_dim = self.cfg['mlp_dim'],
            dropout = self.cfg['dropout'],
            channels = 1,
            emb_dropout = self.cfg['emb_dropout'],
            conv_model = self.conv_model,
            enable_maxpool = self.cfg['enable_maxpool'],
            conv_dimension = self.cfg['conv_dimension']
        )

    def configure_optimizers(self):
        
        optimizer =  optim.Adam(self.model.parameters() ,lr=self.cfg.lr, amsgrad=False)
        scheduler =  optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,verbose=True,patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler,"monitor": "val/loss"} #[optimizer],[scheduler]

    def prepare_batch(self, batch):
        image   =  batch['image'][tio.DATA] 
        #image   =  torch.unsqueeze(image,dim=1)


        return image,batch['label'],batch['patient_disease_id'],batch['image_path']
    
    def training_step(self, batch, batch_idx):

        inputs,y,patient_disease_id,_ = self.prepare_batch(batch)
        
        outputs = self.model(inputs)
        #print(outputs["image"].shape,inputs.shape)
        # calculate loss
        #print(y, y.shape, outputs.shape)

        loss = self.criterion(outputs,y)
        self.log(f'train/loss',loss.item(), prog_bar=False, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)
        return {"loss": loss}

    def on_validation_epoch_start(self): 
        self.val_eval_dict = get_eval_dictionary_classification()

    
    def validation_step(self, batch, batch_idx):
        #self.counter += 1
        inputs,y,patient_disease_id,_ = self.prepare_batch(batch)
        outputs = self.model(inputs)
        #print(outputs["image"].shape,inputs.shape)
        # calculate loss
        
        loss = self.criterion(outputs,y)
        target = self.softmax(outputs)
        
        AnomalyScoreReco_vol = target[:,1]

        self.val_eval_dict['labelPerVol'] = torch.cat((torch.tensor(self.val_eval_dict['labelPerVol']), y.cpu()), 0)
        self.val_eval_dict['AnomalyScorePerVol'] = torch.cat((torch.tensor(self.val_eval_dict['AnomalyScorePerVol']), AnomalyScoreReco_vol.cpu()), 0)
        self.val_eval_dict['patient_disease_id'] = self.val_eval_dict['patient_disease_id'] + patient_disease_id
        self.log('val/loss',loss.item(), prog_bar=False, on_step=False, on_epoch=True, batch_size=inputs.shape[0],sync_dist=True)
        return {"loss": loss}

    def on_validation_epoch_end(self):

        #Calculate threshold 
        thresh = calc_thresh_classification(self.val_eval_dict) 
        eval_dict = evaluateModel(self.val_eval_dict.copy(), thresh=thresh)
        #print(eval_dict)
        print("F1",eval_dict['F1_thresh_1p_prc'] )

        self.log(f'val/F1',eval_dict['F1_thresh_1p_prc'], prog_bar=False, on_step=False, on_epoch=True)



    def on_test_start(self):
        self.eval_dict = get_eval_dictionary_classification()
        

    def test_step(self, batch, batch_idx: int):

        inputs,y,patient_disease_id,image_path = self.prepare_batch(batch)
        outputs = self.model(inputs)
        
        loss = self.criterion(outputs,y)
        target = self.softmax(outputs)
        
        #print(target[:,1],target)
        if target.shape[0] > 1: 
            
            AnomalyScoreReco_vol_mean = torch.mean(target[:,1]).item()
            AnomalyScoreReco_vol_std = torch.std(target[:,1]).item()

            

            AnomalyScoreReco_vol     = AnomalyScoreReco_vol_mean
            AnomalyScoreReco_vol_std = AnomalyScoreReco_vol_std 

        else: 

            AnomalyScoreReco_vol     = target[:,1].item()
            AnomalyScoreReco_vol_std = 0.0


        #AnomalyScoreReco_volL2 = 0
        
        
        self.eval_dict['AnomalyScorePerVol'].append(AnomalyScoreReco_vol)
        self.eval_dict['AnomalyScorePerVol_std'].append(AnomalyScoreReco_vol_std)


      
        
        # calculate metrics
        _test_step(self, None,None,inputs,y[0],patient_disease_id[0],batch_idx) # everything that is independent of the model choice

        
        