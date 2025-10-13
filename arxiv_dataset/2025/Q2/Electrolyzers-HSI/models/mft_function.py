"""
MFT Model Inference for Electrolyzers-HSI Dataset
(Adapted from the original MFT implementation)

This module implements the Multi-modal Feature Transformer (MFT) architecture and the utilities
necessary for the inference pipeline on the Electrolyzers-HSI dataset, combining, both,
hyperspectral (HSI) and RGB imagery.

MFT fuses spectral and spatial modalities using a combination of:
- 3D convolutions on HSI inputs
- 2D convolutions on RGB inputs
- HetConv blocks for efficient mixed convolution
- Multi-head cross-attention transformers for modality interaction
- Position embeddings and learned tokenization weights for fusion

Contents:

1. Model Architecture:
   - `MFT`: Main model class for fusing HSI and RGB modalities.
   - `HetConv`: Heterogeneous convolution layer (groupwise + pointwise).
   - `MCrossAttention`: Cross-attention layer between fused features.
   - `TransformerEncoder`, `Block`, `Mlp`: Transformer subcomponents used in MFT.

2. Inference Utilities:
   - `reports`: Generates classification metrics, including confusion matrix, overall accuracy (OA),
     average accuracy (AA), class-wise accuracy, and Cohen's Kappa.
   - `AA_andEachClassAccuracy`: Computes per-class and average accuracies from a confusion matrix.
   - `set_seed`: Ensures reproducibility by seeding torch and numpy.
   - `create_hsi_rgb_patches_with_labels`: Prepares spatially aligned HSI and RGB patches with labels 
     from a ground truth mask, excluding ignored classes.

3. Execution Details:
   - Supports efficient inference using GPU with torch.
   - Processes large test sets in batches (`testSizeNumber`).
   - The model expects input data to be patch-wise and band-augmented prior to inference.
   - Only label names relevant to the Electrolyzers-HSI dataset are currently active.

Requirements:
- PyTorch
- NumPy
- SciPy
- scikit-learn

Usage Example:
    1. Instantiate the `MFT` model with desired parameters.
    2. Load pretrained weights.
    3. Prepare patch-wise test inputs using `create_hsi_rgb_patches_with_labels`.
    4. Use `reports` to evaluate model performance.

Author: HiF-Explo — Elias Arbash
"""


# mft model functions and classes
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from operator import truediv
from scipy.io import loadmat as loadmat
import torch.nn as nn
import torch
import numpy as np
cudnn.deterministic = True
cudnn.benchmark = False
from torch.nn import LayerNorm,Linear,Dropout
import copy


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding = None, bias = None,p = 64, g = 64):
        super(HetConv, self).__init__()
        # Groupwise Convolution
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,groups=g,padding = kernel_size//3, stride = stride)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1,groups=p, stride = stride)
    def forward(self, x):
        return self.gwc(x) + self.pwc(x)

class MCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(head_dim, dim , bias=qkv_bias)
        self.wk = nn.Linear(head_dim, dim , bias=qkv_bias)
        self.wv = nn.Linear(head_dim, dim , bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:1, ...].reshape(B, 1, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)  # B1C -> B1H(C/H) -> BH1(C/H)
        k = self.wk(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        v = self.wv(x.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3)  # BNC -> BNH(C/H) -> BHN(C/H)
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
        x = torch.einsum('bhij,bhjd->bhid', attn, v).transpose(1, 2)
#         x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, 1, C * self.num_heads)   # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, dim):
        super(Mlp, self).__init__()
        self.fc1 = Linear(dim, 512)
        self.fc2 = Linear(512, dim)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
class Block(nn.Module):
    def __init__(self, dim):
        super(Block, self).__init__()
        self.hidden_size = dim
        self.attention_norm = LayerNorm(dim, eps=1e-6)
        self.ffn_norm = LayerNorm(dim, eps=1e-6)
        self.ffn = Mlp(dim)
#         self.attn = Attention(dim = 64)
        self.attn = MCrossAttention(dim = dim)
    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x= self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x

class TransformerEncoder(nn.Module):

    def __init__(self, dim, num_heads= 8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.1, attn_drop=0.1,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(dim, eps=1e-6)
        for _ in range(2):
            layer = Block(dim)
            self.layer.append(copy.deepcopy(layer))


    def forward(self, x):
        for layer_block in self.layer:
            x= layer_block(x)

        encoded = self.encoder_norm(x)



        return encoded[:,0]


class MFT(nn.Module):
    def __init__(self, FM, NC, NCLidar, Classes, HSIOnly):
        super(MFT, self).__init__()
        self.HSIOnly = HSIOnly
        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 8, (9, 3, 3), padding=(0,1,1), stride = 1),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            HetConv(8 * (NC - 8), FM*4,
                p = 1,
                g = (FM*4)//4 if (8 * (NC - 8))%FM == 0 else (FM*4)//8,
                   ),
            nn.BatchNorm2d(FM*4),
            nn.ReLU()
        )

        self.last_BandSize = NC//2//2//2

        self.lidarConv = nn.Sequential(
                        nn.Conv2d(NCLidar,64,3,1,1),
                        nn.BatchNorm2d(64),
                        nn.GELU()
                        )
        self.ca = TransformerEncoder(FM*4)
        self.out3 = nn.Linear(FM*4 , Classes)
        self.position_embeddings = nn.Parameter(torch.randn(1, 4 + 1, FM*4))
        self.dropout = nn.Dropout(0.1)
        torch.nn.init.xavier_uniform_(self.out3.weight)
        torch.nn.init.normal_(self.out3.bias, std=1e-6)
        self.token_wA = nn.Parameter(torch.empty(1, 4, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.token_wA_L = nn.Parameter(torch.empty(1, 1, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA_L)
        self.token_wV_L = nn.Parameter(torch.empty(1, 64, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV_L)



    def forward(self, x1, x2):
        x1 = x1.reshape(x1.shape[0],-1,patchsize,patchsize)
        x1 = x1.unsqueeze(1)
        x2 = x2.reshape(x2.shape[0],-1,patchsize,patchsize)
        x1 = self.conv5(x1)
        x1 = x1.reshape(x1.shape[0],-1,patchsize,patchsize)

        x1 = self.conv6(x1)
        x2 = self.lidarConv(x2)
        x2 = x2.reshape(x2.shape[0],-1,patchsize**2)
        x2 = x2.transpose(-1, -2)
        wa_L = self.token_wA_L.expand(x1.shape[0],-1,-1)
        wa_L = rearrange(wa_L, 'b h w -> b w h')  # Transpose
        A_L = torch.einsum('bij,bjk->bik', x2, wa_L)
        A_L = rearrange(A_L, 'b h w -> b w h')  # Transpose
        A_L = A_L.softmax(dim=-1)
        wv_L = self.token_wV_L.expand(x2.shape[0],-1,-1)
        VV_L = torch.einsum('bij,bjk->bik', x2, wv_L)
        x2 = torch.einsum('bij,bjk->bik', A_L, VV_L)
        x1 = x1.flatten(2)

        x1 = x1.transpose(-1, -2)
        wa = self.token_wA.expand(x1.shape[0],-1,-1)
        wa = rearrange(wa, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x1, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)
        wv = self.token_wV.expand(x1.shape[0],-1,-1)
        VV = torch.einsum('bij,bjk->bik', x1, wv)
        T = torch.einsum('bij,bjk->bik', A, VV)
        x = torch.cat((x2, T), dim = 1) #[b,n+1,dim]
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        x = self.ca(embeddings)
        x = x.reshape(x.shape[0],-1)
        out3 = self.out3(x)
        return out3



def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def reports (xtest,xtest2,ytest,name,model):
    pred_y = np.empty((len(ytest)), dtype=np.float32)
    number = len(ytest) // testSizeNumber
    for i in range(number):
        temp = xtest[i * testSizeNumber:(i + 1) * testSizeNumber, :, :]
        temp = temp.cuda()
        temp1 = xtest2[i * testSizeNumber:(i + 1) * testSizeNumber, :, :]
        temp1 = temp1.cuda()

        temp2 = model(temp,temp1)

        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[i * testSizeNumber:(i + 1) * testSizeNumber] = temp3.cpu()
        del temp, temp2, temp3,temp1

    if (i + 1) * testSizeNumber < len(ytest):
        temp = xtest[(i + 1) * testSizeNumber:len(ytest), :, :]
        temp = temp.cuda()
        temp1 = xtest2[(i + 1) * testSizeNumber:len(ytest), :, :]
        temp1 = temp1.cuda()

        temp2 = model(temp,temp1)
        temp3 = torch.max(temp2, 1)[1].squeeze()
        pred_y[(i + 1) * testSizeNumber:len(ytest)] = temp3.cpu()
        del temp, temp2, temp3,temp1

    pred_y = torch.from_numpy(pred_y).long()

    if name == 'Electro':
        target_names = ['Mesh', 'Steel_Black', 'Steel_Grey', 'HTEL_Grey', 'HTEL_Black']
    # elif name == 'Trento':
    #     target_names = ['Apples','Buildings','Ground','Woods','Vineyard',
    #                     'Roads']
    # elif name == 'MUUFL' or name == 'MUUFLS' or name == 'MUUFLSR':
    #     target_names = ['Trees','Grass_Pure','Grass_Groundsurface','Dirt_And_Sand', 'Road_Materials','Water',"Buildings'_Shadow",
    #                 'Buildings','Sidewalk','Yellow_Curb','ClothPanels']
    # elif name == 'IP':
    #     target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
    #             ,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
    #             'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
    #             'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
    #             'Stone-Steel-Towers']
    # elif name == 'SA':
    #     target_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth',
    #                     'Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
    #                     'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
    #                     'Vinyard_untrained','Vinyard_vertical_trellis']
    # elif name == 'UP':
    #     target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
    #                     'Self-Blocking Bricks','Shadows']

#     classification = classification_report(ytest, pred_y, target_names=target_names)
    oa = accuracy_score(ytest, pred_y)
    confusion = confusion_matrix(ytest, pred_y)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(ytest, pred_y)

    return confusion, oa*100, each_acc*100, aa*100, kappa*100

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def create_hsi_rgb_patches_with_labels(hsi, mask, rgb, patch_size=7, fill_value_X=0, ignore_classes=[7, 8]):
    """
    Splits HSI and RGB images into patches of the specified size and assigns labels based on the center pixel of the corresponding ground truth mask.
    
    Parameters:
        hsi (numpy.ndarray): The HSI image or list of HSI images.
        mask (numpy.ndarray): The corresponding ground truth mask or list of masks.
        rgb (numpy.ndarray): The RGB image or list of RGB images.
        patch_size (int, optional): Size of the patches to be extracted. Defaults to 7 (7x7 patches).
        fill_value_X (float, optional): Value to fill the remaining pixels in the last incomplete patches. Defaults to 0.
        ignore_classes (list, optional): List of classes to be ignored when creating patches. Defaults to [7, 8].
    
    Returns:
        hsi_patches (numpy.ndarray): Array of extracted HSI patches.
        rgb_patches (numpy.ndarray): Array of extracted RGB patches.
        labels (numpy.ndarray): Array of labels corresponding to the center pixel of each patch.
    """
    hsi_patches = []
    rgb_patches = []
    labels = []
    
    # Calculate the center offset
    center_offset = patch_size // 2
    
    
    # Zip HSI, mask, and RGB together
    for hsi_img, mask_img, rgb_img in zip(hsi, mask, rgb):
        # Extract image dimensions (assuming HSI and RGB have same height and width)
        h, w, _ = hsi_img.shape  # Channels ignored for generality
        
        # Pad HSI, mask, and RGB to handle boundary conditions
        pad_hsi = np.pad(hsi_img, ((center_offset, center_offset), (center_offset, center_offset), (0, 0)), 
                        mode='constant', constant_values=fill_value_X)
        pad_mask = np.pad(mask_img, ((center_offset, center_offset), (center_offset, center_offset)), 
                         mode='constant', constant_values=fill_value_X)
        pad_rgb = np.pad(rgb_img, ((center_offset, center_offset), (center_offset, center_offset), (0, 0)), 
                        mode='constant', constant_values=fill_value_X)
    
        for i in range(h):
            for j in range(w):
                # Extract patches from padded HSI and RGB
                patch_hsi = pad_hsi[i:i+patch_size, j:j+patch_size, :]
                patch_rgb = pad_rgb[i:i+patch_size, j:j+patch_size, :]
            
                # Get the label of the center pixel from the original mask (before padding)
                label = mask_img[i, j]
            
                # Check if the label is in the ignore_classes list
                if label not in ignore_classes:
                    hsi_patches.append(patch_hsi)
                    rgb_patches.append(patch_rgb)
                    labels.append(label)
    
    # Convert lists to numpy arrays
    hsi_patches = np.array(hsi_patches)
    rgb_patches = np.array(rgb_patches)
    labels = np.array(labels)

    return hsi_patches, rgb_patches, labels

