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

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from medcam import medcam

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

        #print(dots.shape,"Dots Transformer")
        attn = self.attend(dots)

        #print(attn.shape,"Attention Transformer")
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        #print(out.shape,"out Transformer")

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


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MRI2Tokens(nn.Module):
    def __init__(self, in_chans=1, out_chans=64, kernel_size=7, stride=2):
        super(MRI2Tokens, self).__init__()
        self.conv = nn.Conv3d(in_chans, out_chans, kernel_size=kernel_size, stride=stride,
                              padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm3d(out_chans)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.maxpool(x)
        return x

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels,kernel_size=3,stride=2,enable_depthwise=True):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, dilation=1, groups=in_channels, bias=False, padding_mode='zeros')
        self.pointwise = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.bn = nn.BatchNorm3d(num_features=out_channels)
        self.enable_depthwise=enable_depthwise

    def forward(self, x):
        
        if self.enable_depthwise: 
            out = self.depthwise(x)
            out = self.pointwise(out)
        else:
            out = self.pointwise(x)

        return self.bn(out)



class LocallyEnhancedFeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 kernel_size=3, with_bn=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # pointwise
        self.conv1 = nn.Conv3d(in_features, hidden_features, kernel_size=1, stride=1, padding=0)
        # depthwise
        self.conv2 = nn.Conv3d(
            hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, groups=hidden_features
        )
        # pointwise
        self.conv3 = nn.Conv3d(hidden_features, out_features, kernel_size=1, stride=1, padding=0)
        self.act = act_layer()
        # self.drop = nn.Dropout(drop)

        self.with_bn = with_bn
        if self.with_bn:
            self.bn1 = nn.BatchNorm2d(hidden_features)
            self.bn2 = nn.BatchNorm2d(hidden_features)
            self.bn3 = nn.BatchNorm2d(out_features)

    def forward(self, x):
        b, n, k = x.size()

        cls_token, tokens = torch.split(x, [1, n - 1], dim=1)
        x = tokens.reshape(b, int(math.sqrt(n - 1)), int(math.sqrt(n - 1)), k).permute(0, 3, 1, 2)
        if self.with_bn:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act(x)
            x = self.conv3(x)
            x = self.bn3(x)
        else:
            x = self.conv1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.act(x)
            x = self.conv3(x)

        tokens = x.flatten(2).permute(0, 2, 1)
        out = torch.cat((cls_token, tokens), dim=1)
        return out


class TransformerBlock(nn.Module): 
    
    def __init__(self, num_patches,dim, depth, heads, dim_head, mlp_dim, patch_size,dropout = 0., depthwise_conv=None,features_to_include=[1,1,1]):
        super().__init__()

        self.pos_embed_single_layer   = nn.Parameter(torch.randn(1, num_patches, dim)).cuda()
         
        self.pos_embed_all_layers     = nn.Parameter(torch.randn(1, num_patches*3, dim)).cuda()



        self.cls_token   = nn.Parameter(torch.randn(1, 1, dim)).cuda()
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout = dropout)
        self.depth = depth
        self.heads = heads
        self.rearrange_to_emb = Rearrange('b c pf p1 p2 -> b (pf p1 p2) c')
        self.rearrange_to_blk = Rearrange('b (pf p1 p2) c -> b c pf p1 p2 ',p1=patch_size,p2=patch_size,pf=patch_size)

        self.features_to_include = features_to_include 

    def forward(self, x1,x2,x3):
        
        x1 = self.rearrange_to_emb(x1)
        x2 = self.rearrange_to_emb(x2)
        x3 = self.rearrange_to_emb(x3)

        #print( self.pos_embed.shape,"Positional head","Input",x.shape)
        x1 = x1 + self.pos_embed_single_layer
        x2 = x2 + self.pos_embed_single_layer
        x3 = x3 + self.pos_embed_single_layer 

        #print(self.features_to_include)
        
        #print("Depth",self.depth,"Gead",self.heads)
        #print(self.pos_embed_single_layer.shape,"Pos mebedding")

        x = torch.cat((x1*self.features_to_include[0],x2*self.features_to_include[1],x3*self.features_to_include[2]),dim=1)
        #print(x.shape,"Concatanated features")

        x = x + self.pos_embed_all_layers
        b,_,_ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        #print(cls_tokens.shape,"Class embedding features")


        x = torch.cat((cls_tokens,x),dim=1)
        #print(x.shape,"Final embedding features")

        x  = self.transformer(x)
        #x  = self.rearrange_to_blk(x)
        
        return x



class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 n_output_channels=64,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2,
                 t_depth=4,
                 t_heads=8,
                 embedding_dim=256,
                 mlp_dim=2048,
                 enable_transformer =True,
                 include_fc=False,
                 features_to_include=[1,1,1]):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.include_fc = include_fc
        self.mri2tokens =  MRI2Tokens(n_input_channels,64,kernel_size=conv1_t_size)
        #TODO Hardcoded number of patches for now. Need to calculate based on depthwise convolution. Please change this later you lazy bastard! 
        
        
        
        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.enable_transformer = enable_transformer
        self.enable_l3_dwconv = False
        self.pool = "cls"

        if self.enable_transformer:
            num_patches = (4**3)
            #self.enable_l3_dwconv = False 
            self.transformer_block = TransformerBlock(num_patches, embedding_dim, depth=t_depth, heads=t_heads, dim_head=n_output_channels, mlp_dim=mlp_dim, patch_size= 4, dropout = 0.1, depthwise_conv=None, features_to_include=features_to_include)  
            
            
            self.depthwise_sep_conv_l1 = DepthwiseSeparableConv(in_channels=n_output_channels, out_channels=embedding_dim,stride=4)
            self.depthwise_sep_conv_l2 = DepthwiseSeparableConv(in_channels=n_output_channels*2, out_channels=embedding_dim,stride=2)

            if embedding_dim != n_output_channels*4:
                self.enable_l3_dwconv = True
                self.depthwise_sep_conv_l3 = DepthwiseSeparableConv(in_channels=n_output_channels*4, out_channels=embedding_dim,stride=1,enable_depthwise=False)
            self.fc = nn.Sequential(nn.Linear(block_inplanes[3] * block.expansion + embedding_dim, block_inplanes[3] * block.expansion //2), nn.ReLU(), nn.Linear(block_inplanes[3] * block.expansion //2, n_classes))
        else:
            self.fc = nn.Sequential(nn.Linear(block_inplanes[3] * block.expansion, block_inplanes[3] * block.expansion //2), nn.ReLU(), nn.Linear(block_inplanes[3] * block.expansion //2, n_classes))
        #self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.mri2tokens(x)

      
        x = self.layer1(x)

        if self.enable_transformer: l1 = self.depthwise_sep_conv_l1(x)
        
        
        x = self.layer2(x)

        if self.enable_transformer: l2 = self.depthwise_sep_conv_l2(x)

        x = self.layer3(x)
        if self.enable_l3_dwconv:
            l3 = self.depthwise_sep_conv_l3(x)
        else:
            l3 = x
        if self.enable_transformer: t_x = self.transformer_block(l1,l2,l3)

        if self.enable_transformer: t_x = t_x.mean(dim = 1) if self.pool == 'mean' else t_x[:, 0]

        
        x = self.layer4(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)


        if self.enable_transformer: x = torch.cat((x,t_x),dim=1)
        
        x = self.fc(x)

        return x#,None,None


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


class ResNetClassifer(LightningModule):

    def __init__(self,cfg,prefix=None):
        super(ResNetClassifer, self).__init__()
        

        self.cfg = cfg
        
        self.model = generate_model(model_depth=self.cfg.model_depth,t_heads=self.cfg.t_heads,t_depth=self.cfg.t_depth,embedding_dim=self.cfg.embedding_dim,mlp_dim=self.cfg.mlp_dim,enable_transformer=self.cfg.enable_transformer, n_output_channels=self.cfg.n_output_channels, features_to_include=self.cfg.features_to_include )
        #print(medcam.get_layers(self.model))
        #if self.cfg.save_attention_maps:
        #    self.model = medcam.inject(self.model, output_dir=self.cfg.attn_map_output_dir, backend=self.cfg.backend, layer=['layer1', 'layer2','layer3','layer4'], label='best', save_maps=True)
        
        self.prefix =prefix
        self.softmax = torch.nn.Softmax(dim=1)
        # Loss function
        cls_weights = torch.tensor(list(self.cfg.class_weights)).float()
        self.criterion = torch.nn.CrossEntropyLoss(weight=cls_weights).cuda()
        
    

    def configure_optimizers(self):
        
        optimizer =  optim.Adam(self.model.parameters() ,lr=self.cfg.lr, amsgrad=False)
        scheduler =  optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,verbose=True,patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler,"monitor": "val/loss"} #[optimizer],[scheduler]

    def prepare_batch(self, batch):
        
        return batch['image'][tio.DATA],batch['label'],batch['patient_disease_id'],batch['image_path']
    
    def training_step(self, batch, batch_idx):

        inputs,y,patient_disease_id,_ = self.prepare_batch(batch)
        #outputs,intermediate_output_1,intermediate_output_2  = self.model(inputs)
        outputs  = self.model(inputs)
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
        #outputs,intermediate_output_1,intermediate_output_2  = self.model(inputs)
        outputs  = self.model(inputs)
        #print(outputs["image"].shape,inputs.shape)
        # calculate loss
        
        loss = self.criterion(outputs,y)
        target = self.softmax(outputs)

        
        AnomalyScoreReco_vol = target[:,1]

        self.val_eval_dict['labelPerVol'] = torch.cat((torch.tensor(self.val_eval_dict['labelPerVol']), y.cpu()), 0)
        self.val_eval_dict['AnomalyScorePerVol'] = torch.cat((torch.tensor(self.val_eval_dict['AnomalyScorePerVol']), AnomalyScoreReco_vol.cpu()), 0)
        # TODO Need to change the line below
        self.val_eval_dict['AnomalyScorePerVol_one_instance'] = torch.cat((torch.tensor(self.val_eval_dict['AnomalyScorePerVol_one_instance']), AnomalyScoreReco_vol.cpu()), 0)
        
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
        #outputs,intermediate_output_1,intermediate_output_2  = self.model(inputs)
        outputs  = self.model(inputs)
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


        AnomalyScoreReco_vol_one_instance  = target[0,1].item()
        self.eval_dict['AnomalyScorePerVol_one_instance'].append(AnomalyScoreReco_vol_one_instance)

        """
        self.eval_dict['AnomalyScoreRecoPerVol'].append(AnomalyScoreReco_vol)
        self.eval_dict['AnomalyScoreRecoPerVolL2'].append(0)
        #Dummy values put in place to prevent code from breaking
        self.eval_dict['AnomalyScoreRegPerVol'].append(0)
        self.eval_dict['AnomalyScoreCombiPerVol'].append(0)
        self.eval_dict['AnomalyScoreCombiPerVolL2'].append(0)

        self.eval_dict['AnomalyScoreCombPriorPerVol'].append(0)
        self.eval_dict['AnomalyScoreCombPriorPerVolL2'].append(0)
        self.eval_dict['KLD_to_learned_prior'].append(0)
        """
        
        # calculate metrics
        _test_step(self, None,None,inputs,y[0],patient_disease_id[0],batch_idx) # everything that is independent of the model choice

        
        