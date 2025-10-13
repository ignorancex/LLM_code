import torch
import numpy as np
import torch.nn as nn
from modules.base_cmn import BaseCMN
from modules.visual_extractor import VisualExtractor


text_list = ['pneumothorax','pleural', 'spine', 'heart', 'hernia', 'lung', 'mediastinal', 'Cardiac', 'Bony', 'Emphysema', 'Atelectasis', 'lobe', 'clavicle', 'Cardiomediastinal', 'osseous', 'mediastinum', 'aorta', 'aortic', 'diaphragm', 'thoracic', 'vascularity', 'pulmonary' ]

text_list_mimic = ['cholecystectomy', 'subclavian', 'emphysema', 'bronchovascular', 'heart', 'neck', 'wires', 'hilum', 'mediastinum', 'cardiac', 'hemithorax', 'rib', 'sternotomy', 'chest', 'tubes', 'osseous', 'diaphragms', 'bony', 'silhouette', 'hilar', 'mediastinal', 'perihilar','vasculature', 'pulmonary', 'hemidiaphragms', 'lung', 'cardiomediastinal', 'pneumothorax', 'pleural']

class CrossAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value):
        # Cross-attention
        attn_output, _ = self.multihead_attn(query, key, value)
        query = self.layernorm1(query + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(query)
        query = self.layernorm2(query + ffn_output)
        return query


class BaseModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(BaseModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = BaseCMN(args, tokenizer)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr
        
        self.image_proj = nn.Linear(2048, 512)
        self.text_proj = nn.Linear(4, 512)
     
        self.cross_attention = CrossAttention(d_model=512, n_heads=8)
        self.out = nn.Linear(512, 2048)
        
        self.text_list = [torch.tensor(tokenizer(i), dtype = torch.float32).reshape(1,-1) for i in text_list]
        self.text_feature = torch.cat(self.text_list, dim=0).cuda()

        self.text_list_mimic = [torch.tensor(tokenizer(i), dtype = torch.float32).reshape(1,-1) for i in text_list_mimic]
        self.text_feature_mimic = torch.cat(self.text_list_mimic, dim=0).cuda()
    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_iu_xray(self, images, targets=None, mode='train', update_opts={}):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        
        # start
        att_feats_0 = self.image_proj(att_feats_0)  # Shape: (batchsize, 49, 512)
        text_features = self.text_proj(self.text_feature)     # Shape: (20, 512)
        text_features = text_features.unsqueeze(0).expand(att_feats_0.size(0), text_features.size(0), text_features.size(1)) 

        att_feats_0 = self.cross_attention(query=text_features, key=att_feats_0, value=att_feats_0)
        
        att_feats_0 = self.out(att_feats_0)
        # end
        
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output, output_probs
        else:
            raise ValueError

    def forward_mimic_cxr(self, images, targets=None, mode='train', update_opts={}):
        att_feats, fc_feats = self.visual_extractor(images)

        att_feats1 = self.image_proj(att_feats)  # Shape: (batchsize, 49, 512)
        text_features = self.text_proj(self.text_feature_mimic)     # Shape: (20, 512)
        text_features = text_features.unsqueeze(0).expand(att_feats.size(0), text_features.size(0), text_features.size(1)) 

        att_feats1 = self.cross_attention(query=text_features, key=att_feats1, value=att_feats1)
        
        att_feats1 = self.out(att_feats1)
        
        att_feats = torch.cat((att_feats, att_feats1), dim=1)
 
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return output
        elif mode == 'sample':
            output, output_probs = self.encoder_decoder(fc_feats, att_feats, mode='sample', update_opts=update_opts)
            return output#, output_probs
        else:
            raise ValueError