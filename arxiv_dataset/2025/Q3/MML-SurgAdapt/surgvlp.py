import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torchvision.models import resnet50
from collections import OrderedDict
from model import GCN
import os
import numpy as np
from config import cfg

# Define MVNet
class MVNet(nn.Module):
    def __init__(self, backbone_img, backbone_text, neck=None, head=None):
        """
        Multimodal Vision and Text Network.

        Args:
            backbone_img (nn.Module): Backbone for image feature extraction.
            backbone_text (nn.Module): Backbone for text feature extraction.
            neck (nn.Module, optional): Neck module for feature processing.
            head (nn.Module, optional): Head module for downstream tasks.
        """
        super(MVNet, self).__init__()

        self.backbone_img = backbone_img  # Pretrained vision model (e.g., ResNet)
        self.backbone_text = backbone_text  # Pretrained text model (e.g., BERT)
        self.neck = neck
        self.head = head

    def extract_feat_img(self, inputs):
        """Extract features from the image backbone."""
        img_emb = self.backbone_img(inputs)
        return img_emb

    def extract_feat_text(self, input_ids, attn_mask, token_type_ids):
        """Extract features from the text backbone."""
        text_emb = self.backbone_text(
            input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids
        )
        return text_emb.last_hidden_state, text_emb.pooler_output  # Adjust as needed

    def forward(self, inputs_img=None, inputs_text=None, mode='video'):
        """
        Forward pass for different modes.

        Args:
            inputs_img (torch.Tensor, optional): Image tensor input.
            inputs_text (dict, optional): Text tensor input containing `input_ids`, `attention_mask`, and `token_type_ids`.
            mode (str): Mode of operation - 'video', 'text', or 'all'.

        Returns:
            dict: Extracted embeddings.
        """
        if mode == 'video' and inputs_img is not None:
            feats_img = self.extract_feat_img(inputs_img)
            return {'img_emb': feats_img}

        elif mode == 'text' and inputs_text is not None:
            feats_text_local, feats_text_global = self.extract_feat_text(
                input_ids=inputs_text['input_ids'],
                attn_mask=inputs_text['attention_mask'],
                token_type_ids=inputs_text['token_type_ids'],
            )
            return {'text_emb': feats_text_global}

        elif mode == 'all' and inputs_img is not None and inputs_text is not None:
            feats_img = self.extract_feat_img(inputs_img)
            feats_text_local, feats_text_global = self.extract_feat_text(
                input_ids=inputs_text['input_ids'],
                attn_mask=inputs_text['attention_mask'],
                token_type_ids=inputs_text['token_type_ids'],
            )
            return {'img_emb': feats_img, 'text_emb': feats_text_global}

        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

class SurgAVLP(nn.Module):

    def __init__(self,clip_model,classnames,weights,ckpt_path):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(weights)
        self.tokenized_prompts = self.tokenize(classnames)
        self.image_encoder, self.text_encoder = self.load_backbones_from_checkpoint(ckpt_path,weights)
        self.gcn = GCN()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.relation = torch.Tensor(np.load(cfg.relation_file))
        self.parent_index = np.load(cfg.super_labels_index)
        self.parent_index = torch.Tensor(self.parent_index).type(torch.long)

        child = self.relation[:cfg.child_num, :cfg.child_num].clone()
        parent = self.relation[cfg.child_num:, cfg.child_num:].clone()
        child = self.split(child)
        parent = self.split(parent)
        
        self.parent_self = parent.clone()
        self.relation = child

    def split(self, relation):
        _ ,max_idx = torch.topk(relation, int(3/4 * len(relation)))
        mask = torch.ones_like(relation).type(torch.bool)
        for i, idx in enumerate(max_idx):
            mask[i][idx] = 0
        relation[mask] = 0
        dialog = torch.eye(len(relation)).type(torch.bool)
        relation[dialog] = 0
        relation = relation / torch.sum(relation, dim=1).reshape(-1, 1) * cfg.reweight_p 
        relation[dialog] = (1-cfg.reweight_p)
        return relation

    def cuda(self, device=None):
        self.tokenized_prompts = {key: value.cuda(device) for key, value in self.tokenized_prompts.items()}
        return super().cuda(device)

    def tokenize(self,classnames):
        return self.tokenizer(classnames,padding=True, return_tensors="pt")

    def load_backbones_from_checkpoint(self,checkpoint_path,weights):
        """
        Load image and text backbones with weights from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the .pth.tar file.

        Returns:
            nn.Module, nn.Module: Image and text backbones with loaded weights.
        """
        # Define backbones
        backbone_img = resnet50(pretrained=False)  # ResNet50 without preloaded weights
        backbone_img.fc = nn.Linear(backbone_img.fc.in_features, 768)
        backbone_text = AutoModel.from_pretrained(weights)  # Bio_ClinicalBERT

        # Initialize MVNet
        model = MVNet(backbone_img=backbone_img, backbone_text=backbone_text)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']  # Extract state_dict from the checkpoint

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # Remove 'module.' prefix if present
            if k.startswith('module.'):
                k = k[7:]  # Remove the 'module.' prefix
            # Remove 'model.' prefix if present for both image and text backbones
            if 'model.' in k:
                k = k.replace('model.', '')  # Remove the 'model.' prefix globally

            # Rename keys for global_embedder to fc for ResNet image backbone
            if 'backbone_img.global_embedder' in k:
                k = k.replace('backbone_img.global_embedder', 'backbone_img.fc')

            new_state_dict[k] = v

        # Load weights into the model
        model.load_state_dict(new_state_dict, strict=False)  # Load with possible mismatched keys

        # Extract backbones
        image_backbone = model.backbone_img
        text_backbone = model.backbone_text

        return image_backbone, text_backbone

    def forward(self,image):

        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1,keepdim=True)
        text_features = self.text_encoder(**self.tokenized_prompts).pooler_output
        text_features = self.gcn(text_features, self.relation)
        text_features = text_features / text_features.norm(dim=-1,keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits
    

class CBertViT(nn.Module):

    def __init__(self,clip_model,classnames,weights):
        super().__init__()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(weights)
        self.tokenized_prompts = self.tokenize(classnames)
        self.image_encoder = clip_model.visual
        self.text_encoder = AutoModel.from_pretrained(weights)
        self.gcn = GCN()
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.relation = torch.Tensor(np.load(cfg.relation_file))
        self.parent_index = np.load(cfg.super_labels_index)
        self.parent_index = torch.Tensor(self.parent_index).type(torch.long)

        child = self.relation[:cfg.child_num, :cfg.child_num].clone()
        parent = self.relation[cfg.child_num:, cfg.child_num:].clone()
        child = self.split(child)
        parent = self.split(parent)
        
        self.parent_self = parent.clone()
        self.relation = child

    def split(self, relation):
        _ ,max_idx = torch.topk(relation, int(3/4 * len(relation)))
        mask = torch.ones_like(relation).type(torch.bool)
        for i, idx in enumerate(max_idx):
            mask[i][idx] = 0
        relation[mask] = 0
        dialog = torch.eye(len(relation)).type(torch.bool)
        relation[dialog] = 0
        relation = relation / torch.sum(relation, dim=1).reshape(-1, 1) * cfg.reweight_p 
        relation[dialog] = (1-cfg.reweight_p)
        return relation

    def cuda(self, device=None):
        self.tokenized_prompts = {key: value.cuda(device) for key, value in self.tokenized_prompts.items()}
        return super().cuda(device)

    def tokenize(self,classnames):
        return self.tokenizer(classnames,padding=True, return_tensors="pt")

    def forward(self,image):

        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1,keepdim=True)
        text_features = self.text_encoder(**self.tokenized_prompts).pooler_output
        text_features = self.gcn(text_features, self.relation)
        text_features = text_features / text_features.norm(dim=-1,keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits