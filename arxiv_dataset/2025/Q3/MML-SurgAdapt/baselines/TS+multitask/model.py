from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from config import cfg
from log import logger

_tokenizer = _Tokenizer()

def load_clip_to_cpu():
    backbone_name = 'ViT-L/14'
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(  # type: ignore
            model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())  # type: ignore

    return model


class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, retrun_adapater_func=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if retrun_adapater_func == None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, retrun_adapater_func])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]),
              tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class ChildPromptLearner(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        classnames = classnames[0:cfg.child_num]
        n_cls = len(classnames)
        n_ctx = cfg.child_n_ctx
        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        ctx_dim = clip_model.ln_final.weight.shape[0]
        #print(f"Ctx_dim : {ctx_dim}")
        self.ctx_dim = ctx_dim

        prompt_prefix = " ".join(["X"] * n_ctx)
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(512, ctx_dim * n_ctx)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(ctx_dim * n_ctx, ctx_dim * n_ctx))
        ]))
    
        # use given words to initialize context vectors
        logger.info(f"Number of child context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix",
                             embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        self.parent_index = np.load(cfg.super_labels_index)
        self.parent_index = torch.Tensor(self.parent_index).type(torch.long)
        logger.info(f"{self.parent_index}")
        assert(self.parent_index.requires_grad == False)

    def forward(self, parent):
        prefix = self.token_prefix
        suffix = self.token_suffix
        #print(f"Parent shape before meta net: {parent.shape}")
        parent = self.meta_net(parent)
        #print(f"Parent shape after meta net: {parent.shape}")
        parent = parent[self.parent_index]
        parent = parent.reshape(-1, self.n_ctx, self.ctx_dim)
        #print(f"Parent shape after reshaping: {parent.shape}")
        #print(f"Prefix shape: {prefix.shape}")
        #print(f"Suffix shape: {suffix.shape}")
        prompts = torch.cat(
            [
                prefix,
                parent,
                suffix
            ],
            dim = 1
        )
        return prompts
    
class ChildPromptLearnerNew(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        classnames = classnames[0:cfg.child_num]
        n_cls = len(classnames)
        n_ctx = cfg.child_n_ctx
        dtype = clip_model.dtype
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        ctx_dim = clip_model.ln_final.weight.shape[0]
        #print(f"Ctx_dim : {ctx_dim}")
        self.ctx_dim = ctx_dim

        prompt_prefix = "a photo "
    
        # use given words to initialize context vectors
        logger.info(f"Number of child context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix",
                             embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        assert (embedding.requires_grad == False)
        self.register_buffer("embedding", embedding)

    def forward(self):
        return self.embedding

def load_clip_model():
    clip_model = load_clip_to_cpu()

    # CLIP's default precision is fp16
    clip_model.float()
    return clip_model, clip._transform(clip_model.visual.input_resolution)

import math
import numpy as np
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        # self.gc1 = GraphConvolution(1024, 2048)
        # self.gc2 = GraphConvolution(2048, 2048)
        # self.gc3 = GraphConvolution(2048, 1024)
        self.gc1 = GraphConvolution(512, 1024)
        self.gc2 = GraphConvolution(1024, 1024)
        self.gc3 = GraphConvolution(1024, 512)
        self.relu = nn.LeakyReLU(0.2)
        self.relu2 = nn.LeakyReLU(0.2)
        self.gamma = torch.nn.Parameter(torch.ones(1) * 0.9, requires_grad=True)
    
    def forward(self, features, relation):
        identity = features
        assert(relation.requires_grad == False)
        text_features = features
        text_features = self.gc1(text_features, relation.cuda())
        text_features = self.relu(text_features)
        text_features = self.gc2(text_features, relation.cuda())
        text_features = self.relu2(text_features)
        text_features = self.gc3(text_features, relation.cuda())
        text_features = self.gamma * text_features + (1-self.gamma) * identity
        return text_features

class HSPNet(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        self.child_prompt_learner = ChildPromptLearnerNew(classnames, clip_model)
        self.child_tokeninzed_prompts = self.child_prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        print(self.image_encoder.transformer.resblocks[11].mlp.c_proj)
        self.image_encoder.transformer.resblocks[11].mlp.c_proj.weight = \
            nn.Parameter(torch.zeros_like(self.image_encoder.transformer.resblocks[11].mlp.c_proj.weight))
        self.image_encoder.transformer.resblocks[11].mlp.c_proj.bias = \
            nn.Parameter(torch.zeros_like(self.image_encoder.transformer.resblocks[11].mlp.c_proj.bias))
        
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.gcn = GCN()

        self.relation = torch.Tensor(np.load(cfg.relation_file))

        child = self.relation.clone()
        child = self.split(child)

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
    
    def forward(self, image):
        # visual_adapter_func = self.adapter_learner()
        # image_features = self.encode_image(image, visual_adapter_func)
        #image_features = self.image_encoder(image)
        #image_features = image_features / image_features.norm(dim=-1,
        #                                                      keepdim=True)
        #print(f"Image features: {image_features.shape}")
        #parent_prompts = self.parent_prompt_learner()
        #parent_prompts, text_adapter_func, visual_adapter_func = self.adapter_learner(parent_prompts)
        #print(f"Parent prompt shape: {parent_prompts.shape}")
        #parent_text_features = self.text_encoder(parent_prompts,self.parent_tokenized_prompts)
        #parent_text_features = self.encode_text(parent_prompts,self.parent_tokenized_prompts,text_adapter_func)
        #print(f"Parent text features: {parent_text_features.shape}")
        
        #parent_text_features = self.gcn(parent_text_features, self.parent_self)
        child_prompts = self.child_prompt_learner()
        #print(f"Child prompt shape: {child_prompts.shape}")
        #child_prompts, text_adapter_func, visual_adapter_func = self.adapter_learner(child_prompts)
        #child_text_features = self.encode_text(child_prompts, self.child_tokeninzed_prompts,text_adapter_func)
        child_text_features = self.text_encoder(child_prompts,self.child_tokeninzed_prompts)
        #print(f"Child text features: {child_text_features.shape}")

        text_features = child_text_features

        text_features = self.gcn(text_features, self.relation)
        
        text_features = text_features / text_features.norm(dim=-1,
                                                            keepdim=True)
        #print(f"Final text features shape : {text_features.shape}")
        #logit_scale = self.logit_scale.exp()
        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        logits = 10 * image_features @ text_features.t()
        #print(f"Logit shape : {logits.shape}")
        #print("-------------------------------------")
        return logits

class Resnet(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        self.image_encoder = models.resnet50(pretrained=True)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features,len(classnames))
    
    def forward(self, image):
        return self.image_encoder(image)
    
class ViT(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        self.image_encoder = models.vit_b_16(weights="ViT_B_16_Weights.DEFAULT")
        self.image_encoder.heads = nn.Sequential(nn.Linear(self.image_encoder.heads[0].in_features, len(classnames)))
        
    def forward(self, image):
        return self.image_encoder(image)
    
class CLIP_for_train(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = ChildPromptLearnerNew(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):

        child_prompts = self.prompt_learner()
        text_features = self.text_encoder(child_prompts,self.tokenized_prompts)
        
        text_features = text_features / text_features.norm(dim=-1,
                                                            keepdim=True)
        logit_scale = self.logit_scale.exp()
        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1,
                                                                keepdim=True)
        logits = logit_scale * image_features @ text_features.t()
        return logits