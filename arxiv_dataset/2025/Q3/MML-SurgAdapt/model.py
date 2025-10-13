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
    backbone_name = cfg.backbone
    if cfg.backbone == 'SurgVLP':
        backbone_name = 'RN50'
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

class ParentPromptLearner(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()

        assert(classnames is None)
        with open(cfg.super_labels, 'r') as f:
            text = f.readlines()
        classnames = [ t.strip() for t in text]

        logger.info(f"Super classnames: {classnames}")

        n_cls = len(classnames)
        n_ctx = cfg.parent_n_ctx
        dtype = clip_model.dtype

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # use given words to initialize context vectors
        logger.info(f"Use {cfg.parent_ctx_init} initialize parent prompt")
        ctx_init = cfg.parent_ctx_init.replace("_", " ")
        assert (n_ctx == len(ctx_init.split(" ")))
        prompt = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        ctx_vectors = embedding[0, 1:1 + n_ctx, :]
        prompt_prefix = ctx_init

        logger.info(f'Initial context: "{prompt_prefix}"')
        logger.info(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # type: ignore

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        #print(f"Prompt shape : {tokenized_prompts.shape}")
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(
                dtype)
        #print(f"Embedding shape : {embedding.shape}")

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix",
                             embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("token_middle", embedding[:, 1:(1 + n_ctx), :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        assert (embedding.requires_grad == False)
        assert (cfg.parent_ctx_init != "random")
        self.register_buffer("embedding", embedding)

    def forward(self):
        return self.embedding

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
            ("linear1", nn.Linear(1024, ctx_dim * n_ctx)),
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
    
class PromptLearner(nn.Module):

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
        if cfg.backbone == 'RN50':
            self.gc1 = GraphConvolution(1024, 2048)
            self.gc2 = GraphConvolution(2048, 2048)
            self.gc3 = GraphConvolution(2048, 1024)
        elif cfg.backbone == 'ViT-B/16':
            self.gc1 = GraphConvolution(512, 1024)
            self.gc2 = GraphConvolution(1024, 1024)
            self.gc3 = GraphConvolution(1024, 512)
        elif cfg.backbone == 'ViT-L/14':
            self.gc1 = GraphConvolution(768, 1536)
            self.gc2 = GraphConvolution(1536, 1536)
            self.gc3 = GraphConvolution(1536, 768)
        elif cfg.backbone == 'SurgVLP':
            self.gc1 = GraphConvolution(768, 1536)
            self.gc2 = GraphConvolution(1536, 1536)
            self.gc3 = GraphConvolution(1536, 768)
        else:
            raise NameError
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
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()

        self.emb_dim = emb_dim
        self.heads = heads
        self.head_dim = emb_dim // heads

        assert (
            self.head_dim * heads == emb_dim
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(emb_dim, emb_dim, bias=False)
        self.keys = nn.Linear(emb_dim, emb_dim, bias=False)
        self.queries = nn.Linear(emb_dim, emb_dim, bias=False)
        self.fc_out = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout_rate)

        # nn.init.xavier_uniform_(self.queries.weight)
        # nn.init.xavier_uniform_(self.keys.weight)
        # nn.init.xavier_uniform_(self.values.weight)
        # nn.init.xavier_uniform_(self.fc_out.weight)
        # nn.init.zeros_(self.fc_out.bias)

    def forward(self, values, keys, queries):
        bs, _, _ = queries.shape

        values = self.values(values).reshape(bs, -1, self.heads, self.head_dim)
        keys = self.keys(keys).reshape(bs, -1, self.heads, self.head_dim)
        queries = self.queries(queries).reshape(bs, -1, self.heads, self.head_dim)

        energy = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])

        attention = F.softmax(energy / (self.emb_dim ** (1 / 2)), dim=3)
        attention = self.dropout(attention)

        out = torch.einsum("bhql,blhd->bqhd", [attention, values]).reshape(
            bs, -1, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out
    
class TextToImageAttentionLayer(nn.Module):
    def __init__(self, emb_dim, dropout_rate=0.1):
        super(TextToImageAttentionLayer, self).__init__()

        self.cross_attention = MultiHeadAttention(
            emb_dim, heads=4, dropout_rate=dropout_rate
        )
        self.self_attention = MultiHeadAttention(
            emb_dim, heads=4, dropout_rate=dropout_rate
        )

        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text, image):

        attended_image = self.cross_attention(text, text, image)
        attended_image = self.dropout(attended_image)
        attended_image = self.layer_norm1(attended_image + image)  # Residual connection

        self_attended_image = self.self_attention(
            attended_image, attended_image, attended_image
        )
        self_attended_image = self.dropout(self_attended_image)
        self_attended_image = self.layer_norm2(
            self_attended_image + attended_image
        )  # Residual connection

        return self_attended_image
    

class ImageToTextAttentionLayer(nn.Module):
    def __init__(self, emb_dim, dropout_rate=0.1):
        super(ImageToTextAttentionLayer, self).__init__()

        self.cross_attention = MultiHeadAttention(
            emb_dim, heads=4, dropout_rate=dropout_rate
        )
        self.self_attention = MultiHeadAttention(
            emb_dim, heads=4, dropout_rate=dropout_rate
        )

        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, text, image):

        attended_text = self.cross_attention(image, image, text)
        attended_text = self.dropout(attended_text)
        attended_text = self.layer_norm1(attended_text + text)  # Residual connection

        self_attended_text = self.self_attention(
            attended_text, attended_text, attended_text
        )
        self_attended_text = self.dropout(self_attended_text)
        self_attended_text = self.layer_norm2(
            self_attended_text + attended_text
        )  # Residual connection

        return self_attended_text
    
class CrossModel(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.gcn = GCN()
        self.attend_txt = True
        self.attend_img = False
        self.attend_both = False
        assert (self.attend_img + self.attend_txt + self.attend_both) == 1

        if self.attend_img or self.attend_both:
            self.image_attention = nn.ModuleList(
                [
                    TextToImageAttentionLayer(emb_dim=512, dropout_rate=0.1)
                    for _ in range(2)
                ]
            )
        if self.attend_txt or self.attend_both:
            self.text_attention = nn.ModuleList(
                [
                    ImageToTextAttentionLayer(emb_dim=512, dropout_rate=0.1)
                    for _ in range(2)
                ]
            )

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
    
    def attend_to_img(self,text_labels,image,label): # 110,512   32,512  32,110
        if self.training:
            mask = label.unsqueeze(-1) # 32,110,1
        image = image.unsqueeze(1) # 32,1,512
        text_labels = text_labels.unsqueeze(0).repeat(image.shape[0],1,1) # 32,110,512
        if self.training:
            text_labels = text_labels * mask # 32,110,512
        img = []
        for i in range(text_labels.shape[1]):
            text = text_labels[:,i,:] # 32,512
            im = image # 32,1,512
            for layer in self.image_attention:
                im = layer(text,im) # 32,1,512
            im = im.squeeze(1) # 32,512
            img.append(im)

        image_features = torch.stack(img) # 110,32,512
        image_features = image_features.permute(1,0,2) # 32,110,512
        if self.training:
            image_features = image_features * mask # 32,110,512
            num_ones = mask.sum(dim=1) # 32,1
            image_features_sum = image_features.sum(dim=1)
            mean_image_features = image_features_sum/num_ones.clamp(min=1)
            mean_image_features = torch.where(num_ones==0,image,mean_image_features) #32,512
            return mean_image_features
        else:
            return text_labels, image_features
    
    def attend_to_text(self,text_labels,image1,label): # 110,512   32,512   32,110
        image = image1.unsqueeze(1) # 32,1,512
        text_labels = text_labels.unsqueeze(0).repeat(image.shape[0],1,1) # 32,110,512
        txt = []
        for i in range(text_labels.shape[1]):
            text = text_labels[:,i,:] # 32,512
            if self.training:
                labels = label[:,i].unsqueeze(1) # 32
            tx = text.unsqueeze(1) # 32,1,512
            for layer in self.text_attention:
                tx = layer(tx,image1) # 32,1,512
            tx = tx.squeeze(1) # 32,512
            if self.training:
                tx = torch.where(labels==0,text,tx)
            txt.append(tx)

        text_features = torch.stack(txt) # 110,32,512
        text_features = text_features.permute(1,0,2) # 32,110,512
        return text_features
    
    def forward(self, image, target = None):
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts,self.tokenized_prompts)

        text_features = self.gcn(text_features, self.relation)
        
        text_features = text_features / text_features.norm(dim=-1,
                                                            keepdim=True)
        logit_scale = self.logit_scale.exp()
        print(f"Logit scale: {logit_scale}")
    
        image_features = self.image_encoder(image)
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        
        if self.attend_txt:
            text_features = self.attend_to_text(text_features,image_features,target)
            image_features = image_features.unsqueeze(1)
            #print(f"text feat: {text_features}")
            logits = logit_scale * torch.matmul(image_features,text_features.transpose(-1,-2)).squeeze(1)
        if self.attend_img:
            if self.training:
                image_features = self.attend_to_img(text_features,image_features,target)
                logits = logit_scale * (image_features @ text_features.t())
            else:
                text_features, image_features = self.attend_to_img(text_features,image_features,target)
                # print(f"Img feat: {image_features.shape}")
                # print(f"txt feat: {text_features.shape}")
                logits = logit_scale * (image_features*text_features).sum(dim=-1)
        if self.attend_both:
            tf = self.attend_to_text(text_features,image_features,target)
            imf = self.attend_to_img(text_features,image_features,target).unsqueeze(1)
            logits = logit_scale * torch.matmul(imf,tf.transpose(-1,-2)).squeeze(1)

        # if not self.training:
        #     print(logits.shape)

        return logits

class MMLSurgAdapt(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.gcn = GCN()

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
    
    def encode_text(self, prompts, tokenized_prompts, text_adapter_func=None):
        if text_adapter_func is not None:
            text_features = self.text_encoder(
                prompts, tokenized_prompts, text_adapter_func
            )
        else:
            text_features = self.text_encoder(
                prompts, tokenized_prompts
            )
        return text_features
    
    def encode_image(self, image, visual_adapter_func=None):
        if visual_adapter_func is not None:
            image_features = self.image_encoder(
                [image.type(self.dtype), visual_adapter_func]
            )
        else:
            image_features = self.image_encoder(
                image.type(self.dtype)
            )
        return image_features
    
    def forward(self, image):

        child_prompts = self.prompt_learner()

        child_text_features = self.text_encoder(child_prompts,self.tokenized_prompts)

        text_features = child_text_features

        text_features = self.gcn(text_features, self.relation)
        
        text_features = text_features / text_features.norm(dim=-1,
                                                            keepdim=True)
        image_features = self.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        logits = 10 * image_features @ text_features.t()
        return logits

class VLPL(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        self.child_prompt_learner = PromptLearner(classnames, clip_model)
        self.child_tokeninzed_prompts = self.child_prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.temp = 0.03

        self.gcn = GCN()

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
    
    def encode_text(self, prompts, tokenized_prompts, text_adapter_func=None):
        if text_adapter_func is not None:
            text_features = self.text_encoder(
                prompts, tokenized_prompts, text_adapter_func
            )
        else:
            text_features = self.text_encoder(
                prompts, tokenized_prompts
            )
        return text_features
    
    def encode_image(self, image, visual_adapter_func=None):
        if visual_adapter_func is not None:
            image_features = self.image_encoder(
                [image.type(self.dtype), visual_adapter_func]
            )
        else:
            image_features = self.image_encoder(
                image.type(self.dtype)
            )
        return image_features
    
    def forward(self, image):
        child_prompts = self.child_prompt_learner()
        child_text_features = self.text_encoder(child_prompts,self.child_tokeninzed_prompts)

        text_features = child_text_features
        
        text_features = text_features / text_features.norm(dim=-1,
                                                            keepdim=True)
        image_features = self.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        logits = (1/self.temp) * image_features @ text_features.t()
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
        self.prompt_learner = PromptLearner(classnames, clip_model)
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
    
class HSPNet(nn.Module):

    def __init__(self, classnames, clip_model):
        super().__init__()
        self.parent_prompt_learner = ParentPromptLearner(None, clip_model)
        self.child_prompt_learner = ChildPromptLearner(classnames, clip_model)
        self.parent_tokenized_prompts = self.parent_prompt_learner.tokenized_prompts
        self.child_tokeninzed_prompts = self.child_prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.gcn = GCN()

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
    
    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1,
                                                              keepdim=True)
        parent_prompts = self.parent_prompt_learner()
        parent_text_features = self.text_encoder(parent_prompts, self.parent_tokenized_prompts)
        
        parent_text_features = self.gcn(parent_text_features, self.parent_self)
        child_prompts = self.child_prompt_learner(parent_text_features)
        child_text_features = self.text_encoder(child_prompts, self.child_tokeninzed_prompts)

        text_features = child_text_features

        text_features = self.gcn(text_features, self.relation)
        
        text_features = text_features / text_features.norm(dim=-1,
                                                            keepdim=True)
        logits = 10 * image_features @ text_features.t()
        return logits