#Using COOP in our CLIP CLASSIFIER MODEL

from torchvision.models import resnet152, resnet101
import torch.nn as nn
import torch 
import clip
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        
        return x

class TokenizedPrompt(nn.Module):
    def __init__(self):
        super().__init__()
        pass


    def forward(self, classnames, clip_model):
        n_cls = len(classnames)
        n_ctx = 3
        ctx_init = "A photo of x"#
        dtype = clip_model.dtype
        ctx_dim = 512#clip_model.ln_final.weight.shape[0]
        clip_imsize = 224#clip_model.visual.input_resolution
        cfg_imsize = 224

        ctx_init = ctx_init.replace("_", " ")
        prompt = clip.tokenize(ctx_init).to("cuda:1")
        

        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)
        prompt_prefix = ctx_init

        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to("cuda:1")

        return tokenized_prompts


class ClipClassifier(nn.Module):
    def __init__(self, settings,  clip_model=None):
        super(ClipClassifier, self).__init__()
        self.clip = clip_model
        self.pdrop = settings['pdrop']
        self.text_encoder = TextEncoder(clip_model)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.d_lin = nn.Linear(10,1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc11 = nn.Linear(512,512).half()
        self.fc12 = nn.Linear(512,512).half()

        # self.fc1 = nn.Linear(512,64).half()
        self.fc1 = nn.Linear(512,64).half()
        self.fc2 = nn.Linear(64,1).half()
        self.n_ctx =  3
        self.n_ctx_d = 1
        self.ctx_init = "A photo of x"#
        self.dtype = clip_model.dtype
        self.ctx_dim = 512#clip_model.ln_final.weight.shape[0]
        self.clip_imsize = 224#clip_model.visual.input_resolution
        self.cfg_imsize = 224
        self.ctx = Parameter()
        self.ctx_init = self.ctx_init.replace("_", " ")        
        self.prompt = clip.tokenize(self.ctx_init).to("cuda:1")
        with torch.no_grad():
            self.embedding = self.clip.token_embedding(self.prompt).type(self.dtype)
        self.ctx_vectors = self.embedding[0, 1 : 1+ self.n_ctx, :]
        self.prompt_prefix = self.ctx_init


        self.ctx = Parameter(self.ctx_vectors,requires_grad= True)
        
        self.dom = nn.Linear(54,512).half()
        
        self.alpha = Parameter(torch.Tensor(1))
        self.alpha.data.fill_(0.9)

        hdim = 512
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)
    def construct_soft_prompt(self, classnames, clip_model, domains):
        n_cls = len(classnames)
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to("cuda:1")

        with torch.no_grad():

            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        token_prefix = embedding[:, :1, :]
        token_suffix = embedding[:, 1 + self.n_ctx+1 :, :]

        if self.ctx.dim() == 2:
            ctx = self.ctx.unsqueeze(0).expand(n_cls, -1, -1)
        ctx_d = self.dom(torch.tensor(domains).to("cuda:1").type(clip_model.dtype))
        if ctx_d.dim() == 2:
            ctx_d = ctx_d.unsqueeze(0).permute(1,0,2)#.expand(64, -1, -1)

        prefix = token_prefix
        suffix = token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                ctx_d,
                suffix  # (n_cls, *, dim)
            ],
            dim=1,
        )

        prompts = prompts.type(clip_model.dtype)

        return prompts,tokenized_prompts,ctx_d
    def forward(self,qimage_clip_processed, qtext_clip_tokenized, caption_u,domain):
        

        encoded_img = self.clip.encode_image(qimage_clip_processed)

        prompt_learn,tokenized_prompts,ctx_d =  self.construct_soft_prompt(caption_u,self.clip,domain)

     
        text_features = self.text_encoder(prompt_learn,tokenized_prompts)

  

        
        encoded_img = encoded_img / encoded_img.norm(dim=-1, keepdim=True) 
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)



        joint_features = encoded_img*text_features
        joint_embeddings = text_features
        
  
        joint_features = F.dropout(joint_features, p=self.pdrop)

        ### CLIP_adapter
        f1 =  self.fc11(joint_features)
        f1 = self.relu(f1)
        f2 =  self.fc12(f1)

        f3 = self.alpha.half() * joint_features + (1-self.alpha.half())* f2
        joint_features = F.dropout(f3, p=self.pdrop)


        consis_out =  self.fc1(joint_features)
        consis_out = self.bn2(consis_out)
        consis_out = self.relu(consis_out)
        consis_out = self.fc2(consis_out) 
        


        return consis_out,  encoded_img,ctx_d
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False).half()
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False).half()
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False).half()
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model).half()
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
 
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        # print(q.size())
        sz_b, len_q,_ = q.size()
        sz_b, len_k,_ = k.size()
        sz_b, len_v,_ = v.size()

        residual = q
        residual = residual.half()

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))

        output = self.layer_norm((output + residual).float())


        return output
    
