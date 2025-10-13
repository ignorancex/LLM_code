import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from perturbedtopk import PerturbedTopK


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, hidden_dim = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, hidden_dim),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, hidden_dim), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A


    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.blocks=nn.Sequential(
            self.build_block(in_dim=self.input_dim, out_dim=hidden_dim),
            self.build_block(in_dim=hidden_dim, out_dim=hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=self.output_dim),
        )
        

    def build_block(self, in_dim, out_dim):
        return nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

    
class MLPMixerLayer(nn.Module):
    def __init__(self, num_tokens, dim, hidden_dim):
        super(MLPMixerLayer, self).__init__()
        
        # Token mixing (across the token/sequence dimension)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens)
        )
        
        # Channel mixing (across the feature dimension)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        # Token mixing
        x = x + self.token_mlp(x.permute(0, 2, 1)).permute(0, 2, 1)
        # Channel mixing
        x = x + self.channel_mlp(x)
        return x

class MLPMixer(nn.Module):
    def __init__(self, num_tokens, dim, hidden_dim, num_layers=2):
        super(MLPMixer, self).__init__()
        
        self.layers = nn.ModuleList([MLPMixerLayer(num_tokens, dim, hidden_dim) for _ in range(num_layers)])
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

class BClassifier(nn.Module):
    def __init__(self, input_size, input_size_deep, stop_gradient='no', top_k=10, mlp_layers=4, N_TOKENS_RNA=4848): # K, L, N
        super(BClassifier, self).__init__()
        
        self.attention_deep = Attn_Net_Gated(L=input_size_deep, hidden_dim=input_size_deep)  

        self.pre_attn = nn.Sequential(
            nn.Linear(input_size_deep, input_size_deep),
            nn.LayerNorm(input_size_deep),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_size_deep, input_size_deep),
            nn.LayerNorm(input_size_deep),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.top_k = top_k
        self.k_sigma = 0.002

        self.aux_ga = Attn_Net_Gated(L=self.top_k, hidden_dim=input_size_deep)
        
        self.mlp_mixer = MLPMixer(num_tokens=input_size, dim=top_k, hidden_dim=input_size_deep, num_layers=mlp_layers) # changed from 128, 2		
        
        self.stop_gradient = stop_gradient
        
        self.rna_embedder = MLP(input_dim=N_TOKENS_RNA, hidden_dim=N_TOKENS_RNA, output_dim=input_size_deep)

        # head for deep features
        self.clip_proj_head = nn.Sequential(
            nn.Linear(input_size_deep, input_size)
        )

        # head for deep features
        self.clip_proj_rna_head = nn.Sequential(
            nn.Linear(input_size_deep, input_size)
        )
        

    def forward(self, feats, feats_deep, rna_seq, training='no'): # B x N x K, B x N x C
        device = feats_deep.device
        
        V_deep = self.pre_attn(feats_deep) # B x N x D

        A_patch = self.attention_deep(V_deep) # B x N x 1

        A_patch = F.softmax(A_patch, 1)  # B x N x 1
        
        B_deep = A_patch*V_deep   #  Final output (B) - BxNxD

        _, topk_indices = torch.sort(A_patch.clone().detach(), 1, descending=True)
            
        if topk_indices.shape[1]<self.top_k:  # repeat necessary to make the model run as we need to fix features input
            repeat_factor = int(self.top_k//topk_indices.shape[1]) + 1
            topk_indices = topk_indices.repeat(1, repeat_factor, 1)
        
        # print('feats shape', feats.shape)
        topk_feats = (feats.gather(1, topk_indices[:, :self.top_k, :].expand(-1, -1, feats.shape[-1])).clone()).permute(0, 2, 1) # B x D x topk
        # print('feats shape', feats.shape)

        topk_feats = self.mlp_mixer(topk_feats)  # B x D x topk

        A_feat = self.aux_ga(topk_feats).squeeze(-1)   # input: B x D x topk, output: B x D

        A_feat = F.sigmoid(A_feat)          # B x D      
        
        # print('feats shape, self.top_k, training', feats.shape, self.top_k, training)

        if feats.shape[1] > self.top_k:
            if self.stop_gradient == 'no' and training=='yes':
                topk_selector = PerturbedTopK(k=self.top_k, num_samples=100, sigma=self.k_sigma)

                # to check if .squeeze() is needed at the end or not.
                # topk_indices = topk_selector(A_patch.squeeze(-1)).squeeze() # feed BxN to get output of size B x top_k X N

                topk_indices = topk_selector(A_patch.squeeze(-1)) # feed BxN to get output of size B x top_k X N
                perturbed_topk_feats = torch.einsum('bkn,bnd->bkd', topk_indices, feats) # B x top_k x D

                perturbed_topk_feats_deep = torch.einsum('bkn,bnd->bkd', topk_indices, V_deep) # B x top_k x D

            else:	
                _, topk_indices = torch.sort(A_patch.clone().detach(), 1, descending=True)		
                # print('topk_indices', topk_indices[:,:5])

                perturbed_topk_feats = (feats.gather(1, topk_indices[:, :self.top_k, :].expand(-1, -1, feats.shape[-1])).clone()) # B x topk x D

                perturbed_topk_feats_deep = (V_deep.gather(1, topk_indices[:, :self.top_k, :].expand(-1, -1, V_deep.shape[-1])).clone()) # B x topk x D
                # print('perturbed_topk_feats[:,0]', perturbed_topk_feats[:,0])

        else:
            perturbed_topk_feats = feats
            perturbed_topk_feats_deep = V_deep

        B = perturbed_topk_feats * A_feat.unsqueeze(1) 	# B x top_k/N x D   # / when self.top_k>feats.shape[0] 
        
        B_deep = B_deep.sum(1)
        B_deep_proj = self.clip_proj_head(B_deep)

        rna_emb = self.rna_embedder(rna_seq)
        rna_emb_proj = self.clip_proj_rna_head(rna_emb)
        
        return B.sum(1), perturbed_topk_feats.sum(1), B_deep_proj, B_deep, rna_emb_proj, rna_emb, A_feat, A_patch


class MILNet(nn.Module):
    def __init__(self, b_classifier):
        super(MILNet, self).__init__()
        self.b_classifier = b_classifier
        
    def forward(self, x, x_deep, rna_seq, training='no'):
        B, perturbed_topk_feats, B_deep_proj, B_deep, rna_emb_proj, rna_emb, A_feat, A_patch = self.b_classifier(x, x_deep, rna_seq, training=training)
        return B, perturbed_topk_feats, B_deep_proj, B_deep, rna_emb_proj, rna_emb, A_feat, A_patch
