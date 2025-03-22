import torch
import torch
import torch.nn as nn
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))) 

from utils_TransTEE.transformers import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
from utils_TransTEE.utils import get_initialiser
from utils_TransTEE.mlp import MLP
from utils_TransTEE.trans_ci import TransformerModel, Embeddings

# replace the feature extractor of x by self-attention
# 0.015
from transformers import CLIPModel, CLIPProcessor

class Linear(nn.Module):
    def __init__(self, ind, outd, act='relu', isbias=1):
        super(Linear, self).__init__()
        # ind does NOT include the extra concat treatment
        self.ind = ind
        self.outd = outd
        self.isbias = isbias

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        # x: batch_size * (treatment, other feature)

        out = torch.matmul(x, self.weight)

        if self.isbias:
            out = out + self.bias

        if self.act is not None:
            out = self.act(out)

        return out

def make_transtee_backbone(args, input_size, hidden_size, cont_treatment,num_heads=2, att_layers=1, init_range_f=0.1, dropout=0.0, cov_dim=498):
    if not cont_treatment:
        if args.dataset_name == "tcga":
            return torch.nn.Sequential(nn.Linear(input_size, 100), Embeddings(hidden_size, initrange=init_range_f), TransformerEncoder(TransformerEncoderLayer(hidden_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout, num_cov=100), att_layers))    
            
        
        else:
            return torch.nn.Sequential(nn.Linear(input_size, input_size), Embeddings(hidden_size, initrange=init_range_f), TransformerEncoder(TransformerEncoderLayer(hidden_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout, num_cov=cov_dim), att_layers))
    else:
        # return torch.nn.Sequential(nn.Linear(input_size, 100), Embeddings(hidden_size, initrange=init_range_f), TransformerEncoder(TransformerEncoderLayer(hidden_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout, num_cov=100), att_layers))    
        return torch.nn.Sequential(Embeddings(hidden_size, initrange=init_range_f), TransformerEncoder(TransformerEncoderLayer(hidden_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout, num_cov=cov_dim), att_layers))#, nn.Flatten(), nn.Linear(cov_dim*hidden_size, hidden_size))


class TransTEE(nn.Module):
    def __init__(self, params, num_heads=2, att_layers=1, dropout=0.0, init_range_f=0.1, init_range_t=0.1, has_dose=False, cont_treatment = False, num_class= None):
        super(TransTEE, self).__init__()
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]
        num_features = params['num_features']
        self.num_treatments = params['num_treatments']
        # self.export_dir = params['export_dir']

        h_dim = params['h_dim']
        # self.h_inv_eqv_dim = params['h_inv_eqv_dim']
        # self.batch_size = params['batch_size']
        # self.alpha = params['alpha']
        # self.num_dosage_samples = params['num_dosage_samples']
        
        self.linear1 = nn.Linear(num_features, 100)

        self.feature_weight = Embeddings(h_dim, initrange=init_range_f)
        self.treat_emb = Embeddings(h_dim, act='id', initrange=init_range_t)
        if has_dose:
            self.dosage_emb = Embeddings(h_dim, act='id', initrange=init_range_t)
            self.linear2 = MLP(
                dim_input=h_dim * 2,
                dim_hidden=h_dim,
                dim_output=h_dim,
            )
        else:
            self.linear2 = MLP(
                dim_input=h_dim,
                dim_hidden=h_dim,
                dim_output=h_dim,
            )

        encoder_layers = TransformerEncoderLayer(h_dim, nhead=num_heads, dim_feedforward=h_dim, dropout=dropout, num_cov=params['cov_dim'])
        self.encoder = TransformerEncoder(encoder_layers, att_layers)

        decoder_layers = TransformerDecoderLayer(h_dim, nhead=num_heads, dim_feedforward=h_dim, dropout=dropout,num_t=1)
        self.decoder = TransformerDecoder(decoder_layers, att_layers)

        out_dim = 1
        if num_class is not None:
            out_dim = num_class
        self.num_class = num_class
        self.Q = MLP(
            dim_input=h_dim,
            dim_hidden=h_dim,
            dim_output=out_dim,
            is_output_activation=False,
        )
        
        self.has_dose = has_dose
        self.cont_treatment = cont_treatment
        self.base_treatment = None
        self.base_dose = None

    def encoding_features(self, x):
        if not self.cont_treatment and self.has_dose:
            hidden = self.feature_weight(self.linear1(x))
        else:
            hidden = self.feature_weight(x)
        return hidden

    def forward(self, x, t, d=None, test=False, hidden=None):
        if hidden is None:
            if not self.cont_treatment and self.has_dose:
                hidden = self.feature_weight(self.linear1(x))
            else:
                hidden = self.feature_weight(x)
        
        memory = self.encoder(hidden)

        if (not test) or (self.cont_treatment):
            t = t.view(t.shape[0], 1)
            
            if self.has_dose:
                d = d.view(d.shape[0], 1)
                tgt = torch.cat([self.treat_emb(t), self.dosage_emb(d)], dim=-1)
            else:
                tgt = self.treat_emb(t)
            tgt = self.linear2(tgt)
            if len(tgt.shape) < 3:
                tgt = tgt.unsqueeze(1)
            out = self.decoder(tgt.permute(1, 0, 2), memory.permute(1, 0, 2))
            if out.shape[0] != 1:
                out = torch.mean(out, dim=1)
            Q = self.Q(out.squeeze(0))
            # return torch.mean(hidden, dim=1).squeeze(), Q
            return torch.mean(hidden, dim=1).squeeze(), Q
        else:
            Q_ls = []         
            for tt in range(self.num_treatments):
                tt = (torch.ones_like(t)*tt).view(t.shape[0], 1)
            
                if self.has_dose:
                    d = d.view(d.shape[0], 1)
                    tgt = torch.cat([self.treat_emb(tt), self.dosage_emb(d)], dim=-1)
                else:
                    tgt = self.treat_emb(tt)
                tgt = self.linear2(tgt)
                if len(tgt.shape) < 3:
                    tgt = tgt.unsqueeze(1)
                out = self.decoder(tgt.permute(1, 0, 2), memory.permute(1, 0, 2))
                if out.shape[0] != 1:
                    out = torch.mean(out, dim=1)
                Q = self.Q(out.squeeze(0))
                Q_ls.append(Q)
            Q = torch.stack(Q_ls, dim=1)
            curr_Q = Q[torch.arange(len(Q)), t.squeeze().long()]
            return torch.mean(hidden, dim=1).squeeze(), curr_Q, Q

    def _initialize_weights(self, initialiser):
        # TODO: maybe add more distribution for initialization
        initialiser = get_initialiser(initialiser)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # if m.in_features == 1:
                #     continue
                initialiser(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def predict_given_treatment_dose(self, x):

        x = torch.from_numpy(x).float().to(self.device)
        treatment_array = torch.ones(len(x), device=self.device)*self.base_treatment
        dose = None
        if self.base_dose is not None:
            dose = torch.ones(len(x), device=self.device)*self.base_dose
        _, out=self.forward(x, treatment_array, dose)
        
        if self.num_class is None:
            return out.detach().cpu().numpy()
        else:
            return out.argmax(-1).detach().cpu().view(-1,1).numpy()
        
    def predict_given_treatment_dose2(self, x):

        x = torch.from_numpy(x).float().to(self.device)
        treatment_array = torch.ones(len(x), device=self.device)*self.base_treatment
        dose = None
        if self.base_dose is not None:
            dose = torch.ones(len(x), device=self.device)*self.base_dose
        _, out=self.forward(x, treatment_array, dose)
        
        if self.num_class is None:
            return out.detach().cpu().numpy()
        else:
            return out.argmax(-1).detach().cpu().view(-1).numpy()

class MonoTransTEE(nn.Module):
    def __init__(self, model, t, dose=None):
        super(MonoTransTEE, self).__init__()
        self.t = t
        self.dose = dose
        self.model = model
    
    # shapley requires forward fn of type: Tensor -> Tensor
    def forward(self, x): 
        n = len(x)
        t_n = torch.ones(len(x), device=x.device)*self.t
        d_n = None
        if self.dose is not None:
            d_n = torch.ones(len(x), device=x.device)*self.dose
            # d_n = self.dose.repeat(n)
        return self.model.forward(x, t_n, d_n)[1]

class TransTEE_image(nn.Module):
    def __init__(self, params, num_heads=2, att_layers=1, dropout=0.0, init_range_f=0.1, init_range_t=0.1, has_dose=False, cont_treatment = False):
        super(TransTEE_image, self).__init__()
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]
        num_features = params['num_features']
        self.num_treatments = params['num_treatments']
        # self.export_dir = params['export_dir']

        h_dim = params['h_dim']
        # self.h_inv_eqv_dim = params['h_inv_eqv_dim']
        # self.batch_size = params['batch_size']
        # self.alpha = params['alpha']
        # self.num_dosage_samples = params['num_dosage_samples']
        
        # self.linear1 = nn.Linear(num_features, 100)
        self.feature_weight = Embeddings(h_dim, initrange=init_range_f)
        self.img_emb = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.img_emb.visual_projection = nn.Linear(self.img_emb.visual_projection.in_features, h_dim)
        # self.feature_weight = Embeddings(h_dim, initrange=init_range_f)
        self.treat_emb = Embeddings(h_dim, act='id', initrange=init_range_t)
        if has_dose:
            self.dosage_emb = Embeddings(h_dim, act='id', initrange=init_range_t)
            self.linear2 = MLP(
                dim_input=h_dim * 2,
                dim_hidden=h_dim,
                dim_output=h_dim,
            )
        else:
            self.linear2 = MLP(
                dim_input=h_dim,
                dim_hidden=h_dim,
                dim_output=h_dim,
            )

        encoder_layers = TransformerEncoderLayer(h_dim, nhead=num_heads, dim_feedforward=h_dim, dropout=dropout, num_cov=num_features)
        self.encoder = TransformerEncoder(encoder_layers, att_layers)

        decoder_layers = TransformerDecoderLayer(h_dim, nhead=num_heads, dim_feedforward=h_dim, dropout=dropout,num_t=1)
        self.decoder = TransformerDecoder(decoder_layers, att_layers)

        self.Q = MLP(
            dim_input=h_dim, #self.img_emb.visual_projection.out_features + self.treat_emb.emb_size,
            dim_hidden=h_dim,
            dim_output=1,
            is_output_activation=False,
        )
        
        self.has_dose = has_dose
        self.cont_treatment = cont_treatment

    def forward(self, images, x, t, d=None, test=False):
        # if not self.cont_treatment:
            # hidden = self.feature_weight(self.linear1(x))
        # else:
        #     hidden = self.feature_weight(x)

        # image = Image.fromarray(x.astype(np.uint8))
        # inputs = self.processor(images=image, return_tensors="pt", padding=False)
        # with torch.no_grad():
        image_emb = self.img_emb.get_image_features(**images)
        # tab_emb = self.feature_weight(self.linear1(x))
        tab_emb = self.feature_weight(x)
        memory = self.encoder(tab_emb)
        input_emd = torch.cat([image_emb.unsqueeze(1), memory], dim=1)
        if (not test):
            tgt = self.treat_emb(t)
            out = self.decoder(tgt.permute(1, 0, 2), input_emd.permute(1, 0, 2))
            if out.shape[0] != 1:
                out = torch.mean(out, dim=1)
            # out = torch.cat([image_emb, tgt], dim=-1)
            Q = self.Q(out.squeeze(0))
            return Q
        else:
            Q_ls = []         
            for tt in range(self.num_treatments):
                tt = (torch.ones_like(t)*tt).view(t.shape[0], 1)
                tgt = self.treat_emb(tt)#.squeeze(1)
                # out = torch.cat([image_emb, tgt], dim=-1)
                out = self.decoder(tgt.permute(1, 0, 2), input_emd.permute(1, 0, 2))
                if out.shape[0] != 1:
                    out = torch.mean(out, dim=1)
                Q = self.Q(out.squeeze(0))
                Q_ls.append(Q)
            Q = torch.cat(Q_ls, dim=-1)
            curr_Q = Q[torch.arange(len(Q)), t.squeeze().long()]
            return curr_Q, Q

        # if (not test) or (self.cont_treatment):
        #     t = t.view(t.shape[0], 1)
            
        #     if self.has_dose:
        #         d = d.view(d.shape[0], 1)
        #         tgt = torch.cat([self.treat_emb(t), self.dosage_emb(d)], dim=-1)
        #     else:
        #         tgt = self.treat_emb(t)
        #     tgt = self.linear2(tgt)
        #     if len(tgt.shape) < 3:
        #         tgt = tgt.unsqueeze(1)
        #     out = self.decoder(tgt.permute(1, 0, 2), memory.permute(1, 0, 2))
        #     if out.shape[0] != 1:
        #         out = torch.mean(out, dim=1)
        #     Q = self.Q(out.squeeze(0))
        #     # return torch.mean(hidden, dim=1).squeeze(), Q
        #     return torch.mean(hidden, dim=1).squeeze(), Q
        # else:
        #     Q_ls = []         
        #     for tt in range(self.num_treatments):
        #         tt = (torch.ones_like(t)*tt).view(t.shape[0], 1)
            
        #         if self.has_dose:
        #             d = d.view(d.shape[0], 1)
        #             tgt = torch.cat([self.treat_emb(tt), self.dosage_emb(d)], dim=-1)
        #         else:
        #             tgt = self.treat_emb(tt)
        #         tgt = self.linear2(tgt)
        #         if len(tgt.shape) < 3:
        #             tgt = tgt.unsqueeze(1)
        #         out = self.decoder(tgt.permute(1, 0, 2), memory.permute(1, 0, 2))
        #         if out.shape[0] != 1:
        #             out = torch.mean(out, dim=1)
        #         Q = self.Q(out.squeeze(0))
        #         Q_ls.append(Q)
        #     Q = torch.cat(Q_ls, dim=-1)
        #     curr_Q = Q[torch.arange(len(Q)), t.squeeze().long()]
        #     return torch.mean(hidden, dim=1).squeeze(), curr_Q, Q

    def _initialize_weights(self, initialiser):
        # TODO: maybe add more distribution for initialization
        initialiser = get_initialiser(initialiser)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # if m.in_features == 1:
                #     continue
                initialiser(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
                    
