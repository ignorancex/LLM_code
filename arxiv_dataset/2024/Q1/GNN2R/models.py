import torch
import torch.nn as nn
from torch_scatter import scatter
from torch import LongTensor, FloatTensor
from torch.nn.utils.rnn import pack_sequence
from transformers import AutoTokenizer, AutoModel


class AttenModel(nn.Module):
    def __init__(self, rel2embeds: dict, in_dim: int, hid_dim: int, num_layers: int,
                 dropout: float):
        super(AttenModel, self).__init__()
        self.hid_dim = hid_dim

        self.rel_embeds = [rel2embeds[r_id].cuda() for r_id in range(len(rel2embeds))]

        # relation encoder
        self.rel_embed_enc = nn.GRU(input_size=in_dim, hidden_size=hid_dim, bidirectional=False)

        # question encoders
        self.que_encs = nn.ModuleList()
        for l_id in range(num_layers):
            self.que_encs.append(
                nn.GRU(input_size=in_dim, hidden_size=hid_dim, bidirectional=True)
            )
        self.fin_que_enc = nn.GRU(input_size=in_dim, hidden_size=hid_dim, bidirectional=True)

        # graph encoders
        self.gra_encs = nn.ModuleList()
        for l_id in range(num_layers):
            self.gra_encs.append(AttenGcnLayer(hid_dim=hid_dim, dropout=dropout))

        # initial embedding of non-topic entities
        self.non_top_embed = nn.Parameter(torch.FloatTensor(1, hid_dim))
        nn.init.xavier_normal_(self.non_top_embed)

    def rel_enc(self):
        # encode relations based on labels
        packed_rel_embeds = pack_sequence(sequences=self.rel_embeds, enforce_sorted=False)
        r = self.rel_embed_enc(packed_rel_embeds)[1].squeeze(0)  # size: (num_rels, hid_dim)
        return r

    def forward(self, que_embeds: FloatTensor, r: FloatTensor, num_subg_ents: int,
                edge_index: LongTensor, edge_attr: LongTensor, loc_tops: LongTensor):
        # compute step-wise and final question embeddings
        h0s = [torch.zeros(2, self.hid_dim).cuda()]
        all_que_embeds = []
        for l_id, que_enc in enumerate(self.que_encs):
            _, h0 = que_enc(que_embeds, h0s[l_id])
            h0s.append(h0)
            all_que_embeds.append(torch.mean(h0, dim=0))
        fin_que_embed = torch.mean(self.fin_que_enc(que_embeds)[1], dim=0).expand(1, -1)  # size: (1, hid_dim)

        # initialize entity embeddings
        # x = torch.randn(num_subg_ents, self.hid_dim).cuda()
        x = self.non_top_embed.repeat(num_subg_ents, 1)
        x[loc_tops] = fin_que_embed.view(-1)

        # update entity embeddings via multistep GCN encoding with question embeddings as reference
        for l_id, gra_enc in enumerate(self.gra_encs):
            x, r = gra_enc(x=x, r=r, que_context=all_que_embeds[l_id], fin_que=fin_que_embed,
                           edge_index=edge_index, edge_attr=edge_attr)

        return (x,  # size: (num_ents_in_subg, hid_dim)
                fin_que_embed)  # size: (1, hid_dim)


class AttenGcnLayer(nn.Module):
    def __init__(self, hid_dim: int, dropout: float):
        super(AttenGcnLayer, self).__init__()
        # network for computing messages
        self.mess_comp = nn.Sequential(
            nn.Linear(2 * hid_dim, hid_dim),
            nn.Tanh()
        )

        # attention network for integrating messages
        self.mess_atten_trans = nn.Sequential(
            nn.Linear(2 * hid_dim, hid_dim),
            nn.LeakyReLU()
        )
        self.mess_atten_weight = nn.Parameter(torch.FloatTensor(1, hid_dim))
        nn.init.xavier_normal_(self.mess_atten_weight)

        # attention network for updating entity embeddings
        self.x_atten_trans = nn.Sequential(
            nn.Linear(2 * hid_dim, hid_dim),
            nn.LeakyReLU()
        )
        self.x_atten_weight = nn.Parameter(torch.FloatTensor(1, hid_dim))
        nn.init.xavier_normal_(self.x_atten_weight)

        # self.x_bn = nn.BatchNorm1d(hid_dim)
        self.r_bn = nn.BatchNorm1d(hid_dim)
        self.dp = nn.Dropout(dropout)

    def forward(self, x: FloatTensor, r: FloatTensor, que_context: FloatTensor, fin_que: FloatTensor,
                edge_index: LongTensor, edge_attr: LongTensor):
        # x = self.x_bn(x)
        r = self.r_bn(r)

        # compute passing messages
        head_ents, tail_ents = edge_index  # size: (num_edges,)
        head_embeds, rel_embeds = x[head_ents], r[edge_attr]  # size: (num_edges, hid_dim)
        mess = self.mess_comp(torch.cat([head_embeds, rel_embeds], dim=1))  # size: (num_edges, hid_dim)

        # entities only receive messages that are relevant to the given question
        mess_coeffs = self.mess_atten_trans(torch.cat([mess, que_context.expand(mess.size()[0], -1)],
                                                      dim=1))  # size: (num_edges, hid_dim)
        mess_coeffs = torch.sum(self.mess_atten_weight * mess_coeffs, dim=1)  # size: (num_edges,)
        mess_coeffs = mess_coeffs - scatter(src=mess_coeffs, index=tail_ents,
                                            dim=0, reduce='max')[tail_ents]  # size: (num_edges,)
        mess_coeffs = torch.exp(mess_coeffs)  # size: (num_edges,)
        mess_sum = scatter(src=mess_coeffs, index=tail_ents, dim=0, reduce='sum')[tail_ents]  # size: (num_edges,)
        mess_weights = mess_coeffs / (mess_sum + 1e-16)  # size: (num_edges,)

        mess = mess * mess_weights.unsqueeze(1)  # size: (num_edges, hid_dim)
        sum_mess = scatter(src=mess, index=tail_ents,
                           dim=0, reduce='sum', dim_size=x.size()[0])  # size: (num_ents_in_subg, hid_dim)

        # use attention to decide how much of incoming messages should be integrated into entity embeddings
        x_coeffs = torch.stack([x, sum_mess], dim=1)  # size: (num_ents_in_subg, 2, hid_dim)
        x_coeffs = self.x_atten_trans(torch.cat([x_coeffs, fin_que.expand(x.size()[0], 2, -1)],
                                                dim=2))  # size: (num_ents_in_subg, 2, hid_dim)
        x_coeffs = torch.sum(self.x_atten_weight.unsqueeze(0) * x_coeffs, dim=2)  # size: (num_ents_in_subg, 2)
        x_weight = nn.functional.softmax(x_coeffs, dim=1)  # size: (num_ents_in_subg, 2)

        x = x_weight[:, 0].unsqueeze(1) * x + x_weight[:, 1].unsqueeze(1) * sum_mess

        x = self.dp(x)  # size: (num_ents_in_subg, dim)
        r = self.dp(r)
        return x, r

# ----------------------------
# expression matching model

class ExpMatchModel(nn.Module):
    def __init__(self, lm_name: str, norm: int, reinit_n: int):
        super(ExpMatchModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self.model = AutoModel.from_pretrained(lm_name)
        self.norm = norm

        for n in range(reinit_n):
            self.model.transformer.layer[-(n+1)].apply(self._init_params)

    def que_embed(self, que: str):
        que_input = self.tokenizer(que, padding=True, truncation=True, return_tensors='pt').to(torch.cuda.current_device())
        que_embed = self.model(**que_input, return_dict=True).last_hidden_state
        que_mask_expanded = que_input['attention_mask'].unsqueeze(-1).expand(que_embed.size()).float()
        que_embed = torch.sum(que_embed * que_mask_expanded, 1) / torch.clamp(que_mask_expanded.sum(1), min=1e-16)
        que_embed = nn.functional.normalize(que_embed, p=self.norm, dim=1)
        return que_embed  # size: (1, 768)

    def forward(self, subg_exps: list):
        subg_exps_input = self.tokenizer(subg_exps, padding=True, truncation=True, return_tensors='pt').to(torch.cuda.current_device())
        subg_exps_embeds = self.model(**subg_exps_input, return_dict=True).last_hidden_state
        subg_exps_mask_expanded = subg_exps_input['attention_mask'].unsqueeze(-1).expand(
            subg_exps_embeds.size()).float()
        subg_exps_embeds = torch.sum(subg_exps_embeds * subg_exps_mask_expanded, 1) / torch.clamp(
            subg_exps_mask_expanded.sum(1), min=1e-16)
        subg_exps_embeds = nn.functional.normalize(subg_exps_embeds, p=self.norm, dim=1)
        return subg_exps_embeds  # size: (num_subg_exps, 768)

    def _init_params(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
