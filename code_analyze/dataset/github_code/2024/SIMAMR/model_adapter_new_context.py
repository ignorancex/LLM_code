import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# import nlp
from torch.distributions import Categorical
from nn_utils import PositionalEncoding, has_nan, universal_sentence_embedding, clip_and_normalize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers import BertForSequenceClassification, ElectraForSequenceClassification, AutoTokenizer, AutoModel


from Transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,

)

cc = SmoothingFunction()

device = "cuda" if torch.cuda.is_available() else "cpu"

class AttentionSelector(nn.Module):
    def __init__(self, config):
        super(AttentionSelector, self).__init__()
        self.dense = nn.Linear(2 * config.d_enc_concept, 1)

    def forward(self, hidden_states):
        #first_token_tensor = hidden_states[:, 0]
        # [batch_size, seq_length, hiddden_size] -> [batch_size, seq_length]
        selector_tensor = self.dense(hidden_states)
        # .squeeze(-1)
        return torch.sigmoid(selector_tensor)
        
class BERTAttentionSelector(nn.Module):
    def __init__(self, config):
        super(BERTAttentionSelector, self).__init__()
        self.dense = nn.Linear(config.d_enc_concept, 1)

    def forward(self, hidden_states):
        #first_token_tensor = hidden_states[:, 0]
        # [batch_size, seq_length, hiddden_size] -> [batch_size, seq_length]
        selector_tensor = self.dense(hidden_states)
        # .squeeze(-1)
        return torch.sigmoid(selector_tensor)
class BeamInstance:
    def __init__(self, ids, neg_logp, is_finish):
        self.ids = ids
        self.neg_logp = neg_logp
        self.is_finish = is_finish

    def get_logp_norm(self, eos_id):
        try:
            l = self.ids.index(eos_id)
            return self.neg_logp / (l)
        except ValueError:
            return self.neg_logp / (len(self.ids) - 1)

    def get_ids(self, eos_id):
        try:
            i = self.ids.index(eos_id)
            return self.ids[1:i]
        except ValueError:
            return self.ids[1:]

class BERTEncoder(nn.Module):
    def __init__(self, config):
        super(BERTEncoder, self).__init__()
        self.config = config
        self.encoder = BertForSequenceClassification.from_pretrained('./test-ddbothc1')

    def forward(self, batch):
        
        # if len(batch["st"]) == len(batch["token_type_ids"]):
        # print("batch[st]", batch["st"].shape)
        # print("batch[tyid]", batch["token_type_ids"].shape)
        output = self.encoder(input_ids=batch["st"].to(device), token_type_ids=batch["token_type_ids"][:,:512].to(device), attention_mask=batch["st_mask"].to(device), output_hidden_states=True)
        src_emb = output["hidden_states"][0]                   # [batch, seq, dim]
        src_emb = torch.mean(src_emb, dim=1)

        # print("BERTEncoder src_hidden", src_emb)      
        # batch["src_emb"] = src_emb
        # assert has_nan(src_emb) is False
        if has_nan(src_emb) is False:
            return src_emb

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class SentBERTEncoder(nn.Module):
    def __init__(self, config):
        super(SentBERTEncoder, self).__init__()
        self.config = config
        self.encoder = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

    def forward(self, batch):
       
        output = self.encoder(batch["src"].to(device), attention_mask=batch["src_mask"].to(device), output_hidden_states=True)

        # src_emb = mean_pooling(output, batch["src_mask"].to(device))
                            #   , attention_mask=, output_hidden_states=True)
        # print('output ', [i.shape for i in output["hidden_states"]])
        src_emb = output["hidden_states"][0]                   # [batch, seq, dim]
        src_emb = torch.mean(src_emb, dim=1)

        assert has_nan(src_emb) is False
        return src_emb  # [batch, dim]

class SentTransformer(nn.Module):
    def __init__(self, config, bpemb, vocab):
        super(SentTransformer, self).__init__()

        self.config = config
        self.embedding = bpemb
        self.vocab = vocab

        assert vocab is not None
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.position_encoder = PositionalEncoding(config.d_enc_sent)
        # encoder_layer = nn.TransformerEncoderLayer(
        #     config.d_model, config.n_head, dim_feedforward=1024, dropout=config.dropout
        # )
        encoder_layer = TransformerEncoderLayer(
            d_model=config.d_enc_sent,
            heads=config.n_head,
            d_ff=getattr(config, "d_ff", 1024),
            dropout=config.dropout,
            att_drop=config.dropout,
            use_structure=False,
        )
        encoder_norm = nn.LayerNorm(config.d_enc_sent)
        self.encoder = TransformerEncoder(encoder_layer, config.num_layer, encoder_norm)

        if vocab is not None:
            self.vocab_size = len(self.vocab)
            self.BOS = self.vocab["<bos>"]
            self.EOS = self.vocab["<eos>"]
        else:
            self.vocab_size = self.bpemb.vectors.shape[0]
            self.BOS = self.bpemb.BOS
            self.EOS = self.bpemb.EOS

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch):
        # encoding
        src_mask_inv = (batch["src_mask"] == 0).to(device)   # [batch, seq]
        src_emb = self.embedding(batch["src"].to(device))  # [batch, seq, dim]
        src_emb = self.position_encoder(src_emb.transpose(0, 1)).transpose(0, 1) # [batch, seq, dim]
        src_emb = self.encoder(src_emb, src_key_padding_mask=src_mask_inv)  # [batch, seq, dim]
        
        tgt_mask_inv = (batch["tgt_mask"] == 0).to(device)   # [batch, seq]
        tgt_emb = self.embedding(batch["tgt_input"].to(device))  # [batch, seq, dim]
        tgt_emb = self.position_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1) # [batch, seq, dim]
        tgt_emb = self.encoder(tgt_emb, src_key_padding_mask=tgt_mask_inv)  # [batch, seq, dim]
        
        src_emb = torch.mean(src_emb, dim=1)
        tgt_emb = torch.mean(tgt_emb, dim=1)
        
        src_emb = torch.cat([src_emb, tgt_emb], dim=-1)

        # assert has_nan(src_emb) is False
        return src_emb

class ResTransformer(nn.Module):
    def __init__(self, config, bpemb, vocab):
        super(ResTransformer, self).__init__()

        self.config = config
        self.embedding = bpemb
        self.vocab = vocab

        assert vocab is not None
        self.vocab_inv = {v: k for k, v in vocab.items()}
        self.position_encoder = PositionalEncoding(config.d_enc_sent)
        # encoder_layer = nn.TransformerEncoderLayer(
        #     config.d_model, config.n_head, dim_feedforward=1024, dropout=config.dropout
        # )
        encoder_layer = TransformerEncoderLayer(
            d_model=config.d_enc_sent,
            heads=config.n_head,
            d_ff=getattr(config, "d_ff", 1024),
            dropout=config.dropout,
            att_drop=config.dropout,
            use_structure=False,
        )
        encoder_norm = nn.LayerNorm(config.d_enc_sent)
        self.encoder = TransformerEncoder(encoder_layer, config.num_layer, encoder_norm)

        if vocab is not None:
            self.vocab_size = len(self.vocab)
            self.BOS = self.vocab["<bos>"]
            self.EOS = self.vocab["<eos>"]
        else:
            self.vocab_size = self.bpemb.vectors.shape[0]
            self.BOS = self.bpemb.BOS
            self.EOS = self.bpemb.EOS

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch):
        # encoding
        src_mask_inv = (batch["tgt_mask"] == 0).to(device)   # [batch, seq]
        # batch["src_mask_inv"] = src_mask_inv
        src_emb = self.embedding(batch["tgt_input"].to(device))  # [batch, seq, dim]
        # print("input_size", history_emb.size())
        src_emb = self.position_encoder(src_emb.transpose(0, 1)).transpose(0, 1)                                       # [batch, seq, dim]
        src_emb = self.encoder(src_emb, src_key_padding_mask=src_mask_inv)  # [batch, seq, dim]
        assert has_nan(src_emb) is False
        return src_emb

class ElectraEncoder(nn.Module):
    def __init__(self, config):
        super(ElectraEncoder, self).__init__()
        self.config = config
        self.encoder = ElectraForSequenceClassification.from_pretrained("bhadresh-savani/electra-base-emotion")

    def forward(self, batch):
        output = self.encoder(input_ids=batch["src"].to(device), attention_mask=batch["src_mask"].to(device), output_hidden_states=True)
        src_emb = output["hidden_states"][2]                   # [batch, seq, dim]
        # print("src_hidden", src_emb.size())      
        batch["src_emb"] = src_emb
        assert has_nan(src_emb) is False
        return src_emb


class GraphTransformer(nn.Module):
    def __init__(self, config, bpemb, vocab, relation_vocab):
        super(GraphTransformer, self).__init__()

        self.config = config
        self.embedding = bpemb
        self.vocab = vocab
        self.relation_vocab = relation_vocab
        self.n_layer = getattr(config, "g_num_layer", 4)
        self.use_pe = getattr(config, "g_pe", True)

        assert vocab is not None
        self.vocab_inv = {v: k for k, v in vocab.items()}
        assert relation_vocab is not None
        self.relation_embedding = nn.Embedding(len(self.relation_vocab), config.d_relation)
        assert config.d_relation * config.g_n_head == config.d_enc_concept

        self.position_encoder = PositionalEncoding(config.d_enc_concept)

        # encoder_layer = nn.TransformerEncoderLayer(
        #     config.d_model, config.n_head, dim_feedforward=1024, dropout=config.dropout
        # )
        encoder_layer = TransformerEncoderLayer(
            d_model=config.d_enc_concept,
            heads=config.g_n_head,
            d_ff=getattr(config, "g_d_ff", config.d_enc_concept*2),
            dropout=config.dropout,
            att_drop=config.dropout,
            use_structure=True,
        )
        encoder_norm = nn.LayerNorm(config.d_enc_concept)
        self.encoder = TransformerEncoder(encoder_layer, self.n_layer, encoder_norm)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch):
        # encoding

        src_mask_inv = (batch["con_mask"] == 0).to(device)  # [batch, seq]
        # batch["con_mask_inv"] = src_mask_inv
        # print('batch["con"]', batch["con"])
        src_emb = self.embedding(batch["con"].to(device))  # [batch, seq, dim]
        structure_emb = self.relation_embedding(batch["rel"].to(device))  # [batch, seq, seq, s_dim]

        # print('src_emb', src_emb)
        if self.use_pe:
            src_emb = self.position_encoder(src_emb.transpose(0, 1)).transpose(0, 1)  # [batch, seq, dim]
        src_emb = self.encoder(src_emb, src_key_padding_mask=src_mask_inv, structure=structure_emb)  # [batch, seq, dim]
        # print('src_emb after pe', src_emb)
        
        ###
        # src_emb = torch.cat([src_emb,\
        #     self.encoder(self.embedding(batch["tgt_input"].to(device)), src_key_padding_mask=(batch["tgt_mask"] == 0).to(device))], dim=1)  # [batch, seq, dim]
        # ###

        tgt_mask_inv = (batch["context_concept_mask"] == 0).to(device)  # [batch, seq]
        # batch["con_mask_inv"] = src_mask_inv
        tgt_emb = self.embedding(batch["context_concept"].to(device))  # [batch, seq, dim]
        tgt_structure_emb = self.relation_embedding(batch["context_rel"].to(device))  # [batch, seq, seq, s_dim]

        if self.use_pe:
            tgt_emb = self.position_encoder(tgt_emb.transpose(0, 1)).transpose(0, 1)  # [batch, seq, dim]
        tgt_emb = self.encoder(
            tgt_emb, src_key_padding_mask=tgt_mask_inv, structure=tgt_structure_emb
        )  # [batch, seq, dim]

        # ###
        # tgt_emb = torch.cat([tgt_emb,\
        #     self.encoder(self.embedding(batch["src"].to(device)), src_key_padding_mask=(batch["src_mask"] == 0).to(device))], dim=1)  # [batch, seq, dim]
        # ##

        # print('src_emb', src_emb.shape)
        assert has_nan(src_emb) is False
        # if has_nan(src_emb) is True:
        return [src_emb, tgt_emb]

class AdapterGraphTransformer(nn.Module):
    def __init__(self, config, relation_vocab):
        super(AdapterGraphTransformer, self).__init__()

        self.config = config
        self.relation_vocab = relation_vocab
        self.n_layer = getattr(config, "adapter_layer", 2)
        self.use_pe = getattr(config, "adapter_pe", True)

        assert relation_vocab is not None
        d_relation = config.d_enc_sent // config.n_head
        self.relation_embedding = nn.Embedding(len(self.relation_vocab), d_relation)
        self.position_encoder = PositionalEncoding(config.d_enc_sent)

        encoder_layer = TransformerEncoderLayer(
            d_model=config.d_enc_sent,
            heads=config.n_head,
            d_ff=getattr(config, "g_d_ff", 1024),
            dropout=config.dropout,
            att_drop=config.dropout,
            use_structure=True,
        )
        encoder_norm = nn.LayerNorm(config.d_enc_sent)
        self.encoder = TransformerEncoder(encoder_layer, self.n_layer, encoder_norm)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_emb, batch):
        # encoding
        src_mask_inv = (batch["src_mask"] == 0).to(device)   # [batch, seq]
        # src_mask_inv = batch["src_mask_inv"]
        # print('src_emb:', src_emb.size())
        # assert len(src_emb.size()) == 3, 'Invalid input size:{}, should be 3!!'.format(' x '.join([str(itm) for itm in src_emb.size()]))
        structure_emb = self.relation_embedding(batch["wr"].to(device))  # [batch, seq, seq, s_dim]
        assert len(structure_emb.size()) == 4, 'Invalid input size:{}, should be 3!!'.format(' x '.join([str(itm) for itm in structure_emb.size()]))
        assert src_emb.size(1) == structure_emb.size(1) and src_emb.size(1) == structure_emb.size(2)
        # print('structure size:', structure_emb.size())
        # print("input_size", src_emb.size())
        if self.use_pe:
            src_emb = self.position_encoder(src_emb.transpose(0, 1)).transpose(0, 1)  # [batch, seq, dim]
        # print('src_emb_pos:', src_emb.size())
        src_emb = self.encoder(
            src_emb, src_key_padding_mask=src_mask_inv, structure=structure_emb
        )  # [batch, seq, dim]
        # exit()
        assert has_nan(src_emb) is False
        return src_emb


class DualTransformerBert(nn.Module):
    
    def __init__(self, config, word_emb, con_emb=None, word_vocab=None, concept_vocab=None, relation_vocab=None, word_rel_vocab=None):
        super(DualTransformerBert, self).__init__()

        self.config = config
        self.word_vocab = word_vocab
        self.concept_vocab = concept_vocab
        self.relation_vocab = relation_vocab

        self.enc_word_embedding = self.build_embedding(word_emb, word_vocab, self.config.d_enc_sent)

        
        # self.word_encoder = SentTransformer(config, self.enc_word_embedding, word_vocab)
        # self.res_encoder = ResTransformer(config, self.enc_word_embedding, word_vocab)
        self.word_encoder = BERTEncoder(config)

        if config.dual_enc and self.concept_vocab is not None and relation_vocab is not None:
            if config.share_con_vocab:
                self.enc_concept_embedding = self.enc_word_embedding
            else:
                self.enc_concept_embedding = self.build_embedding(con_emb, concept_vocab, self.config.d_enc_concept)
            
            self.graph_encoder = GraphTransformer(config, self.enc_concept_embedding, concept_vocab, relation_vocab)
        else:
            self.graph_encoder = None
        if config.use_adapter and word_rel_vocab is not None:
            self.adapter_enc = AdapterGraphTransformer(config, word_rel_vocab)
            self.adapter_norm = nn.LayerNorm(config.d_enc_sent)
        else:
            self.adapter_enc = None

        self.dec_word_embedding = self.enc_word_embedding
        self.position_encoder = PositionalEncoding(config.d_dec)
        dual_mode = getattr(config, "dual_mode", "cat")
      

        if word_vocab is not None:
            self.word_vocab_size = len(self.word_vocab)
            self.BOS = self.word_vocab["<bos>"]
            self.EOS = self.word_vocab["<eos>"]

        # self.projector = nn.Linear(config.d_dec, self.word_vocab_size)
        # if self.config.share_vocab:             # existing bugs to be fixed
        #     self.projector.weight = self.dec_word_embedding.weight
        if self.config.use_kl_loss:
            self.kl = nn.KLDivLoss(size_average=False)

        if self.config.rl_ratio > 0.0 and self.config.rl_type == "bertscore":
            self.rl_metric = nlp.load_metric("bertscore")

        self.class_num = 2 if config.binary else 3

        if config.ablation:
            if config.graph:
                self.cls_projector = nn.Linear(config.d_enc_concept, self.class_num)
            # self.cls_projector = nn.Linear(2 * config.d_dec, self.class_num)
            else:
                self.cls_projector = nn.Linear(config.d_dec, self.class_num)
        else:
            # self.cls_projector = nn.Linear(3 * config.d_dec, self.class_num)
            self.cls_projector = nn.Linear(2*config.d_dec, self.class_num)
        self.selector = BERTAttentionSelector(config)
        


    def decode_into_string(self, ids):
        try:
            i = ids.index(self.EOS)
            ids = ids[:i]
        except ValueError:
            pass
        if self.word_vocab is not None:
            return " ".join([self.word_encoder.vocab_inv[x] for x in ids])
        # else:
        #     return self.bpemb.decode_ids(ids)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def build_embedding(self, pretrain_emb, vocab, d_emb):
        print("bulid_embedding", len(vocab))
        freeze_emb = getattr(self.config, "freeze_emb", True)
        if pretrain_emb is not None:
            if pretrain_emb.shape[1] == d_emb:
                embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrain_emb).to(device), freeze=freeze_emb)
            else: 
                embedding = nn.Sequential(
                    nn.Embedding.from_pretrained(torch.from_numpy(pretrain_emb).to(device), freeze=freeze_emb),
                    nn.Linear(pretrain_emb.shape[1], d_emb),
                )
        else:
            embedding = nn.Embedding(len(vocab), d_emb)
        return embedding


    def forward(self, batch):
        # encoding
        # label = torch.tensor(batch["label"][0]).to(device) 
        label = batch["label"].to(device) 
        sent_mask_inv = (batch["src_mask"] == 0).to(device)  # [batch, seq]
        graph_mask_inv = (batch["con_mask"] == 0).to(device)

        sent_mem_ori = self.word_encoder(batch)   # [batch, dim]
        # res_mem = self.res_encoder(batch)
       
        graph_mem = self.graph_encoder(batch) if self.graph_encoder is not None else None

        # sent_fea = torch.mean(sent_mem, dim=1)
        # res_fea = torch.mean(res_mem, dim=1)
        graph_con_fea = torch.mean(graph_mem[0], dim=1)
        graph_res_fea = torch.mean(graph_mem[1], dim=1)
    
        if self.config.ablation:    
            #non bert
            if self.config.graph:
                if self.config.gate:
                    score = self.selector(sent_mem_ori)
                    sent_mem = sent_mem_ori * score 
                    graph_ori = torch.cat([graph_con_fea, graph_res_fea], dim=-1)  # [batch, 2 dim]
                    graph = graph_ori * (1 - score)
                    con_fea = graph
                else:
                    con_fea = torch.cat([graph_con_fea, graph_res_fea], dim=-1)
            else:
                # con_fea = torch.cat([sent_fea, res_fea], dim=-1)
                con_fea = sent_mem_ori

        else:
            # score = self.selector(torch.cat([graph_con_fea, graph_res_fea], dim=-1))  # [batch, 2 dim]
            graph = graph_con_fea + graph_res_fea
            sent_mem = sent_mem_ori
            if self.config.gate:
                score = self.selector(sent_mem_ori)
                # extended_score = score[:,:,None]  # [batch, seq, 1]
                sent_mem = sent_mem_ori * score 
                
                # graph_ori = torch.cat([graph_con_fea, graph_res_fea], dim=-1)  # [batch, 2 dim]
                graph = graph * (1 - score)
          
            con_fea = torch.cat([graph, sent_mem], dim=-1)

            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            denominator = 0
            for i in range(len(batch["label"])):
                if label[i] == 0:
                    # denominator += torch.exp(cos(graph_con_fea[i], graph_res_fea[i]))
                    denominator += torch.exp(cos(graph[i], sent_mem[i]))
                    # denominator += torch.exp(torch.dot(graph[i], sent_mem[i]))
                
            cnt_loss = 0
            for i in range(len(batch["label"])):
                if label[i] == 1:
                    # numerator = torch.exp(cos(graph_con_fea[i], graph_res_fea[i]))
                    numerator = torch.exp(cos(graph[i], sent_mem[i]))
                    # numerator = torch.exp(torch.dot(graph[i], sent_mem[i]))\
                    # print('numerator', numerator)
                    cnt_loss += numerator / denominator
            cnt_loss = -cnt_loss / len(batch["label"])
            
        cls_logits = self.cls_projector(con_fea)

        cls_pred = cls_logits.argmax(dim=-1)
        cls_loss = F.cross_entropy(
            cls_logits.contiguous().view(-1, self.class_num),
            label.contiguous().view(-1),
        )

        cls_train_right = (torch.tensor((cls_pred == label),dtype=float).to(device)).sum()
        cls_train_total = torch.tensor(label.shape, dtype=float).to(device)

        # print("cnt_loss", torch.tensor(cnt_loss, dtype=float))
        if self.config.ablation:  
            return {
            # "preds": preds,
            # "cnt_loss": torch.tensor(cnt_loss, dtype=float),
            "cls_loss": cls_loss,
            "counts": (cls_train_right, cls_train_total),
            "selected_kn": None,
            "trg_selected_kn": None,
        }  
        
        else:
            return {
                # "preds": preds,
                "cnt_loss": torch.tensor(cnt_loss, dtype=float),
                "cls_loss": cls_loss,
                "counts": (cls_train_right, cls_train_total),
                "selected_kn": None,
                "trg_selected_kn": None,
            }
    # else:

class DualTransformer(nn.Module):
    
    def __init__(self, config, word_emb, con_emb=None, word_vocab=None, concept_vocab=None, relation_vocab=None, word_rel_vocab=None):
        super(DualTransformer, self).__init__()

        self.config = config
        self.word_vocab = word_vocab
        self.concept_vocab = concept_vocab
        self.relation_vocab = relation_vocab

        self.enc_word_embedding = self.build_embedding(word_emb, word_vocab, self.config.d_enc_sent)

        self.word_encoder = SentTransformer(config, self.enc_word_embedding, word_vocab)
        # self.res_encoder = ResTransformer(config, self.enc_word_embedding, word_vocab)
        # self.word_encoder = BERTEncoder(config)

        if config.dual_enc and self.concept_vocab is not None and relation_vocab is not None:
            if config.share_con_vocab:
                self.enc_concept_embedding = self.enc_word_embedding
            else:
                self.enc_concept_embedding = self.build_embedding(con_emb, concept_vocab, self.config.d_enc_concept)
            
            self.graph_encoder = GraphTransformer(config, self.enc_concept_embedding, concept_vocab, relation_vocab)
        else:
            self.graph_encoder = None
        # if config.use_adapter and word_rel_vocab is not None:
        #     self.adapter_enc = AdapterGraphTransformer(config, word_rel_vocab)
        #     self.adapter_norm = nn.LayerNorm(config.d_enc_sent)
        # else:
        #     self.adapter_enc = None

        self.dec_word_embedding = self.enc_word_embedding
        self.position_encoder = PositionalEncoding(config.d_dec)
        dual_mode = getattr(config, "dual_mode", "cat")
       
        if word_vocab is not None:
            self.word_vocab_size = len(self.word_vocab)
            self.BOS = self.word_vocab["<bos>"]
            self.EOS = self.word_vocab["<eos>"]

        self.projector = nn.Linear(config.d_dec, self.word_vocab_size)
        if self.config.share_vocab:             # existing bugs to be fixed
            self.projector.weight = self.dec_word_embedding.weight
        if self.config.use_kl_loss:
            self.kl = nn.KLDivLoss(size_average=False)

        if self.config.rl_ratio > 0.0 and self.config.rl_type == "bertscore":
            self.rl_metric = nlp.load_metric("bertscore")
        
        

        self.class_num = 2 if config.binary else 3

        if config.ablation:
            if config.graph:
                self.cls_projector = nn.Linear(2*config.d_enc_concept, self.class_num)
            else:
                self.cls_projector = nn.Linear(2*config.d_dec, self.class_num)
        else:
            self.cls_projector = nn.Linear(4*config.d_dec, self.class_num)
            # self.cls_projector = nn.Linear(2 * config.d_dec, self.class_num)
            # self.cls_projector = nn.Linear(config.d_dec + config.d_enc_concept, self.class_num)
        
        self.selector = AttentionSelector(config)


    def decode_into_string(self, ids):
        try:
            i = ids.index(self.EOS)
            ids = ids[:i]
        except ValueError:
            pass
        if self.word_vocab is not None:
            return " ".join([self.word_encoder.vocab_inv[x] for x in ids])
        # else:
        #     return self.bpemb.decode_ids(ids)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def build_embedding(self, pretrain_emb, vocab, d_emb):
        print("bulid_embedding", len(vocab))
        freeze_emb = getattr(self.config, "freeze_emb", True)
        if pretrain_emb is not None:
            if pretrain_emb.shape[1] == d_emb:
                embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrain_emb).to(device), freeze=freeze_emb)
            else: 
                embedding = nn.Sequential(
                    nn.Embedding.from_pretrained(torch.from_numpy(pretrain_emb).to(device), freeze=freeze_emb),
                    nn.Linear(pretrain_emb.shape[1], d_emb),
                )
        else:
            embedding = nn.Embedding(len(vocab), d_emb)
        return embedding


    def forward(self, batch):
        # encoding
        # label = torch.tensor(batch["label"][0]).to(device) 
        label = batch["label"].to(device) 
        sent_mask_inv = (batch["src_mask"] == 0).to(device)  # [batch, seq]
        graph_mask_inv = (batch["con_mask"] == 0).to(device)

        sent_mem_ori = self.word_encoder(batch)
        # res_mem = self.res_encoder(batch)
        # if self.adapter_enc is not None:
        #     sent_mem_new = self.adapter_enc(sent_mem, batch)
        #     sent_mem = self.adapter_norm(sent_mem_new + sent_mem)
        graph_mem = self.graph_encoder(batch) if self.graph_encoder is not None else None

        # sent_fea = torch.mean(sent_mem, dim=1)
        # res_fea = torch.mean(res_mem, dim=1)
        graph_con_fea = torch.mean(graph_mem[0], dim=1)
        graph_res_fea = torch.mean(graph_mem[1], dim=1)
    
        if self.config.ablation:    
            #non bert
            if self.config.graph:
                if self.config.gate:
                    score = self.selector(sent_mem_ori)
                    sent_mem = sent_mem_ori * score 
                    graph_ori = torch.cat([graph_con_fea, graph_res_fea], dim=-1)  # [batch, 2 dim]
                    graph = graph_ori * (1 - score)
                    con_fea = graph
                else:
                    # print('ablation gate graph')
                    con_fea = torch.cat([graph_con_fea, graph_res_fea], dim=-1)
            else:
                # con_fea = torch.cat([sent_fea, res_fea], dim=-1)
                # print('ablation sent')
                con_fea = sent_mem_ori
        else:
            # score = self.selector(torch.cat([graph_con_fea, graph_res_fea], dim=-1))  # [batch, 2 dim]
            graph = torch.cat([graph_con_fea, graph_res_fea], dim=-1)
            sent_mem = sent_mem_ori
            if self.config.gate:
                score = self.selector(sent_mem_ori)
                # extended_score = score[:,:,None]  # [batch, seq, 1]
                sent_mem = sent_mem_ori * score 
                
                graph_ori = torch.cat([graph_con_fea, graph_res_fea], dim=-1)  # [batch, 2 dim]
                graph = graph_ori * (1 - score)
          
            con_fea = torch.cat([graph, sent_mem], dim=-1)

            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            denominator = 0
            
            # print(len(batch["label"]))
            for i in range(len(batch["label"])):
                if label[i] == 0:
                    # denominator += torch.exp(cos(graph_con_fea[i], graph_res_fea[i]))
                    denominator += torch.exp(cos(graph[i], sent_mem[i]))
                    # denominator += torch.exp(torch.dot(graph[i], sent_mem[i]))
                
        
            cnt_loss = 0
            for i in range(len(batch["label"])):
                if label[i] == 1:
                    # numerator = torch.exp(cos(graph_con_fea[i], graph_res_fea[i]))
                    numerator = torch.exp(cos(graph[i], sent_mem[i]))
                    # numerator = torch.exp(torch.dot(graph[i], sent_mem[i]))
                    cnt_loss += numerator / denominator
            cnt_loss = -cnt_loss / len(batch["label"])

        cls_logits = self.cls_projector(con_fea)

        cls_pred = cls_logits.argmax(dim=-1)
        cls_loss = F.cross_entropy(
            cls_logits.contiguous().view(-1, self.class_num),
            label.contiguous().view(-1),
            # ignore_index=0,
        )

        cls_train_right = (torch.tensor((cls_pred == label),dtype=float).to(device)).sum()
        cls_train_total = torch.tensor(label.shape, dtype=float).to(device)

        # softmax = torch.softmax(dim=0)
        if self.config.ablation:  
            return {
            # "preds": preds,
            # "cls_logits": softmax(cls_logits),
            "cls_logits": nn.functional.softmax(cls_logits, dim=1),
            "cls_loss": cls_loss,
            "counts": (cls_train_right, cls_train_total),
            "selected_kn": None,
            "trg_selected_kn": None,
        }  

        else:
            return {
                "cls_logits": nn.functional.softmax(cls_logits, dim=1),
                "cnt_loss": torch.tensor(cnt_loss, dtype=float),
                "cls_loss": cls_loss,
                "counts": (cls_train_right, cls_train_total),
                "selected_kn": None,
                "trg_selected_kn": None,
            }
    # else:
    #     return {
    #         "sent_memory_emb": sent_mem,
    #         "sent_memory_mask": batch["src_mask"],
    #         "graph_memory_emb": graph_mem,
    #         "graph_memory_mask": batch["con_mask"],
    #         "selected_kn": None,
    #         "trg_selected_kn": None,
    #     }

class bert(nn.Module):
    
    def __init__(self, config, word_emb, con_emb=None, word_vocab=None, concept_vocab=None, relation_vocab=None, word_rel_vocab=None):
        super(bert, self).__init__()

        self.config = config
        self.word_vocab = word_vocab
        self.concept_vocab = concept_vocab
        self.relation_vocab = relation_vocab

        self.enc_word_embedding = self.build_embedding(word_emb, word_vocab, self.config.d_enc_sent)

        
        # self.word_encoder = SentTransformer(config, self.enc_word_embedding, word_vocab)
        # self.res_encoder = ResTransformer(config, self.enc_word_embedding, word_vocab)
        self.word_encoder = BERTEncoder(config)

        if config.dual_enc and self.concept_vocab is not None and relation_vocab is not None:
            if config.share_con_vocab:
                self.enc_concept_embedding = self.enc_word_embedding
            else:
                self.enc_concept_embedding = self.build_embedding(con_emb, concept_vocab, self.config.d_enc_concept)
            
            self.graph_encoder = GraphTransformer(config, self.enc_concept_embedding, concept_vocab, relation_vocab)
        else:
            self.graph_encoder = None
        if config.use_adapter and word_rel_vocab is not None:
            self.adapter_enc = AdapterGraphTransformer(config, word_rel_vocab)
            self.adapter_norm = nn.LayerNorm(config.d_enc_sent)
        else:
            self.adapter_enc = None

        self.dec_word_embedding = self.enc_word_embedding
        self.position_encoder = PositionalEncoding(config.d_dec)
        dual_mode = getattr(config, "dual_mode", "cat")
      

        if word_vocab is not None:
            self.word_vocab_size = len(self.word_vocab)
            self.BOS = self.word_vocab["<bos>"]
            self.EOS = self.word_vocab["<eos>"]

        # self.projector = nn.Linear(config.d_dec, self.word_vocab_size)
        # if self.config.share_vocab:             # existing bugs to be fixed
        #     self.projector.weight = self.dec_word_embedding.weight
        if self.config.use_kl_loss:
            self.kl = nn.KLDivLoss(size_average=False)

        if self.config.rl_ratio > 0.0 and self.config.rl_type == "bertscore":
            self.rl_metric = nlp.load_metric("bertscore")

        self.class_num = 2 if config.binary else 3

        if config.ablation:
            if config.graph:
                self.cls_projector = nn.Linear(config.d_enc_concept, self.class_num)
            # self.cls_projector = nn.Linear(2 * config.d_dec, self.class_num)
            else:
                self.cls_projector = nn.Linear(config.d_dec, self.class_num)
        else:
            # self.cls_projector = nn.Linear(3 * config.d_dec, self.class_num)
            self.cls_projector = nn.Linear(2*config.d_dec, self.class_num)
        self.selector = BERTAttentionSelector(config)
        


    def decode_into_string(self, ids):
        try:
            i = ids.index(self.EOS)
            ids = ids[:i]
        except ValueError:
            pass
        if self.word_vocab is not None:
            return " ".join([self.word_encoder.vocab_inv[x] for x in ids])
        # else:
        #     return self.bpemb.decode_ids(ids)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def build_embedding(self, pretrain_emb, vocab, d_emb):
        print("bulid_embedding", len(vocab))
        freeze_emb = getattr(self.config, "freeze_emb", True)
        if pretrain_emb is not None:
            if pretrain_emb.shape[1] == d_emb:
                embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrain_emb).to(device), freeze=freeze_emb)
            else: 
                embedding = nn.Sequential(
                    nn.Embedding.from_pretrained(torch.from_numpy(pretrain_emb).to(device), freeze=freeze_emb),
                    nn.Linear(pretrain_emb.shape[1], d_emb),
                )
        else:
            embedding = nn.Embedding(len(vocab), d_emb)
        return embedding


    def forward(self, batch):
        # encoding
        # label = torch.tensor(batch["label"][0]).to(device) 
        label = batch["label"].to(device) 
        sent_mask_inv = (batch["src_mask"] == 0).to(device)  # [batch, seq]
        graph_mask_inv = (batch["con_mask"] == 0).to(device)

        sent_mem_ori = self.word_encoder(batch)   # [batch, dim]
        # res_mem = self.res_encoder(batch)
       
        graph_mem = self.graph_encoder(batch) if self.graph_encoder is not None else None

        # sent_fea = torch.mean(sent_mem, dim=1)
        # res_fea = torch.mean(res_mem, dim=1)
        graph_con_fea = torch.mean(graph_mem[0], dim=1)
        graph_res_fea = torch.mean(graph_mem[1], dim=1)
    
        if self.config.ablation:    
            #non bert
            if self.config.graph:
                if self.config.gate:
                    score = self.selector(sent_mem_ori)
                    sent_mem = sent_mem_ori * score 
                    graph_ori = torch.cat([graph_con_fea, graph_res_fea], dim=-1)  # [batch, 2 dim]
                    graph = graph_ori * (1 - score)
                    con_fea = graph
                else:
                    con_fea = torch.cat([graph_con_fea, graph_res_fea], dim=-1)
            else:
                # con_fea = torch.cat([sent_fea, res_fea], dim=-1)
                con_fea = sent_mem_ori

        else:
            # score = self.selector(torch.cat([graph_con_fea, graph_res_fea], dim=-1))  # [batch, 2 dim]
            graph = graph_con_fea + graph_res_fea
            sent_mem = sent_mem_ori
            if self.config.gate:
                score = self.selector(sent_mem_ori)
                # extended_score = score[:,:,None]  # [batch, seq, 1]
                sent_mem = sent_mem_ori * score 
                
                # graph_ori = torch.cat([graph_con_fea, graph_res_fea], dim=-1)  # [batch, 2 dim]
                graph = graph * (1 - score)
          
            con_fea = torch.cat([graph, sent_mem], dim=-1)

            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            denominator = 0
            for i in range(len(batch["label"])):
                if label[i] == 0:
                    # denominator += torch.exp(cos(graph_con_fea[i], graph_res_fea[i]))
                    denominator += torch.exp(cos(graph[i], sent_mem[i]))
                    # denominator += torch.exp(torch.dot(graph[i], sent_mem[i]))
                
            cnt_loss = 0
            for i in range(len(batch["label"])):
                if label[i] == 1:
                    # numerator = torch.exp(cos(graph_con_fea[i], graph_res_fea[i]))
                    numerator = torch.exp(cos(graph[i], sent_mem[i]))
                    # numerator = torch.exp(torch.dot(graph[i], sent_mem[i]))\
                    # print('numerator', numerator)
                    cnt_loss += numerator / denominator
            cnt_loss = -cnt_loss / len(batch["label"])
            
        cls_logits = self.cls_projector(con_fea)

        cls_pred = cls_logits.argmax(dim=-1)
        cls_loss = F.cross_entropy(
            cls_logits.contiguous().view(-1, self.class_num),
            label.contiguous().view(-1),
        )

        cls_train_right = (torch.tensor((cls_pred == label),dtype=float).to(device)).sum()
        cls_train_total = torch.tensor(label.shape, dtype=float).to(device)

        # print("cnt_loss", torch.tensor(cnt_loss, dtype=float))
        if self.config.ablation:  
            return {
            # "preds": preds,
            # "cnt_loss": torch.tensor(cnt_loss, dtype=float),
            "cls_loss": cls_loss,
            "counts": (cls_train_right, cls_train_total),
            "selected_kn": None,
            "trg_selected_kn": None,
        }  
        
        else:
            return {
                # "preds": preds,
                "cnt_loss": torch.tensor(cnt_loss, dtype=float),
                "cls_loss": cls_loss,
                "counts": (cls_train_right, cls_train_total),
                "selected_kn": None,
                "trg_selected_kn": None,
            }
    # else:
