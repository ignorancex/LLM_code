import numpy as np
import torch
# from pytorch_pretrained_bert import BertTokenizer, BertModel
from transformers import BertTokenizer, BertModel, logging
from torchvision import models
from torch import nn
import fasttext as ft
import os
import pdb
import langid
from torch.nn import MultiheadAttention
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.set_verbosity_warning()
logging.set_verbosity_error()




# 特征整合
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, tokens, input_ids, input_mask, input_type_ids):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


# BERT的分词处理
def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example)

        if len(tokens_a) > seq_length - 2:
            tokens_a = tokens_a[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        features.append(
            InputFeatures(
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


# BERT的词向量提取过程
def bert_feature(examples, model, tokenizer, seq_length=64):
    features = convert_examples_to_features(
        examples=examples, seq_length=seq_length, tokenizer=tokenizer)

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(device)
    outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)
    pooled_output = outputs[1]

    return pooled_output


def ft_feature(examples, model):
    ft_list = []
    for t in examples:
        vec = model.get_word_vector(t)
        ft_list.append(vec)
    out = torch.tensor(ft_list).to(device)
    return out

class youtube_MLP(nn.Module):
    def __init__(self, seq_length, batch_size):
        super(youtube_MLP, self).__init__()
        self.text_num = 5
        self.meta_num = 6

        self.batch_size = batch_size
        self.img_feature = nn.Sequential(*list(models.resnet101(pretrained=True).children())[:-1])

        # embedding vocabulary
        self.cate_vocab = {"People  Blogs": 1,
                           "Gaming": 2,
                           "News  Politics": 3,
                           "Entertainment": 4,
                           "Music": 5,
                           "Education": 6,
                           "Sports": 7,
                           "Howto  Style": 8,
                           "Film  Animation": 9,
                           "Nonprofits  Activism": 10,
                           "Travel": 11,
                           "Comedy": 12,
                           "Science  Technology": 13,
                           "Autos  Vehicles": 14,
                           "Pets  Animals": 15,
                           "OOA": 0,
                           }
        self.lang_vocab = {"en": 1,
                           "zh": 2,
                           "ko": 3,
                           "ja": 4,
                           "hi": 5,
                           "ru": 6,
                           "OOA": 0,
                           }

        # BERT-Multilingual
        self.tokenizer = BertTokenizer.from_pretrained('../bert_multilingual')
        self.bert_model = BertModel.from_pretrained('../bert_multilingual').to(device)

        self.conv = nn.Conv2d(self.text_num, 1, 1)
        self.conv.weight.data.normal_(1 / self.text_num, 0.01)

        # visual_MLP
        self.img_MLP = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024, affine=False),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        # embeddings & MLP
        self.cate_embedding = nn.Sequential(
            nn.Embedding(16, 128),
        )
        self.lang_embedding = nn.Sequential(
            nn.Embedding(7, 128),
        )
        self.emb_MLP = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256, affine=False),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # textual MLP
        self.text_MLP = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
        self.meta_MLP = nn.Sequential(
            nn.Linear(self.meta_num, 128),
            nn.BatchNorm1d(128, affine=False),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        self.fusion_MLP = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, affine=False),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128, affine=False),
            nn.ReLU(),
            nn.Linear(128, seq_length),
        )

    def forward(self, img, texts, meta, cat):
        # '''======================= visual features =========================='''
        with torch.no_grad():
            img_features = self.img_feature(img).squeeze()
        img_features = self.img_MLP(img_features)

        # '''======================= embedding features =========================='''

        cate_voc = []
        lang_voc = []
        for c in cat[0]:
            if c in self.cate_vocab.keys():
                cate_voc.append(self.cate_vocab[c])
            else:
                cate_voc.append(self.cate_vocab["OOA"])

        for c in cat[1]:
            lang = langid.classify(c)[0]
            if lang in self.lang_vocab.keys():
                lang_voc.append(self.lang_vocab[lang])
            else:
                lang_voc.append(self.lang_vocab["OOA"])

        cate_voc = torch.LongTensor(cate_voc).to(device)
        lang_voc = torch.LongTensor(lang_voc).to(device)
        cate_feature = self.cate_embedding(cate_voc)
        lang_feature = self.lang_embedding(lang_voc)
        embedding_features = torch.cat([cate_feature, lang_feature], 1)
        embedding_features = self.emb_MLP(embedding_features)

        # '''======================= textual features =========================='''

        text_features_list = []
        for text in texts:
            with torch.no_grad():
                # bert feature
                bert_features = bert_feature(text, self.bert_model, self.tokenizer)
            text_features = bert_features

            text_features_list.append(text_features)
        text_features = torch.stack(text_features_list, 1).unsqueeze(3)
        text_features = self.conv(text_features).permute(0, 2, 1, 3).squeeze()
        text_features = self.text_MLP(text_features)

        #     text_features_list.append(self.text_MLP(text_features))
        # text_features = torch.stack(text_features_list, 1).unsqueeze(3)
        # text_features = self.conv(text_features).permute(0, 2, 1, 3).squeeze()

        # '''======================= numerical features =========================='''
        meta_features = self.meta_MLP(meta)

        # '''======================= feature fusion=========================='''
        # feature_vector = torch.cat([img_features, text_features, meta_features], 1)
        feature_vector = torch.cat([img_features, embedding_features, text_features, meta_features], 1)
        out = self.fusion_MLP(feature_vector)
        return out, []


class youtube_lstm3(nn.Module):
    def __init__(self, seq_length, batch_size):
        super(youtube_lstm3, self).__init__()
        self.text_num = 5
        self.meta_num = 6

        self.batch_size = batch_size
        self.img_feature = nn.Sequential(*list(models.resnet101(pretrained=True).children())[:-1])

        # embedding vocabulary
        self.cate_vocab = {"People  Blogs": 1,
                           "Gaming": 2,
                           "News  Politics": 3,
                           "Entertainment": 4,
                           "Music": 5,
                           "Education": 6,
                           "Sports": 7,
                           "Howto  Style": 8,
                           "Film  Animation": 9,
                           "Nonprofits  Activism": 10,
                           "Travel": 11,
                           "Comedy": 12,
                           "Science  Technology": 13,
                           "Autos  Vehicles": 14,
                           "Pets  Animals": 15,
                           "OOA": 0,
                           }
        self.lang_vocab = {"en": 1,
                           "zh": 2,
                           "ko": 3,
                           "ja": 4,
                           "hi": 5,
                           "ru": 6,
                           "OOA": 0,
                           }

        # BERT
        self.tokenizer = BertTokenizer.from_pretrained('../bert_multilingual')
        self.bert_model = BertModel.from_pretrained('../bert_multilingual').to(device)

        self.conv = nn.Conv2d(self.text_num, 1, 1)
        self.conv.weight.data.normal_(1 / self.text_num, 0.01)

        # visual_MLP
        self.img_MLP = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024, affine=False),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        # embeddings & MLP
        self.cate_embedding = nn.Sequential(
            nn.Embedding(16, 128),
        )
        self.lang_embedding = nn.Sequential(
            nn.Embedding(7, 128),
        )
        self.emb_MLP = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, affine=False),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.lang_MLP = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, affine=False),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.cate_MLP = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, affine=False),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        # textual MLP
        self.text_MLP = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
        self.meta_MLP = nn.Sequential(
            nn.Linear(self.meta_num, 128),
            nn.BatchNorm1d(128, affine=False),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # all_feature_length and feature_vector_length
        all_f_len = 4*128
        self.vector_len = 128
        # self.v_MLP = nn.Sequential(
        #     nn.Linear(all_f_len, 1024),
        #     nn.BatchNorm1d(1024, affine=False),
        #     nn.ReLU(),
        #     nn.Linear(1024, self.vector_len),
        # )

        # MLP for h & c initialization
        self.hc_MLP = nn.Sequential(
            nn.Linear(all_f_len, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(),
            nn.Linear(512, self.vector_len),
        )

        self.proj_img = nn.Linear(128, 512)
        self.proj_embedding = nn.Linear(128, 512)
        self.proj_text = nn.Linear(128, 512)
        self.proj_meta = nn.Linear(128, 512)

        self.seq_length = seq_length

        # output_MLP
        self.MLPo_list = nn.ModuleList([nn.Sequential(
            nn.Linear(self.vector_len*2, self.vector_len),
            nn.BatchNorm1d(self.vector_len, affine=False),
            nn.ReLU(),
            nn.Linear(self.vector_len, self.vector_len//2),
            nn.BatchNorm1d(self.vector_len//2, affine=False),
            nn.ReLU(),
            nn.Linear(self.vector_len//2, 1),
        )for _ in range(self.seq_length)])

        # The MLP for every time LSTM's x
        self.x_MLP_list = nn.ModuleList([nn.Sequential(
            nn.Linear(all_f_len, 768),
            nn.BatchNorm1d(768, affine=False),
            nn.ReLU(),
            nn.Linear(768, self.vector_len),
        )for _ in range(self.seq_length)])

        # self.x_MLP_abl = nn.Sequential(
        #     nn.Linear(all_f_len, 128),
        # )
        self.dropout=nn.Dropout(p=0.1)
        self.LSTMCell_list = nn.ModuleList([nn.LSTMCell(input_size=self.vector_len, hidden_size=self.vector_len)
                                            for _ in range(self.seq_length)])

    def forward(self, img, texts, meta, cat):
        # '''======================= visual features =========================='''
        with torch.no_grad():
            img_features = self.img_feature(img).squeeze()
        img_features = self.img_MLP(img_features)
        #img_features = self.dropout(img_features)
        #img_features = self.proj_img(img_features)

        # '''======================= embedding features =========================='''

        cate_voc = []
        lang_voc = []
        for c in cat[0]:
            if c in self.cate_vocab.keys():
                cate_voc.append(self.cate_vocab[c])
            else:
                cate_voc.append(self.cate_vocab["OOA"])

        for c in cat[1]:
            lang = langid.classify(c)[0]
            if lang in self.lang_vocab.keys():
                lang_voc.append(self.lang_vocab[lang])
            else:
                lang_voc.append(self.lang_vocab["OOA"])

        cate_voc = torch.LongTensor(cate_voc).to(device)
        lang_voc = torch.LongTensor(lang_voc).to(device)
        cate_feature = self.cate_embedding(cate_voc)
        lang_feature = self.lang_embedding(lang_voc)
        cate_feature = self.cate_MLP(cate_feature)
        lang_feature = self.lang_MLP(lang_feature)

        embedding_features = cate_feature * lang_feature
        #embedding_features = torch.cat([cate_feature,lang_feature],dim=1)


        embedding_features = self.emb_MLP(embedding_features)
        #embedding_features = self.proj_embedding(embedding_features)

        # '''======================= textual features =========================='''

        text_features_list = []
        for text in texts:
            with torch.no_grad():
                # bert feature
                bert_features = bert_feature(text, self.bert_model, self.tokenizer)
            text_features = bert_features

            text_features_list.append(text_features)
        text_features = torch.stack(text_features_list, 1).unsqueeze(3)
        text_features = self.conv(text_features).permute(0, 2, 1, 3).squeeze()
        text_features = self.text_MLP(text_features)
        #text_features = self.dropout(text_features)
        #text_features = self.proj_text(text_features)

        #     text_features_list.append(self.text_MLP(text_features))
        # text_features = torch.stack(text_features_list, 1).unsqueeze(3)
        # text_features = self.conv(text_features).permute(0, 2, 1, 3).squeeze()

        # '''======================= numerical features =========================='''
        meta_features = self.meta_MLP(meta)
        #meta_features = self.proj_meta(meta_features)
        
        # Ensure all features have the same shape
        # img_features = img_features.view(self.batch_size, 1, -1)
        # embedding_features = embedding_features.view(self.batch_size, 1, -1)
        # text_features = text_features.view(self.batch_size, 1, -1)
        # meta_features = meta_features.view(self.batch_size, 1, -1)

        # '''======================= feature fusion=========================='''
        # feature_vector = torch.cat([img_features, text_features, meta_features], 1)
        # 将所有特征在时间维度上拼接
        feature_vector = torch.cat([img_features,embedding_features,text_features,meta_features], dim=1)  # shape: (batch_size, 4, seq_length, 128)


        # '''======================= h & c initial setting =========================='''
        # h = torch.zeros(self.batch_size, self.vector_len).to(device)
        # c = torch.zeros(self.batch_size, self.vector_len).to(device)
        h = self.hc_MLP(feature_vector)
        c = self.hc_MLP(feature_vector)
        out_f = torch.empty(self.batch_size, self.seq_length, self.vector_len*2).to(device)
        out = torch.empty(self.batch_size, self.seq_length).to(device)
        for i in range(self.seq_length):
            # x_vector = self.x_MLP_abl(feature_vector)
            x_vector = self.x_MLP_list[i](feature_vector)
            h, c = self.LSTMCell_list[i](x_vector, (h, c))
            # x = self.x_MLP(h)
            s = torch.cat([h, c], 1)
            o_i = self.MLPo_list[i](s)
            # print(o_i.size())
            out_f[:, i, :] = s
            out[:, i] = o_i.squeeze(1)

        return out, out_f

class youtube_cb(nn.Module):
    def __init__(self, seq_length, batch_size):
        super(youtube_cb, self).__init__()
        self.text_num = 5
        self.meta_num = 9

        self.batch_size = batch_size
        self.img_feature = nn.Sequential(*list(models.resnet101(pretrained=True).children())[:-1])

        # embedding vocabulary
        self.cate_vocab = {"People  Blogs": 1,
                           "Gaming": 2,
                           "News  Politics": 3,
                           "Entertainment": 4,
                           "Music": 5,
                           "Education": 6,
                           "Sports": 7,
                           "Howto  Style": 8,
                           "Film  Animation": 9,
                           "Nonprofits  Activism": 10,
                           "Travel": 11,
                           "Comedy": 12,
                           "Science  Technology": 13,
                           "Autos  Vehicles": 14,
                           "Pets  Animals": 15,
                           "OOA": 0,
                           }
        self.lang_vocab = {"en": 1,
                           "zh": 2,
                           "ko": 3,
                           "ja": 4,
                           "hi": 5,
                           "ru": 6,
                           "OOA": 0,
                           }

        # BERT-Multilingual
        self.tokenizer = BertTokenizer.from_pretrained('../bert_multilingual')
        self.bert_model = BertModel.from_pretrained('../bert_multilingual').to(device)

        self.conv = nn.Conv2d(self.text_num, 1, 1)
        self.conv.weight.data.normal_(1 / self.text_num, 0.01)

        # visual_MLP
        self.img_MLP = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024, affine=False),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        # embeddings & MLP
        self.cate_embedding = nn.Sequential(
            nn.Embedding(16, 128),
        )
        self.lang_embedding = nn.Sequential(
            nn.Embedding(7, 128),
        )
        self.emb_MLP = nn.Sequential(
            nn.Linear(256,256),
            nn.BatchNorm1d(256, affine=False),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # textual MLP
        self.text_MLP = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
        self.meta_MLP = nn.Sequential(
            nn.Linear(self.meta_num, 128),
            nn.BatchNorm1d(128, affine=False),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # all_feature_length and feature_vector_length
        all_f_len = 4 * 128
        self.vector_len = 128

        # MLP for h & c initialization
        self.hc_MLP = nn.Sequential(
            nn.Linear(all_f_len, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(),
            nn.Linear(512, self.vector_len),
        )

        self.seq_length = seq_length

        # output_MLP
        self.MLPo = nn.Sequential(
            nn.Linear(self.vector_len*2, self.vector_len),
            nn.BatchNorm1d(self.vector_len, affine=False),
            nn.ReLU(),
            nn.Linear(self.vector_len, self.vector_len//2),
            nn.BatchNorm1d(self.vector_len//2, affine=False),
            nn.ReLU(),
            nn.Linear(self.vector_len//2, 1),
        )

        # The MLP for every time LSTM's x
        self.x_MLP_list = nn.ModuleList([nn.Sequential(
            nn.Linear(all_f_len, 768),
            nn.BatchNorm1d(768, affine=False),
            nn.ReLU(),
            nn.Linear(768, self.vector_len),
        )for _ in range(self.seq_length)])

        self.x_MLP = nn.Sequential(
            nn.Linear(all_f_len, 768),
            nn.BatchNorm1d(768, affine=False),
            nn.ReLU(),
            nn.Linear(768, self.vector_len),
        )

        self.LSTMCell_list = nn.ModuleList([nn.LSTMCell(input_size=self.vector_len, hidden_size=self.vector_len)
                                            for _ in range(self.seq_length)])

    def forward(self, img, texts, meta, cat):
        # '''======================= visual features =========================='''
        with torch.no_grad():
            img_features = self.img_feature(img).squeeze()
        img_features = self.img_MLP(img_features)

        # '''======================= embedding features =========================='''

        cate_voc = []
        lang_voc = []
        for c in cat[0]:
            if c in self.cate_vocab.keys():
                cate_voc.append(self.cate_vocab[c])
            else:
                cate_voc.append(self.cate_vocab["OOA"])

        for c in cat[1]:
            lang = langid.classify(c)[0]
            if lang in self.lang_vocab.keys():
                lang_voc.append(self.lang_vocab[lang])
            else:
                lang_voc.append(self.lang_vocab["OOA"])

        cate_voc = torch.LongTensor(cate_voc).to(device)
        lang_voc = torch.LongTensor(lang_voc).to(device)
        cate_feature = self.cate_embedding(cate_voc)
        lang_feature = self.lang_embedding(lang_voc)
        embedding_features = torch.cat([cate_feature, lang_feature], 1)
        #embedding_features = lang_feature
        embedding_features = self.emb_MLP(embedding_features)

        # '''======================= textual features =========================='''

        text_features_list = []
        for text in texts:
            with torch.no_grad():
                # bert feature
                bert_features = bert_feature(text, self.bert_model, self.tokenizer)
            text_features = bert_features

            text_features_list.append(text_features)
        text_features = torch.stack(text_features_list, 1).unsqueeze(3)
        text_features = self.conv(text_features).permute(0, 2, 1, 3).squeeze()
        text_features = self.text_MLP(text_features)

        #     text_features_list.append(self.text_MLP(text_features))
        # text_features = torch.stack(text_features_list, 1).unsqueeze(3)
        # text_features = self.conv(text_features).permute(0, 2, 1, 3).squeeze()

        # '''======================= numerical features =========================='''
        meta_features = self.meta_MLP(meta)

        # '''======================= feature fusion=========================='''
        # feature_vector = torch.cat([img_features, text_features, meta_features], 1)
        feature_vector = torch.cat([img_features, embedding_features, text_features, meta_features], 1)

        # 特征融合能否针对分类有改进
        # 预测30天的峰值，针对每天的流行度/峰值

        # '''======================= h & c initial setting =========================='''
        # h = torch.zeros(self.batch_size, self.vector_len).to(device)
        # c = torch.zeros(self.batch_size, self.vector_len).to(device)
        h = self.hc_MLP(feature_vector)
        c = self.hc_MLP(feature_vector)
        out_f = torch.empty(self.batch_size, self.seq_length, self.vector_len*2).to(device)
        out = torch.empty(self.batch_size, self.seq_length).to(device)
        for i in range(self.seq_length):
            # x_vector = feature_vector
            # x_vector = self.x_MLP_list[i](feature_vector)
            x_vector = self.x_MLP(feature_vector)
            h, c = self.LSTMCell_list[i](x_vector, (h, c))
            # x = self.x_MLP(h)
            s = torch.cat([h, c], 1)
            o_i = self.MLPo(s)
            # print(o_i.size())
            out_f[:, i, :] = s
            out[:, i] = o_i.squeeze(1)
        return out, out_f


