import torch
import utils
import numpy as np
from transformers import BertTokenizer, AutoModel, AutoTokenizer
from transformers import BertModel


class Dataset(torch.utils.data.Dataset):
    def __init__(self, sen_data):
        self.dataset = sen_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]
        return text

class Model(torch.nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert_model = bert_model

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = self.bert_model(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids)

        out = out.last_hidden_state.cpu().numpy()
        return out

def refresh_word_embeddings(input_ids, outs, stat_emb, target_embeds, keyword, token, sen_emb=False):
    for i, ids in enumerate(input_ids):
        last_word = ''
        last_word_v = []
        targets_in_sen = []
        for j, id in enumerate(ids):
            mor = token.decode([id])
            if mor.startswith("##"):
                last_word = last_word+mor[2:]
                last_word_v.append(outs[i][j])
            else:
                if j > 0:
                    ems = np.array(last_word_v)
                    avg = np.mean(ems, axis=0)
                    if last_word in stat_emb:
                        stat_emb[last_word].append(avg)
                    else:
                        stat_emb[last_word] = [avg]
                    if keyword == last_word:
                        targets_in_sen.append(avg)

                last_word = mor
                last_word_v = [outs[i][j]]
        if sen_emb:
            sen_embs = []
            for j, id in enumerate(ids):
                sen_embs.append(outs[i][j])
            cloud_point = np.mean(np.array(sen_embs), axis=0)
            target_embeds.append(cloud_point)
            continue
        # make sure each sentence only count on cloud point for target keyword,
        # and assume keyword appears in the same sentence always has same meanings.
        if len(targets_in_sen) > 0:
            cloud_point = np.mean(np.array(targets_in_sen), axis=0)
            target_embeds.append(cloud_point)

    return stat_emb


def avg_stat_emb(stat_emb):
    emb_threshold = 1
    del_keys = []
    for key in stat_emb:
        if len(stat_emb[key]) < emb_threshold:
            del_keys.append(key)
    for key in del_keys:
        del stat_emb[key]
    for key in stat_emb:
        ems = np.array(stat_emb[key])
        avg = np.mean(ems, axis=0)
        stat_emb[key] = avg
    return stat_emb


def bert_cloud(keyword, sendata, prbert_model, sen_emb=False):
    token = BertTokenizer.from_pretrained(prbert_model)
    dataset = Dataset(sendata)

    def collate_fn(data_raw):
        data = token.batch_encode_plus(batch_text_or_text_pairs=data_raw,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=180,
                                       return_tensors='pt',
                                       return_length=True)
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        token_type_ids = data['token_type_ids']

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()

        return input_ids, attention_mask, token_type_ids

    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=1,
                                         collate_fn=collate_fn,
                                         shuffle=False,
                                         drop_last=True)
    p_model = BertModel.from_pretrained(prbert_model)
    if torch.cuda.is_available():
        p_model = p_model.cuda()

    for param in p_model.parameters():
        param.requires_grad_(False)
    model = Model(p_model)

    target_embeds = []
    stat_emb = {}
    for i, (input_ids, attention_mask, token_type_ids) in enumerate(loader):
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)

        refresh_word_embeddings(input_ids, out, stat_emb, target_embeds, keyword, token, sen_emb=sen_emb)
    stat_emb = avg_stat_emb(stat_emb)

    cloud = target_embeds
    return cloud, stat_emb




