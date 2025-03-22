# coding:utf-8
import math
import torch
from torch.utils import data
from dataset_utils import load_file, load_json_file
import numpy as np
from transformers import BertTokenizer, AutoTokenizer, AutoModel
# bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("./test-ddbothc1")
# sent_bert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

def get_string_text(tokens_a, tokens_b):


    tokens = []
    segment_ids = []

    tokens.append(bert_tokenizer.cls_token)
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)

    tokens.append(bert_tokenizer.sep_token)
    segment_ids.append(0)

    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)

    # tokens.append(bert_tokenizer.sep_token)
    # segment_ids.append(1)

    return tokens, segment_ids

class AMRDialogSetNew(data.Dataset):
# filtering
    def __init__(self, data_path, tokenize_fn, word_vocab, concept_vocab, relation_vocab, lower=True, use_bert=False, sent_bert=False):
        self.instance = []
        self.count = 0
        self.max_tok_len = 512 if use_bert else 1000
        # f1 = open(data_path +  type + "/con.concept", "w", encoding="utf-8")
        # f2 = open(data_path +  type + "/con.path", "w", encoding="utf-8")
        # data_path = "./data/dev/dev_"
        # for type in ['adv', 'pos', 'ran']:
        # for type in ['adv', 'pos', 'ran']:
        for type in ['adv']:
            #adv ==0 original
            #sa bert pos=0
            if type in 'adv':
                label = 0
            elif type == 'pos':
                label = 1
            else:
                label = 2
            # f1 = open(data_path + type + "/con_new.concept", "w", encoding="utf-8")
            # f2 = open(data_path + type + "/con_new.path", "w", encoding="utf-8")
            # f3 = open(data_path + type + "/new.concept", "w", encoding="utf-8")
            # f4 = open(data_path + type + "/new.path", "w", encoding="utf-8")
            # f5 = open(data_path + type + "/new.src", "w", encoding="utf-8")
            # f6 = open(data_path + type + "/new.tgt", "w", encoding="utf-8")

            # with open(data_path + type + '/ori.src', 'r', encoding='utf-8') as srcf:
            #     with open(data_path + type + '/ori.concept', 'r', encoding='utf-8') as conf:
            #         with open(data_path + type + '/ori.path', 'r', encoding='utf-8') as con_relf:
            #             with open(data_path + type + '/ori.tgt', 'r', encoding='utf-8') as tgtf:
            with open(data_path + type + '/new.src', 'r', encoding='utf-8') as srcf, \
                open(data_path + type + '/new.concept', 'r', encoding='utf-8') as conf,\
                open(data_path + type + '/new.path', 'r', encoding='utf-8') as con_relf, \
                open(data_path + type + '/new.tgt', 'r', encoding='utf-8') as tgtf,\
                open(data_path + type + '/con_sim.concept', 'r', encoding='utf-8') as conc, \
                open(data_path + type + '/con_sim.path', 'r', encoding='utf-8') as conp:
                for src_tok, src_concept, con_relation_raw, tgt_tok, con_concept, con_relation in zip(srcf, conf, con_relf, tgtf, conc, conp):
                    all_segment_ids = []
                    all_texts = []
                    # nsp_labels = []
                    context_texts = []
                    response_texts = []

                    tgt_tok = tgt_tok.replace('\n', '').replace('@@ ', '')
                    # tgt_tok = tgt_tok

                    #only for building vocab and pt
                    src_tok = src_tok.replace('  ', '').replace('\n', '').replace('@@ ', '')
                    # print('src_tok', src_tok)
                    # src_tok = src_tok

                    con_relation_lst_raw = con_relation_raw.strip().split(" ")
                    context_relation_lst_raw = con_relation.strip().split(" ")
                    seg_len = int(math.sqrt(len(con_relation_lst_raw)))
                    con_seg_len = int(math.sqrt(len(context_relation_lst_raw)))

                    assert seg_len * seg_len == len(con_relation_lst_raw)
                    print(data_path + type, con_seg_len, con_seg_len * con_seg_len, len(context_relation_lst_raw))

                    if lower:
                        con_relation_lst = [
                            ' '.join(con_relation_lst_raw[i: i + seg_len]).lower()
                            for i in range(0, len(con_relation_lst_raw), seg_len)
                        ]
                        context_relation_lst = [
                            ' '.join(context_relation_lst_raw[i: i + con_seg_len]).lower()
                            for i in range(0, len(context_relation_lst_raw), con_seg_len)
                        ]
                    else:
                        con_relation_lst = [
                            ' '.join(con_relation_lst_raw[i: i + seg_len])
                            for i in range(0, len(con_relation_lst_raw), seg_len)
                        ]

                    if lower:
                        src_tok = src_tok.lower().strip()
                        src_id = tokenize_fn(word_vocab, src_tok, 1, 1, dtype="word")
                        tgt_id = tokenize_fn(word_vocab, tgt_tok.strip().lower(), 1, 1, dtype="word")

                        concept_id = tokenize_fn(concept_vocab, src_concept.strip().lower(), 1, 1, dtype="concept")
                        context_concept_id = tokenize_fn(concept_vocab, con_concept.strip().lower(), 1, 1, dtype="concept")


                    con_rel_id = tokenize_fn(relation_vocab, con_relation_lst, 1, 2, dtype="relation")
                    context_rel_id = tokenize_fn(relation_vocab, context_relation_lst, 1, 2, dtype="relation")

                    
                    if len(con_rel_id) == len(concept_id) and len(con_rel_id[0]) == len(concept_id):
                        if len(context_rel_id) == len(context_concept_id) and len(context_rel_id[0]) == len(context_concept_id):
                            print(tgt_tok)
                            if label != 2:
                                self.instance.append((src_id, concept_id, con_rel_id, tgt_id, tgt_tok, context_concept_id, context_rel_id, label))
                                self.count += 1
                                               

    def __len__(self):
        return len(self.instance)

    def __getitem__(self, index):
        src_id, concept_id, con_rel_id, tgt_id, tgt_tok, segment_ids, st_id, label = self.instance[index]
        return src_id, concept_id, con_rel_id, tgt_id, tgt_tok, segment_ids, label


    def cal_max_len(self, ids, curdepth, maxdepth):
        """calculate max sequence length"""
        assert curdepth <= maxdepth
        if isinstance(ids[0], list):
            res = max([self.cal_max_len(k, curdepth + 1, maxdepth) for k in ids])
        else:
            res = len(ids)
        return res


    def collate_fn(self, batch, ):
        # sv, cv, rv, tv, tstr, mask, wr = zip(*batch)    # sv, cv, rv and tv: [batch, seq];
        sv, cv, rv, tv, tstr, context_concept_id, context_rel, label = zip(*batch)
        pad_sv, pad_cv, pad_rv, pad_tv, pad_context_concept, pad_context_rel = [], [], [], [], [], []
        pad_st = []

        sv_max_len = min(max([self.cal_max_len(s, 1, 1) for s in sv]), self.max_tok_len)
        cv_max_len = max([self.cal_max_len(c, 1, 1) for c in cv])
        rv_max_len = max([self.cal_max_len(r, 1, 2) for r in rv])
        tv_max_len = max([self.cal_max_len(t, 1, 1) for t in tv])
        context_concept_id_max_len = max([self.cal_max_len(t, 1, 1) for t in context_concept_id])
        context_rel_max_len = max([self.cal_max_len(t, 1, 2) for t in context_rel])

        st_max_len = min(max([self.cal_max_len(st, 1, 1) for st in (sv+tv)]), self.max_tok_len)

        assert rv_max_len == cv_max_len, "Error, rv_max_len should be equal with cv_max_len!!"

        for i in range(len(sv)):
            tmp_sv = [0] * sv_max_len
            tmp_cv = [0] * cv_max_len
            tmp_tv = [0] * tv_max_len
            tmp_st = [0] * st_max_len
            tmp_rv = [
                [0 for i in range(rv_max_len)] for j in range(rv_max_len)
            ]  # [rv_max_len, rv_max_len]
            tmp_context_concept = [0] * context_concept_id_max_len
            tmp_context_rel = [
                [0 for i in range(context_rel_max_len)] for j in range(context_rel_max_len)
            ]  # [rv_max_len, rv_max_len]
    
            ith_sv_len = min(len(sv[i]), self.max_tok_len)
            # print('sv: ori_len: {}, sv_len: {}'.format(len(sv[i]), ith_sv_len))
            # tmp_sv[:ith_sv_len] = map(int, sv[i][:self.max_tok_len])    #
            tmp_sv[:ith_sv_len] = map(int, sv[i][-self.max_tok_len:])    #

            ith_cv_len = len(cv[i])
            tmp_cv[:ith_cv_len] = map(int, cv[i])

            ith_tv_len = len(tv[i])
            tmp_tv[:ith_tv_len] = map(int, tv[i])

            ith_st_len = min(len((sv+tv)[i]), self.max_tok_len)
            tmp_st[:ith_st_len] = map(int, ((sv+tv)[i])[-self.max_tok_len:])
            # tmp_st = tmp_sv + tmp_tv
            ith_rv_len = len(rv[i])     # rv_len
            for j in range(ith_rv_len):
                rv_len_j = len(rv[i][j])
                tmp_rv[j][:rv_len_j] = map(int, rv[i][j])

            # padded_length = len(tmp_st)
            # tmp_sid = [ x + [0] * (padded_length-len(x)) for x in segment_ids]
            # print('segment_ids[i]', segment_ids[i])
            # print("tmp_st", tmp_st)
 
            # tmp_sid = segment_ids[i] + [0] * (padded_length-len(segment_ids[i]))
            # print('len(tmp_sid)', len(tmp_sid))  
            # print("tmp_sid", tmp_sid)
            # print("length type_id", len(tmp_sid))

            ith_context_concept_len = len(context_concept_id[i])
            tmp_context_concept[:ith_context_concept_len] = map(int, context_concept_id[i])

            ith_context_rel_len = len(context_rel[i])     # rv_len
            for j in range(ith_context_rel_len):
                context_rel_len_j = len(context_rel[i][j])
                tmp_context_rel[j][:context_rel_len_j] = map(int, context_rel[i][j])

            pad_sv.append(tmp_sv)   # [batch, sv_max_len]
            pad_cv.append(tmp_cv)
            pad_rv.append(tmp_rv)   # [batch, len_c, len_c]
            pad_tv.append(tmp_tv)
            # pad_sid.append(tmp_sid)
            pad_st.append(tmp_st)   

            pad_context_concept.append(tmp_context_concept)
            pad_context_rel.append(tmp_context_rel)
            # pad_mask.append(tmp_mask)
            # pad_wr.append(tmp_wr)
            # pad_kv.append(tmp_kv)  # [batch, know_num, know_seq]

        sv_len = [min(len(s), self.max_tok_len) for s in sv]
        st_len = [min(len(st), self.max_tok_len) for st in (sv+tv)]
       
        cv_len = [len(c) for c in cv]
        tv_len = [len(t) for t in tv]

        context_concept_len = [len(t) for t in context_concept_id]
        context_rel_len = [len(t) for t in context_rel]
        # kv_len = [
        #     [len(kt) for kt in k] + [0 for _ in range(kn_max_num - len(k))] for k in kv
        # ]  # [batch, know_num]
        # print(pad_sv)
        # tmp_sv_ = np.asarray(pad_sv)
        # print('tmp_sv', tmp_sv_.shape)
        return (
            torch.tensor(pad_sv, dtype=torch.long).view((-1, sv_max_len)),
            torch.tensor(sv_len, dtype=torch.long),
            torch.tensor(pad_cv, dtype=torch.long).view((-1, cv_max_len)),
            torch.tensor(cv_len, dtype=torch.long),
            torch.tensor(pad_rv, dtype=torch.long).view((-1, rv_max_len, rv_max_len)),
            torch.tensor(pad_tv, dtype=torch.long).view((-1, tv_max_len)),
            torch.tensor(tv_len, dtype=torch.long),
            tstr,
            torch.tensor(pad_context_concept, dtype=torch.long).view((-1, context_concept_id_max_len)),
            torch.tensor(context_concept_len, dtype=torch.long),
            torch.tensor(pad_context_rel, dtype=torch.long).view((-1, context_rel_max_len, context_rel_max_len)),
            torch.tensor(context_rel_len, dtype=torch.long),
            
            torch.tensor(pad_st, dtype=torch.long).view((-1, st_max_len)),
            torch.tensor(st_len, dtype=torch.long),
            # torch.tensor(pad_sid, dtype=torch.long),

            torch.tensor(label, dtype=torch.long)
        )

    def collate_fn_bert(self, batch, ):
        # sv, cv, rv, tv, tstr, mask, wr = zip(*batch)    # sv, cv, rv and tv: [batch, seq];
        sv, cv, rv, tv, tstr, context_concept_id, context_rel, segment_ids, st, label = zip(*batch)
        # src_id, concept_id, con_rel_id, tgt_id, tgt_tok

        #only valid when sent_bert
        # sv = sv[0]
        # sv = [s[:-1] for s in sv]                       # remove eos 

        # pad_sv, pad_cv, pad_rv, pad_tv, pad_mask, pad_wr = [], [], [], [], [], []
        pad_sv, pad_cv, pad_rv, pad_tv, pad_context_concept, pad_context_rel = [], [], [], [], [], []
        pad_sid, pad_st = [], []

        sv_max_len = min(max([self.cal_max_len(s, 1, 1) for s in sv]), self.max_tok_len)
        cv_max_len = max([self.cal_max_len(c, 1, 1) for c in cv])
        rv_max_len = max([self.cal_max_len(r, 1, 2) for r in rv])
        tv_max_len = max([self.cal_max_len(t, 1, 1) for t in tv])
        context_concept_id_max_len = max([self.cal_max_len(t, 1, 1) for t in context_concept_id])
        context_rel_max_len = max([self.cal_max_len(t, 1, 2) for t in context_rel])

        st_max_len = min(max([self.cal_max_len(s, 1, 1) for s in st]), self.max_tok_len)

        assert rv_max_len == cv_max_len, "Error, rv_max_len should be equal with cv_max_len!!"

        for i in range(len(sv)):
            tmp_sv = [0] * sv_max_len
            tmp_cv = [0] * cv_max_len
            tmp_tv = [0] * tv_max_len
            tmp_st = [0] * st_max_len
            tmp_rv = [
                [0 for i in range(rv_max_len)] for j in range(rv_max_len)
            ]  # [rv_max_len, rv_max_len]
            tmp_context_concept = [0] * context_concept_id_max_len
            tmp_context_rel = [
                [0 for i in range(context_rel_max_len)] for j in range(context_rel_max_len)
            ]  # [rv_max_len, rv_max_len]
            # tmp_mask = [
            #     [0 for i in range(mask_max_len)] for j in range(mask_max_len)
            # ]
            # tmp_wr = [
            #     [0 for i in range(wr_max_len)] for j in range(wr_max_len)
            # ]

            ith_sv_len = min(len(sv[i]), self.max_tok_len)
            # print('sv: ori_len: {}, sv_len: {}'.format(len(sv[i]), ith_sv_len))
            # tmp_sv[:ith_sv_len] = map(int, sv[i][:self.max_tok_len])    #
            tmp_sv[:ith_sv_len] = map(int, sv[i][-self.max_tok_len:])    #

            ith_cv_len = len(cv[i])
            tmp_cv[:ith_cv_len] = map(int, cv[i])

            ith_tv_len = len(tv[i])
            tmp_tv[:ith_tv_len] = map(int, tv[i])

            ith_st_len = min(len(st[i]), self.max_tok_len)
            tmp_st[:ith_st_len] = map(int, (st[i])[-self.max_tok_len:])
            # print('len(tmp_st)', len(tmp_st))  
            ith_rv_len = len(rv[i])     # rv_len
            for j in range(ith_rv_len):
                rv_len_j = len(rv[i][j])
                tmp_rv[j][:rv_len_j] = map(int, rv[i][j])

            padded_length = len(tmp_st)
            # tmp_sid = [ x + [0] * (padded_length-len(x)) for x in segment_ids]
            # print('segment_ids[i]', segment_ids[i])
            # print("tmp_st", tmp_st)
 
            tmp_sid = segment_ids[i] + [0] * (padded_length-len(segment_ids[i]))
            # print('len(tmp_sid)', len(tmp_sid))  
            # print("tmp_sid", tmp_sid)
            # print("length type_id", len(tmp_sid))

            ith_context_concept_len = len(context_concept_id[i])
            tmp_context_concept[:ith_context_concept_len] = map(int, context_concept_id[i])

            ith_context_rel_len = len(context_rel[i])     # rv_len
            for j in range(ith_context_rel_len):
                context_rel_len_j = len(context_rel[i][j])
                tmp_context_rel[j][:context_rel_len_j] = map(int, context_rel[i][j])

            pad_sv.append(tmp_sv)   # [batch, sv_max_len]
            pad_cv.append(tmp_cv)
            pad_rv.append(tmp_rv)   # [batch, len_c, len_c]
            pad_tv.append(tmp_tv)
            pad_sid.append(tmp_sid[:512])
            pad_st.append(tmp_st)   

            pad_context_concept.append(tmp_context_concept)
            pad_context_rel.append(tmp_context_rel)
            # pad_mask.append(tmp_mask)
            # pad_wr.append(tmp_wr)
            # pad_kv.append(tmp_kv)  # [batch, know_num, know_seq]

        sv_len = [min(len(s), self.max_tok_len) for s in sv]
        st_len = [min(len(s), self.max_tok_len) for s in st]
       
        cv_len = [len(c) for c in cv]
        tv_len = [len(t) for t in tv]

        context_concept_len = [len(t) for t in context_concept_id]
        context_rel_len = [len(t) for t in context_rel]
        # kv_len = [
        #     [len(kt) for kt in k] + [0 for _ in range(kn_max_num - len(k))] for k in kv
        # ]  # [batch, know_num]
        # print(pad_sv)
        # tmp_sv_ = np.asarray(pad_sv)
        # print('tmp_sv', tmp_sv_.shape)
        return (
            torch.tensor(pad_sv, dtype=torch.long).view((-1, sv_max_len)),
            torch.tensor(sv_len, dtype=torch.long),
            torch.tensor(pad_cv, dtype=torch.long).view((-1, cv_max_len)),
            torch.tensor(cv_len, dtype=torch.long),
            torch.tensor(pad_rv, dtype=torch.long).view((-1, rv_max_len, rv_max_len)),
            torch.tensor(pad_tv, dtype=torch.long).view((-1, tv_max_len)),
            torch.tensor(tv_len, dtype=torch.long),
            tstr,
            torch.tensor(pad_context_concept, dtype=torch.long).view((-1, context_concept_id_max_len)),
            torch.tensor(context_concept_len, dtype=torch.long),
            torch.tensor(pad_context_rel, dtype=torch.long).view((-1, context_rel_max_len, context_rel_max_len)),
            torch.tensor(context_rel_len, dtype=torch.long),
            torch.tensor(pad_sid, dtype=torch.long),
            torch.tensor(pad_st, dtype=torch.long).view((-1, st_max_len)),
            torch.tensor(st_len, dtype=torch.long),

            torch.tensor(label, dtype=torch.long)
        )

    def GetDataloader(self, batch_size, shuffle, num_workers):
        data_loader = data.DataLoader(
            self.instance,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )
        return data_loader

# class AMRDialogSetNew(data.Dataset):
#     # def __init__(self, data_path, tokenize_fn, word_vocab, concept_vocab, relation_vocab, word_rel_vocab, lower=True, use_bert=False):
#     def __init__(self, data_path, tokenize_fn, word_vocab, concept_vocab, relation_vocab, lower=True, use_bert=False):
#         self.instance = []
#         self.count = 0
#         self.max_tok_len = 512 if use_bert else 1000
#         # f1 = open(data_path + "_new.src", "w", encoding="utf-8")
#         # f2 = open(data_path + "_new.concept", "w", encoding="utf-8")
#         # f3 = open(data_path + "_new.path", "w", encoding="utf-8")
#         # f4 = open(data_path + "_new.tgt", "w", encoding="utf-8")
        
#         # with open(data_path + '_new.src', 'r', encoding='utf-8') as srcf:
#         #     with open(data_path + '_new.concept', 'r', encoding='utf-8') as conf:
#         #         with open(data_path + '_new.path', 'r', encoding='utf-8') as con_relf:
#         #             with open(data_path + '_new.tgt', 'r', encoding='utf-8') as tgtf:
#         # for type in ['adv', 'pos', 'ran']:
#         for type in ['adv', 'pos']:
#             if type == 'adv':
#                 label = 0
#             elif type == 'pos':
#                 label = 1
#             else:
#                 label = 2
                
#             with open(data_path + type + '/new.src', 'r', encoding='utf-8') as srcf:
#                 with open(data_path + type + '/new.concept', 'r', encoding='utf-8') as conf:
#                     with open(data_path + type + '/new.path', 'r', encoding='utf-8') as con_relf:
#                         with open(data_path + type + '/new.tgt', 'r', encoding='utf-8') as tgtf:
#                             for src_tok, src_concept, con_relation_raw, tgt_tok in zip(srcf, conf, con_relf, tgtf):
#                                 tgt_tok = tgt_tok.replace('\n', '')
#                                 # print('tgt_tok',tgt_tok)
#                                 src_tok_ori = src_tok
#                                 # src_tok = src_tok.replace(' <sep>', '')
#                                 src_tok = src_tok.replace('<sep>', '')
#                                 # print('src', src_tok)
#                                 con_relation_lst_raw = con_relation_raw.strip().split(" ")
#                                 seg_len = int(math.sqrt(len(con_relation_lst_raw)))
#                                 assert seg_len * seg_len == len(con_relation_lst_raw)

#                                 # print('src_concept', len(src_concept.split()))
#                                 # print('con_relation_lst_raw', len(con_relation_lst_raw))
#                                 if lower:
#                                     con_relation_lst = [
#                                         ' '.join(con_relation_lst_raw[i: i + seg_len]).lower()
#                                         for i in range(0, len(con_relation_lst_raw), seg_len)
#                                     ]
#                                 else:
#                                     con_relation_lst = [
#                                         ' '.join(con_relation_lst_raw[i: i + seg_len])
#                                         for i in range(0, len(con_relation_lst_raw), seg_len)
#                                     ]

#                                 if lower:
#                                     if use_bert:
#                                         # print('using bert ...')
#                                         # src_id = bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + src_tok.strip().lower().split())
#                                         src_id = bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + src_tok.strip().lower().split() + ['[SEP]'] + tgt_tok.strip().lower().split())
#                                     else:
#                                         src_tok = src_tok.lower().strip()
#                                         src_id = tokenize_fn(word_vocab, src_tok, 1, 1, dtype="word")

#                                     concept_id = tokenize_fn(concept_vocab, src_concept.strip().lower(), 1, 1, dtype="concept")
#                                     tgt_id = tokenize_fn(word_vocab, tgt_tok.strip().lower(), 1, 1, dtype="word")
#                                 else:
#                                     if use_bert:
#                                         src_id = bert_tokenizer.convert_tokens_to_ids(src_tok.strip().lower().split())
#                                     else:
#                                         src_id = tokenize_fn(word_vocab, src_tok.strip(), 1, 1, dtype="word")
#                                     concept_id = tokenize_fn(concept_vocab, src_concept.strip(), 1, 1, dtype="concept")
#                                     tgt_id = tokenize_fn(word_vocab, tgt_tok.strip(), 1, 1, dtype="word")

#                                 con_rel_id = tokenize_fn(relation_vocab, con_relation_lst, 1, 2, dtype="relation")

     
#                                 if len(con_rel_id) == len(concept_id) and len(con_rel_id[0]) == len(concept_id):
                            
                                    
#                                     self.instance.append((src_id, concept_id, con_rel_id, tgt_id, tgt_tok, label))
#                                     self.count += 1
                             

#     def __len__(self):
#         return len(self.instance)

#     def __getitem__(self, index):
#         # src_id, concept_id, con_rel_id, tgt_id, tgt_tok, word_mask, word_rel_id = self.instance[index]
#         # src_id, concept_id, con_rel_id, tgt_id, tgt_tok = self.instance[index]
#         src_id, concept_id, con_rel_id, tgt_id, tgt_tok, label = self.instance[index]
#         # return src_id, concept_id, con_rel_id, tgt_id, tgt_tok, word_mask, word_rel_id
#         # return src_id, concept_id, con_rel_id, tgt_id, tgt_tok
#         return src_id, concept_id, con_rel_id, tgt_id, tgt_tok, label


#     def cal_max_len(self, ids, curdepth, maxdepth):
#         """calculate max sequence length"""
#         assert curdepth <= maxdepth
#         if isinstance(ids[0], list):
#             res = max([self.cal_max_len(k, curdepth + 1, maxdepth) for k in ids])
#         else:
#             res = len(ids)
#         return res

#     def collate_fn(self, batch, ):
#         # sv, cv, rv, tv, tstr, mask, wr = zip(*batch)    # sv, cv, rv and tv: [batch, seq];
#         sv, cv, rv, tv, tstr, label = zip(*batch)
#         # src_id, concept_id, con_rel_id, tgt_id, tgt_tok
#         sv = [s[:-1] for s in sv]                       # remove eos 

#         # pad_sv, pad_cv, pad_rv, pad_tv, pad_mask, pad_wr = [], [], [], [], [], []
#         pad_sv, pad_cv, pad_rv, pad_tv = [], [], [], []


#         sv_max_len = min(max([self.cal_max_len(s, 1, 1) for s in sv]), self.max_tok_len)
#         cv_max_len = max([self.cal_max_len(c, 1, 1) for c in cv])
#         rv_max_len = max([self.cal_max_len(r, 1, 2) for r in rv])
#         tv_max_len = max([self.cal_max_len(t, 1, 1) for t in tv])
#         # mask_max_len = min(max([self.cal_max_len(m, 1, 2) for m in mask]), self.max_tok_len)
#         # wr_max_len = min(max([self.cal_max_len(w, 1, 2) for w in wr]), self.max_tok_len)

#         # kn_max_len = max([self.cal_max_len(k, 1, 2) for k in kv])
#         # kn_max_num = max([len(k) for k in kv])

#         assert rv_max_len == cv_max_len, "Error, rv_max_len should be equal with cv_max_len!!"
#         # assert sv_max_len == mask_max_len and sv_max_len == wr_max_len, "Error, sv_max_len should be equal with mask_max_len and wr_max_len!!"
#         # print('sv_max_len', sv_max_len)
#         for i in range(len(sv)):
#             tmp_sv = [0] * sv_max_len
#             tmp_cv = [0] * cv_max_len
#             tmp_tv = [0] * tv_max_len
#             tmp_rv = [
#                 [0 for i in range(rv_max_len)] for j in range(rv_max_len)
#             ]  # [rv_max_len, rv_max_len]
#             # tmp_mask = [
#             #     [0 for i in range(mask_max_len)] for j in range(mask_max_len)
#             # ]
#             # tmp_wr = [
#             #     [0 for i in range(wr_max_len)] for j in range(wr_max_len)
#             # ]

#             ith_sv_len = min(len(sv[i]), self.max_tok_len)
#             # print('sv: ori_len: {}, sv_len: {}'.format(len(sv[i]), ith_sv_len))
#             # tmp_sv[:ith_sv_len] = map(int, sv[i][:self.max_tok_len])    #
#             tmp_sv[:ith_sv_len] = map(int, sv[i][-self.max_tok_len:])    #

#             ith_cv_len = len(cv[i])
#             tmp_cv[:ith_cv_len] = map(int, cv[i])

#             ith_tv_len = len(tv[i])
#             tmp_tv[:ith_tv_len] = map(int, tv[i])

#             ith_rv_len = len(rv[i])     # rv_len
#             for j in range(ith_rv_len):
#                 rv_len_j = len(rv[i][j])
#                 tmp_rv[j][:rv_len_j] = map(int, rv[i][j])

#             # ith_mask_len = min(len(mask[i]), self.max_tok_len)  # rv_len
#             # for j in range(ith_mask_len):
#             #     mask_len_j = min(len(mask[i][j]), self.max_tok_len)
#             #     # tmp_mask[j][:mask_len_j] = map(int, mask[i][j][:self.max_tok_len])
#             #     tmp_mask[j][:mask_len_j] = map(int, mask[i][j][-self.max_tok_len:])

#             # ith_wr_len = min(len(wr[i]), self.max_tok_len)  # rv_len
#             # for j in range(ith_wr_len):
#             #     wr_len_j = min(len(wr[i][j]), self.max_tok_len)
#             #     # tmp_wr[j][:wr_len_j] = map(int, wr[i][j][:self.max_tok_len])
#             #     tmp_wr[j][:wr_len_j] = map(int, wr[i][j][-self.max_tok_len:])

#             # kv_num = len(kv[i])
#             # for j in range(kv_num):
#             #     kn_len = len(kv[i][j])
#             #     tmp_kv[j][:kn_len] = map(int, kv[i][j])

#             pad_sv.append(tmp_sv)   # [batch, sv_max_len]
#             pad_cv.append(tmp_cv)
#             pad_rv.append(tmp_rv)   # [batch, len_c, len_c]
#             pad_tv.append(tmp_tv)
#             # pad_mask.append(tmp_mask)
#             # pad_wr.append(tmp_wr)
#             # pad_kv.append(tmp_kv)  # [batch, know_num, know_seq]

#         sv_len = [min(len(s), self.max_tok_len) for s in sv]
#         cv_len = [len(c) for c in cv]
#         tv_len = [len(t) for t in tv]
#         # kv_len = [
#         #     [len(kt) for kt in k] + [0 for _ in range(kn_max_num - len(k))] for k in kv
#         # ]  # [batch, know_num]
#         # print(pad_sv)
#         # tmp_sv_ = np.asarray(pad_sv)
#         # print('tmp_sv', tmp_sv_.shape)
#         return (
#             torch.tensor(pad_sv, dtype=torch.long).view((-1, sv_max_len)),
#             torch.tensor(sv_len, dtype=torch.long),
#             torch.tensor(pad_cv, dtype=torch.long).view((-1, cv_max_len)),
#             torch.tensor(cv_len, dtype=torch.long),
#             torch.tensor(pad_rv, dtype=torch.long).view((-1, rv_max_len, rv_max_len)),
#             torch.tensor(pad_tv, dtype=torch.long).view((-1, tv_max_len)),
#             torch.tensor(tv_len, dtype=torch.long),
#             tstr,
#             torch.tensor(label, dtype=torch.long)
#         )

#     def GetDataloader(self, batch_size, shuffle, num_workers):
#         data_loader = data.DataLoader(
#             self.instance,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             num_workers=num_workers,
#             collate_fn=self.collate_fn,
#         )
#         return data_loader