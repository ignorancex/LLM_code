import torch
import torch.utils.data as Data
import pickle
import random
from utils import load_json, load_pickle


class AmzDataset(Data.Dataset):
    def __init__(self, data_path, set='train', task='ctr', max_hist_len=10, augment=False,
                 item_aug_prefix=None, hist_aug_prefix=None, use_cl_pair=False):
        self.task = task
        self.max_hist_len = max_hist_len
        self.augment = augment
        self.set = set
        self.use_cl_pair = use_cl_pair
        self.data = load_pickle(data_path + f'/{task}.{set}')
        self.stat = load_json(data_path + '/stat.json')
        self.item_num = self.stat['item_num']
        self.attr_num = self.stat['attribute_num']
        self.attr_ft_num = self.stat['attribute_ft_num']
        self.rating_num = self.stat['rating_num']
        self.dense_dim = self.stat['dense_dim']
        if task == 'rerank':
            self.max_list_len = self.stat['rerank_list_len']
        self.length = len(self.data)
        self.sequential_data = load_json(data_path + '/sequential_data.json')
        self.item2attribution = load_json(data_path + '/item2attributes.json')
        datamaps = load_json(data_path + '/datamaps.json')
        self.id2item = datamaps['id2item']
        self.id2user = datamaps['id2user']
        if augment:
            self.hist_aug_data = load_json(data_path + f'/{hist_aug_prefix}_augment.emb')
            self.item_aug_data = load_json(data_path + f'/{item_aug_prefix}_augment.emb')
            # print('item key', list(self.item_aug_data.keys())[:6], len(self.item_aug_data), self.item_num)
        if use_cl_pair:
            pos_pair_dict = load_json(data_path + '/swing_scores.json')
            self.pos_user_pairs = pos_pair_dict['pos_user_pair']
            self.pos_item_pairs = pos_pair_dict['pos_item_pair']

    def __len__(self):
        return self.length

    def __getitem__(self, _id):
        if self.task == 'ctr':
            uid, seq_idx, lb = self.data[_id]
            item_seq, rating_seq = self.sequential_data[str(uid)]
            iid = item_seq[seq_idx]
            hist_seq_len = seq_idx - max(0, seq_idx - self.max_hist_len)
            attri_id = self.item2attribution[str(iid)]
            hist_item_seq = item_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
            hist_rating_seq = rating_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
            hist_attri_seq = [self.item2attribution[str(idx)] for idx in hist_item_seq]
            out_dict = {
                'iid': torch.tensor(iid).long(),
                'aid': torch.tensor(attri_id).long(),
                'lb': torch.tensor(lb).long(),
                'hist_iid_seq': torch.tensor(hist_item_seq).long(),
                'hist_aid_seq': torch.tensor(hist_attri_seq).long(),
                'hist_rate_seq': torch.tensor(hist_rating_seq).long(),
                'hist_seq_len': torch.tensor(hist_seq_len).long()
            }
            if self.augment:
                item_aug_vec = self.item_aug_data[str(self.id2item[str(iid)])]
                hist_aug_vec = self.hist_aug_data[str(self.id2user[str(uid)])]
                out_dict['item_aug_vec'] = torch.tensor(item_aug_vec).float()
                out_dict['hist_aug_vec'] = torch.tensor(hist_aug_vec).float()
                if self.use_cl_pair:
                    pos_item_candi = self.pos_item_pairs[str(iid)]
                    pos_user_candi = self.pos_user_pairs[str(uid)]
                    if len(pos_item_candi) == 0:
                        pos_item_candi = list(self.pos_item_pairs.keys())
                    if len(pos_user_candi) == 0:
                        pos_user_candi = list(self.pos_user_pairs.keys())
                    pos_iid = random.choice(pos_item_candi)
                    pos_uid = random.choice(pos_user_candi)
                    pos_item_aug_vec = self.item_aug_data[str(self.id2item[str(pos_iid)])]
                    pos_hist_aug_vec = self.hist_aug_data[str(self.id2user[str(pos_uid)])]
                    out_dict['pos_item_aug_vec'] = torch.tensor(pos_item_aug_vec).float()
                    out_dict['pos_hist_aug_vec'] = torch.tensor(pos_hist_aug_vec).float()

        elif self.task == 'rerank':
            uid, seq_idx, candidates, candidate_lbs = self.data[_id]
            candidates_attr = [self.item2attribution[str(idx)] for idx in candidates]
            item_seq, rating_seq = self.sequential_data[str(uid)]
            hist_seq_len = seq_idx - max(0, seq_idx - self.max_hist_len)
            hist_item_seq = item_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
            hist_rating_seq = rating_seq[max(0, seq_idx - self.max_hist_len): seq_idx]
            hist_attri_seq = [self.item2attribution[str(idx)] for idx in hist_item_seq]
            out_dict = {
                'iid_list': torch.tensor(candidates).long(),
                'aid_list': torch.tensor(candidates_attr).long(),
                'lb_list': torch.tensor(candidate_lbs).long(),
                'hist_iid_seq': torch.tensor(hist_item_seq).long(),
                'hist_aid_seq': torch.tensor(hist_attri_seq).long(),
                'hist_rate_seq': torch.tensor(hist_rating_seq).long(),
                'hist_seq_len': torch.tensor(hist_seq_len).long()
            }
            if self.augment:
                item_aug_vec = [torch.tensor(self.item_aug_data[str(self.id2item[str(idx)])]).float()
                                for idx in candidates]
                hist_aug_vec = self.hist_aug_data[str(self.id2user[str(uid)])]
                out_dict['item_aug_vec_list'] = item_aug_vec
                out_dict['hist_aug_vec'] = torch.tensor(hist_aug_vec).float()
        else:
            raise NotImplementedError

        return out_dict


