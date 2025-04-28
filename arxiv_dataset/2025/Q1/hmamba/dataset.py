import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

class GoPDataset(Dataset):
    def __init__(self, set, data_dir=None, data_dir2=None, data_dir3=None, am='librispeech', mode='apa'):

        # normalize the input to 0 mean and unit std.
        if am=='librispeech':
            norm_mean, norm_std = 3.203, 4.045
        elif am=='paiia':
            norm_mean, norm_std = -0.652, 9.737
        elif am=='paiib':
            norm_mean, norm_std = -0.516, 9.247
        else:
             raise ValueError('Acoustic Model Unrecognized.')

        self.mode = mode
        self.data_dir2 = data_dir2
        self.data_dir3 = data_dir3
        self.feat2 = []
        self.feat3 = []

        # NOTE: features
        self.feat = torch.tensor(np.load(data_dir+'/'+set+'/feat.npy'), dtype=torch.float)
        # normalize the GOP feature using the training set mean and std (only count the valid token features, exclude the padded tokens).
        self.feat = self.norm_valid(self.feat, norm_mean, norm_std)
        if data_dir2:
            self.feat2 = [torch.tensor(np.load(data+'/'+set+'/feat.npy'), dtype=torch.float) for data in data_dir2.split()]
        if data_dir3:
            self.feat3 = torch.tensor(np.load(data_dir3+'/'+set+'/feat.npy'), dtype=torch.float)

        # NOTE: labels
        self.phn_label = torch.tensor(np.load(data_dir+'/'+set+'/label_phn.npy'), dtype=torch.float)
        if self.mode=="apa":
            # last dim is phone score
            self.phn_label[:, :, -1] = self.phn_label[:, :, -1]
            self.utt_label = torch.tensor(np.load(data_dir+'/'+set+'/label_utt.npy'), dtype=torch.float)
            self.word_label = torch.tensor(np.load(data_dir+'/'+set+'/label_word.npy'), dtype=torch.float)
            # normalize the utt_label to 0-2 (same with phn score range)
            self.utt_label = self.utt_label / 5
            # the last dim is word_id, so not normalizing
            self.word_label[:, :, 0:3] = self.word_label[:, :, 0:3] / 5

        # NOTE: record utt_id
        self.utt_id = np.load(data_dir+'/'+set+'/utt_id.npy', allow_pickle=True)


    # only normalize valid tokens, not padded token
    def norm_valid(self, feat, norm_mean, norm_std):
        norm_feat = torch.zeros_like(feat)
        # batch
        for i in range(feat.shape[0]):
            # seq
            for j in range(feat.shape[1]):
                if feat[i, j, 0] != 0:
                    norm_feat[i, j, :] = (feat[i, j, :] - norm_mean) / norm_std
                else:
                    break
        return norm_feat

    def __len__(self):
        return self.feat.shape[0]

    def __getitem__(self, idx):

        feat2 = [ x[idx,:] for x in self.feat2 ] if self.data_dir2 else []
        feat3 = self.feat3[idx, :] if self.data_dir3 else []
        # NOTE: please check gen_seq_data_phn.py
        if self.mode=="apa":
            # audio_input, audio_input2, audio_input3
            # cano phns, real phns, bies,
            # phn_label, word_label, utt_label, uttid
            return self.feat[idx, :], feat2, feat3, \
                self.phn_label[idx, :, 0], self.phn_label[idx, :, 1], self.phn_label[idx, :, 2], \
                self.phn_label[idx, :, -1], self.word_label[idx, :], self.utt_label[idx, :], self.utt_id[idx]
        elif self.mode=="mdd":
            # audio_input, audio_input2, audio_input3
            # cano phns, real phns, bies, uttid
            return self.feat[idx, :], feat2, feat3, \
                self.phn_label[idx, :, 0], self.phn_label[idx, :, 1], self.phn_label[idx, :, 2], self.utt_id[idx]
        else:
            raise ValueError(f"Dataset mode must to be 'apa' or 'mdd', {self.mode} is provided.")

