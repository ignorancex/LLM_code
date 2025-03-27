#
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#


import numpy as np
import os
import glob
import tqdm
from scipy.special import logsumexp


class ComputeParallelDict:
    def __init__(self, keys_in, key_out, dict_op):
        self.keys_in = keys_in
        self.key_out = key_out
        self.dict_op = dict_op
        self.list_name = list(self.dict_op.keys())

    def reset(self, list_name=None, **key):
        if list_name is not None:
            self.list_name = [_ for _ in list_name if _ in self.dict_op]
        else:
            self.list_name = list(self.dict_op.keys())

        for name in self.list_name:
            self.dict_op[name].reset(**key)

        return self

    def save(self):
        for name in self.dict_op:
            self.dict_op[name].save()

    def __call__(self, inp):
        out = dict()
        for name in self.list_name[:-1]:
            inp_ = {_: inp[_] for _ in self.keys_in if _ in inp}
            out[name] = self.dict_op[name](inp_)[self.key_out]

        inp = self.dict_op[self.list_name[-1]](inp)
        out[self.list_name[-1]] = inp[self.key_out]
        del inp[self.key_out]
        for name in self.list_name:
            inp[self.key_out+'_'+name] = out[name]
            if name == '':
                inp[self.key_out] = out[name]

        return inp


class ComputeDistanceAudioVideo:
    def __init__(self, folder_refs, key_feats=('embs_feat_video', 'embs_feat_audio'), normalize=True):
        self.key_feats = key_feats
        self.list_refs = dict()
        self.fileall = os.path.join(folder_refs, 'embs_all_tracks.npz')
        if len(self.key_feats) > 1:
            self.mu = np.zeros(len(self.key_feats)+1)
            self.sigma = np.ones(len(self.key_feats)+1)
        else:
            self.mu = np.zeros(1)
            self.sigma = np.ones(1)

        if os.path.isfile(self.fileall):
            metadata = dict(np.load(self.fileall, allow_pickle=True))
            assert np.array_equal(metadata['key_feats'], self.key_feats)
            self.list_refs = metadata['list_refs'].tolist()
            self.mu = metadata['mu']
            self.sigma = metadata['sigma']
            normalize = False
            del metadata
            print(self.fileall)
            for filename in self.list_refs:
                print(filename, self.list_refs[filename].shape, self.list_refs[filename].dtype)
            print('Statistics:', self.mu, self.sigma, flush=True)
        else:
            for folder_vid in glob.glob(os.path.join(folder_refs, '*')):
                if os.path.isdir(folder_vid):
                    filename = os.path.basename(folder_vid)
                    listfile = glob.glob(os.path.join(folder_vid, 'embs_track*.npz'))
                    if len(listfile) > 0:
                        self.list_refs[filename] = np.stack(
                            [np.concatenate([np.load(filename)[key] for filename in listfile], 0)
                                for key in self.key_feats], -1)

                        print(filename, self.list_refs[filename].shape, self.list_refs[filename].dtype)

        if normalize:
            self.list_insize = None
            file_statistics = os.path.join(folder_refs, 'statistics.npz')
            if os.path.isfile(file_statistics):
                print('Loading statistics from ', file_statistics)
                print('If you want to recompute the statistics, delete the file', file_statistics)
                self.mu = np.load(file_statistics)['mu']
                self.sigma = np.load(file_statistics)['sigma']
            else:
                print('Computing statistics on the reference videos .....', flush=True)
                cum_mean = 0.0
                cum_vqm = 0.0
                cum_dem = 0.0

                for filename in self.list_refs:
                    feat = self.list_refs[filename]
                    self.reset(filename)
                    cum_mean_vid = 0.0
                    cum_vqm_vid = 0.0
                    cum_dem_vid = 0.0
                    for split in tqdm.tqdm(range(0, len(feat), 100)):
                        dist = self.compute_dist(feat[split:(split + 100)], normalize=False)
                        cum_mean_vid = cum_mean_vid + np.sum(dist, 0)
                        cum_vqm_vid = cum_vqm_vid + np.sum(dist * dist, 0)
                        cum_dem_vid += len(dist)
                    cum_mean = cum_mean + (cum_mean_vid / cum_dem_vid)
                    cum_vqm = cum_vqm + (cum_vqm_vid / cum_dem_vid)
                    cum_dem += 1

                self.mu = cum_mean / cum_dem
                self.sigma = np.sqrt(cum_vqm / cum_dem - (self.mu ** 2))
                try:
                    np.savez(file_statistics, mu=self.mu, sigma=self.sigma)
                    print('Saving statistics to ', file_statistics)
                except:
                    pass
            print('Statistics:', self.mu, self.sigma, flush=True)
        self.list_insize = None

    def save(self):
        if not os.path.isfile(self.fileall):
            np.savez(self.fileall,
                     key_feats=self.key_feats, list_refs=self.list_refs,
                     mu=self.mu, sigma=self.sigma)
        else:
            print(f'WARNING: {self.fileall} already exits!')

    def compute_dist(self, feats, filename=None, normalize=True):
        if filename is not None:
            self.reset(filename)
        num_feat = feats.shape[0]
        if len(self.key_feats) > 1:
            out_dist = np.nan*np.zeros((num_feat, len(self.key_feats) + 1))
        else:
            out_dist = np.nan * np.zeros((num_feat, 1))

        if num_feat == 0:
            return out_dist

        for filename in self.list_insize:
            for index in range(len(self.key_feats)):
                ref_embs = self.list_refs[filename][None, :, :, index]
                dist = np.min(np.sum(np.square(ref_embs - feats[:, None, :, index]), -1), -1)
                out_dist[:, index] = np.fmin(out_dist[:, index], dist)

            if len(self.key_feats) > 1:
                ref_embs = self.list_refs[filename][None, :, :, :]
                dist = np.min(np.sum(np.square(ref_embs - feats[:, None, :, :]), (-2, -1)), -1)
                out_dist[:, -1] = np.fmin(out_dist[:, -1], dist)

        if normalize:
            out_dist = (out_dist-self.mu[None, :])/self.sigma[None, :]

        return out_dist

    def reset(self, filename=None):
        if filename is None:
            self.list_insize = list(self.list_refs.keys())
            return self

        exclude = os.path.splitext(filename.split('/')[-1])[0]
        self.list_insize = list()
        for key in self.list_refs:
            if exclude == key:
                print('exclude:', key)
            else:
                self.list_insize.append(key)

        return self

    def __call__(self, inp):
        nan_feats = [inp[key] for key in self.key_feats if key in inp][0]
        nan_feats = np.nan * nan_feats if len(nan_feats)>0 else list()
        feats = [inp[key] if key in inp else nan_feats for key in self.key_feats]
        feats = np.stack(feats, -1)
        inp['embs_dists'] = self.compute_dist(feats)
        for key_feat in self.key_feats:
            if key_feat in inp:
                del inp[key_feat]

        return inp


class ComputeTemporalMulti:

    def __init__(self, time, stride, list_elem, function, outkeys):
        self.time = time
        self.stride = stride
        self.function = function
        self.outkeys = outkeys
        self.dict_feats = {key: dict() for key in list_elem}
        self.dict_count = dict()
        self.dict_inds = dict()
        assert self.stride <= self.time

    def reset(self):
        self.dict_feats = {key: dict() for key in self.dict_feats}
        self.dict_count = dict()
        self.dict_inds = dict()
        return self

    def num_tracks(self):
        return len(self.dict_inds)

    def __call__(self, inp):
        out = {'embs_track': list(), 'embs_range': list() }
        for _ in self.outkeys:
            out['embs_' + _] = list()

        for index in range(len(inp['id_track'])):
            i = inp['id_track'][index]
            t = inp['image_inds'][index]
            if i < 0:
                continue
            if i in self.dict_count:
                if self.dict_inds[i] == t:
                    for key in self.dict_feats:
                        self.dict_feats[key][i].append(inp[key][index])
                    self.dict_count[i] = self.dict_count[i]+1
                    self.dict_inds[i] = t+1
                else:
                    for key in self.dict_feats:
                        self.dict_feats[key][i] = [inp[key][index], ]
                    self.dict_count[i] = 1
                    self.dict_inds[i] = t+1
            else:
                for key in self.dict_feats:
                    self.dict_feats[key][i] = [inp[key][index], ]
                self.dict_count[i] = 1
                self.dict_inds[i] = t+1

        for i in self.dict_count:
            while self.dict_count[i] >= self.time:
                t0 = self.dict_inds[i] - self.dict_count[i]
                self.dict_count[i] = self.dict_count[i] - self.stride
                out['embs_range'].append([t0, t0 + self.time])
                out['embs_track'].append(i)

                f0 = self.function(**{key: self.dict_feats[key][i][:self.time] for key in self.dict_feats})
                for _ in self.outkeys:
                    out['embs_' + _].append(f0[_])

                for key in self.dict_feats:
                    self.dict_feats[key][i] = self.dict_feats[key][i][self.stride:]

        return out


class PassIdentity:

    def __init__(self):
        pass

    def reset(self):
        return self

    def __call__(self, inp):
        return inp


