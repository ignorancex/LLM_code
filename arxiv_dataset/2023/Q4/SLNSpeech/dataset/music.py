import os
import random
from .base import BaseDataset


class MUSICMixDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(MUSICMixDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.num_mix = opt.num_mix

    def __getitem__(self, index):
        N = self.num_mix
        frames = [None for n in range(N)]
        signs = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_signs = [[] for n in range(N)]
        path_audios = ['' for n in range(N)]
        center_frames = [0 for n in range(N)]

        # the first video
        sample = self.list_sample[index]
        infos[0] = sample[0].split(',')
        # sample other videos
        # if not self.split == 'train':
        #     random.seed(index)
        for n in range(1, N):
            # indexN = random.randint(0, len(self.list_sample)-1)
            # infos[n] = self.list_sample[indexN]
            infos[n] = sample[n].split(',')
        flag = sample[N].strip()

        # select frames
        # idx_margin = max(int(self.fps * 4) - 1, (self.num_frames // 2) * self.stride_frames)

        for n, infoN in enumerate(infos):
            fake_path, path_audioN, path_frameN, count_framesN = infoN
            path_signN = path_frameN.replace("frames224", "signs")

            if self.split == 'train':
                # random, not to sample start and end n-frames
                # center_frameN = random.randint(idx_margin+1, int(count_framesN)-idx_margin)
                center_frameN = random.randint(10, int(count_framesN) - 10)
                # center_frameN = int(count_framesN) // 2
            else:
                center_frameN = int(count_framesN) // 2
            center_frames[n] = center_frameN

            # absolute frame/audio paths
            for i in range(self.num_frames):
                idx_offset = (i - self.num_frames // 2) * self.stride_frames
                path_frames[n].append(
                    os.path.join(
                        path_frameN,
                        '{:06d}.jpg'.format(center_frameN + idx_offset)))
                path_signs[n].append(
                    os.path.join(
                        path_signN,
                        '{:06d}.jpg'.format(center_frameN + idx_offset)))
            path_audios[n] = path_audioN

            if flag is '0':
                path_frames[n].pop(0)
            #     path_frames[n].pop(0)
            #     path_frames[n].pop(0)
                path_frames[n].append(fake_path)
            #     path_frames[n].append(fake_path)
            #     path_frames[n].append(fake_path)


        # load frames and audios, STFT
        try:
            for n, infoN in enumerate(infos):
                frames[n] = self._load_frames(path_frames[n])
                signs[n] = self._load_signs(path_signs[n])
                # jitter audio
                # center_timeN = (center_frames[n] - random.random()) / self.fps
                center_timeN = (center_frames[n] - 0.5) / self.fps
                audios[n] = self._load_audio(path_audios[n], center_timeN)
            mag_mix, mags, phase_mix = self._mix_n_and_stft(audios)

        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mag_mix, mags, frames, signs, audios, phase_mix = \
                self.dummy_mix_data(N)

        ret_dict = {'mag_mix': mag_mix, 'mags': mags, 'frames': frames, 'signs': signs}
        if self.split != 'train':
            ret_dict['audios'] = audios
            ret_dict['phase_mix'] = phase_mix
            ret_dict['infos'] = infos

        return ret_dict
