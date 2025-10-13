from pathlib import Path

from torch.utils.data import Dataset
import torch
import SimpleITK as sitk
import torchio as tio
import random
import torch.nn.functional as F
import os

sitk.ProcessObject.SetGlobalDefaultThreader("Platform")
from multiprocessing import Manager


def Train(csv, cfg, preload=True, patches=False, patch_size=None):
    if patches:
        assert patch_size, "patch_size cannot be None when patches is True"
    subjects = []
    for _, sub in csv.iterrows():
        subject_dict = {
            'vol': tio.ScalarImage(sub.img_path, reader=sitk_reader),
            'age': sub.age,
            'ID': sub.img_name,
            'label': sub.label,
            'Dataset': sub.setname,
            'stage': sub.settype,
            'path': sub.img_path
        }
        if sub.mask_path != None:  # if we have masks
            subject_dict['mask'] = tio.LabelMap(sub.mask_path, reader=sitk_reader)
        else:  # if we don't have masks, we create a mask from the image
            subject_dict['mask'] = tio.LabelMap(tensor=tio.ScalarImage(sub.img_path, reader=sitk_reader).data > 0)

        subject = tio.Subject(subject_dict)
        subjects.append(subject)

    if preload:
        manager = Manager()
        cache = DatasetCache(manager)
        ds = tio.SubjectsDataset(subjects, transform=get_transform(cfg))
        ds = preload_wrapper(ds, cache, augment=get_augment(cfg))
    else:
        ds = tio.SubjectsDataset(subjects, transform=tio.Compose([get_transform(cfg), get_augment(cfg)]))

    if cfg.spatialDims == '2D':
        slice_ind = cfg.get('startslice', None)
        seq_slices = cfg.get('sequentialslices', None)
        if not patches:
            ds = vol2slice(ds, cfg, slice=slice_ind, seq_slices=seq_slices)
        else:
            ds = ContrastiveAnchorPositiveDataset(ds,cfg,patch_size=patch_size)
    return ds


def Eval(csv, cfg):
    subjects = []
    for _, sub in csv.iterrows():
        if sub.mask_path is not None and tio.ScalarImage(sub.img_path, reader=sitk_reader).shape != tio.ScalarImage(
                sub.mask_path, reader=sitk_reader).shape:
            print(
                f'different shapes of vol and mask detected. Shape vol: {tio.ScalarImage(sub.img_path, reader=sitk_reader).shape}, shape mask: {tio.ScalarImage(sub.mask_path, reader=sitk_reader).shape} \nsamples will be resampled to the same dimension')

        subject_dict = {
            'vol': tio.ScalarImage(sub.img_path, reader=sitk_reader),
            'vol_orig': tio.ScalarImage(sub.img_path, reader=sitk_reader),
            # we need the image in original size for evaluation
            'age': sub.age,
            'ID': sub.img_name,
            'label': sub.label,
            'Dataset': sub.setname,
            'stage': sub.settype,
            'seg_available': False,
            'path': sub.img_path}
        if sub.seg_path != None:  # if we have segmentations
            subject_dict['seg'] = tio.LabelMap(sub.seg_path, reader=sitk_reader),
            subject_dict['seg_orig'] = tio.LabelMap(sub.seg_path,
                                                    reader=sitk_reader)  # we need the image in original size for evaluation
            subject_dict['seg_available'] = True
        if sub.mask_path != None:  # if we have masks
            subject_dict['mask'] = tio.LabelMap(sub.mask_path, reader=sitk_reader)
            subject_dict['mask_orig'] = tio.LabelMap(sub.mask_path,
                                                     reader=sitk_reader)  # we need the image in original size for evaluation
        else:
            tens = tio.ScalarImage(sub.img_path, reader=sitk_reader).data > 0
            subject_dict['mask'] = tio.LabelMap(tensor=tens)
            subject_dict['mask_orig'] = tio.LabelMap(tensor=tens)

        subject = tio.Subject(subject_dict)
        subjects.append(subject)
    ds = tio.SubjectsDataset(subjects, transform=get_transform(cfg))
    return ds


## got it from https://discuss.pytorch.org/t/best-practice-to-cache-the-entire-dataset-during-first-epoch/19608/12
class DatasetCache(object):
    def __init__(self, manager, use_cache=True):
        self.use_cache = use_cache
        self.manager = manager
        self._dict = manager.dict()

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return str(key) in self._dict

    def reset(self):
        self._dict.clear()

    def get(self, key):
        if not self.use_cache:
            raise AttributeError('Data caching is disabled and get funciton is unavailable! Check your config.')
        return self._dict[str(key)]

    def cache(self, key, subject):
        # only store if full data in memory is enabled
        if not self.use_cache:
            return
        # only store if not already cached
        if str(key) in self._dict:
            return
        self._dict[str(key)] = (subject)


class preload_wrapper(Dataset):
    def __init__(self, ds, cache, augment=None):
        self.cache = cache
        self.ds = ds
        self.augment = augment

    def reset_memory(self):
        self.cache.reset()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        if self.cache.is_cached(index):
            subject = self.cache.get(index)
        else:
            subject = self.ds.__getitem__(index)
            self.cache.cache(index, subject)
        if self.augment:
            subject = self.augment(subject)
        return subject



class RandomPopper:
    def __init__(self):
        self.indexes = None

    def set_indexes(self, low: int, high: int):
        self.indexes = list(range(low, high))  # Create range [low, high)

    def pop_random(self, low: int, high: int) -> torch.Tensor:
        if not self.indexes:  # Check if None or empty
            self.set_indexes(low, high)
        # Pop a random value
        value = self.indexes.pop(random.randint(0, len(self.indexes) - 1))
        return torch.tensor([value], dtype=torch.int)


class vol2slice(Dataset):
    def __init__(self, ds, cfg, onlyBrain=False, slice=None, seq_slices=None):
        self.ds = ds
        self.onlyBrain = onlyBrain
        self.slice = slice
        self.seq_slices = seq_slices
        self.counter = 0
        self.ind = None
        self.cfg = cfg
        self.random_popper = RandomPopper()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        subject = self.ds.__getitem__(index)
        if self.onlyBrain:
            start_ind = None
            for i in range(subject['vol'].data.shape[-1]):
                if subject['mask'].data[0, :, :, i].any() and start_ind is None:  # only do this once
                    start_ind = i
                if not subject['mask'].data[0, :, :,
                       i].any() and start_ind is not None:  # only do this when start_ind is set
                    stop_ind = i
            low = start_ind
            high = stop_ind
        else:
            low = 0
            high = subject['vol'].data.shape[-1]
        if self.slice is not None:
            self.ind = self.slice
            if self.seq_slices is not None:
                low = self.ind
                high = self.ind + self.seq_slices
                self.ind = torch.randint(low, high, size=[1])
        else:
            if self.cfg.get('unique_slice', False):  # if all slices in one batch need to be at the same location
                if self.counter % self.cfg.batch_size == 0 or self.ind is None:  # only change the index when changing to new batch
                    self.ind = torch.randint(low, high, size=[1])
                self.counter = self.counter + 1
            else:
                # self.ind = torch.randint(low, high, size=[1])
                self.ind = self.random_popper.pop_random(low,high)

        subject['ind'] = self.ind
        subject['total_slices'] = high

        subject['vol'].data = subject['vol'].data[..., self.ind]
        subject['mask'].data = subject['mask'].data[..., self.ind]

        return subject


def get_transform(cfg):  # only transforms that are applied once before preloading
    h, w, d = tuple(cfg.get('imageDim', (160, 192, 160)))

    if not cfg.resizedEvaluation:
        exclude_from_resampling = ['vol_orig', 'mask_orig', 'seg_orig']
    else:
        exclude_from_resampling = None
    mode = cfg.get('mode')
    landmark_path = Path(__file__).resolve().parent.parent.parent / 'landmarks' / f'{mode}_landmarks.pt'
    preprocess = tio.Compose([
        tio.transforms.HistogramStandardization(landmark_path),
        tio.CropOrPad((h, w, d), padding_mode=0),
        tio.RescaleIntensity((0, 1), percentiles=(cfg.get('perc_low', 1), cfg.get('perc_high', 99)),
                             masking_method='mask'),
        tio.Resample(cfg.get('rescaleFactor', 3.0), image_interpolation='bspline', exclude=exclude_from_resampling),
    ])

    return preprocess


def get_augment(cfg):  # augmentations that may change every epoch
    augmentations = []
    # individual augmentations
    if cfg.get('random_bias', False):
        augmentations.append(tio.RandomBiasField(p=0.25))
    if cfg.get('random_motion', False):
        augmentations.append(tio.RandomMotion(p=0.1))
    if cfg.get('random_noise', False):
        augmentations.append(tio.RandomNoise(p=0.5))
    if cfg.get('random_ghosting', False):
        augmentations.append(tio.RandomGhosting(p=0.5))
    if cfg.get('random_blur', False):
        augmentations.append(tio.RandomBlur(p=0.5))
    if cfg.get('random_gamma', False):
        augmentations.append(tio.RandomGamma(p=0.5))
    if cfg.get('random_elastic', False):
        augmentations.append(tio.RandomElasticDeformation(p=0.5))
    if cfg.get('random_affine', False):
        augmentations.append(tio.RandomAffine(p=0.5))
    if cfg.get('random_flip', False):
        augmentations.append(tio.RandomFlip(p=0.5))
    if cfg.get('random_biasfield', False):
        augmentations.append(tio.RandomBiasField())
    if cfg.get('random_rescale', False):
        augmentations.append(tio.RescaleIntensity(0, 1))
    if cfg.get('random_spike', False):
        augmentations.append(tio.RandomSpike())

    # policies/groups of augmentations
    if cfg.get('aug_intensity', False):  # augmentations that change the intensity of the image rather than the geometry
        augmentations.append(tio.RandomGamma(p=0.5))
        augmentations.append(tio.RandomBiasField(p=0.25))
        augmentations.append(tio.RandomBlur(p=0.25))
        augmentations.append(tio.RandomGhosting(p=0.5))

    augment = tio.Compose(augmentations)
    return augment

class ContrastiveAnchorPositiveDataset(Dataset):
    def __init__(self, ds, cfg, patch_size=32):
        self.ds = ds
        self.cfg = cfg
        self.patch_size = patch_size
        self.valid_slices = self._cache_valid_slices()

    def _cache_valid_slices(self):
        """Cache valid slices (i.e., containing brain) per subject."""
        valid = {}
        for i, subj in enumerate(self.ds):
            mask = subj['mask'].data[0]
            z_valid = [z for z in range(mask.shape[-1]) if mask[:, :, z].any()]
            if z_valid:
                valid[i] = z_valid
        return valid

    def __len__(self):
        return len(self.ds)

    def _get_patch(self, slice_img, y, x):
        hs = ws = self.patch_size // 2
        padded = F.pad(slice_img, (ws, ws, hs, hs), mode='constant', value=0)
        y += hs
        x += ws
        return padded[:, y - hs:y + hs, x - ws:x + ws]

    def __getitem__(self, index):
        subj = self.ds[index]
        valid_z = self.valid_slices.get(index)
        if not valid_z:
            return self.__getitem__((index + 1) % len(self.ds))  # fallback

        z = random.choice(valid_z)
        mask = subj['mask'].data[0][:, :, z]
        coords = torch.nonzero(mask, as_tuple=False)

        if coords.numel() == 0:
            return self.__getitem__((index + 1) % len(self.ds))

        y, x = random.choice(coords).tolist()
        anchor = self._get_patch(subj['vol'].data[0][:, :, z].unsqueeze(0), y, x)

        positive = None
        # Positives: same location across subjects
        for j, other in enumerate(self.ds):
            if j == index: continue
            if z >= other['mask'].data.shape[-1]: continue
            if other['mask'].data[0][y, x, z]:
                patch = self._get_patch(other['vol'].data[0][:, :, z].unsqueeze(0), y, x)
                positive = patch
                break
            
        if positive == None:
            return self.__getitem__((index + 1) % len(self.ds))
        return {
            'anchor': anchor,
            'positive': positive,
            'center': (z, y, x)
        }



def sitk_reader(path):
    image_nii = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if not 'mask' in str(path) and not 'seg' in str(path):  # only for volumes / scalar images
        image_nii = sitk.CurvatureFlow(image1=image_nii, timeStep=0.125, numberOfIterations=3)
    vol = sitk.GetArrayFromImage(image_nii).transpose(2, 1, 0)
    return vol, None


