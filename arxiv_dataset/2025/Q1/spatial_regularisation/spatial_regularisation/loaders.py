import math
import copy
import torch
import numpy as np
import torch.utils.data
import numpy.random as npr

from spatial_regularisation.utils import preprocess
from spatial_regularisation.augmenters import SpatialAugmenter, IntensityAugmenter


class LoaderRXFM(torch.utils.data.IterableDataset):
    """Implements a generator that will give training example at each minibatch to the rigid transform estimator
    network in pytorch."""

    def __init__(self,
                 subj_dict,
                 augm_params=None,
                 use_same_subject=False,
                 validator_mode=False,
                 return_masks=False,
                 return_clean_images=False,
                 steps_per_epoch=None):
        """
        :param subj_dict: dictionary of the form {'im basename': [path_im, path_mask]} as built by the function
        utils.build_subject_dict. the 'path_mask' entry is optional.
        :param augm_params: dictionary containing values for the preprocessing and augmentation:
            augment_params = {'resize': image_size,             # resize input images to this size with padding/cropping
                              'rotation_range': rotation_range, # maximum rotation angle for augmentation (in degrees)
                              'shift_range': shift_range,       # maximum shift for augmentation (in voxels)
                              'max_noise_std': max_noise_std,   # maximum standard deviation for the Gaussian noise
                              'max_bias_std': max_bias_std,     # maximum std. dev for the bias filed corruption
                              "bias_scale": bias_scale,         # scale of the bias field (lower = smoother)
                              "gamma_std": gamma_std}           # std dev for random exponentiation (higher = stronger)
        :param validator_mode: whether to build a loader for validation data. In this case, each validation example
        will be augmented the same way for each validation step
        :param return_masks: whether to return masks as additional volumes at each minibatch. These must be given in
        subj_dict.
        :param return_clean_images: whether to return the input example before intensity augmentation as
        additional volumes at each minibatch (useful to compute image loss).
        :param steps_per_epoch: number of mini-batches per epoch. Default is None, which corresponds to the number of
        training images.
        """

        # input data
        self.subj_dict = subj_dict
        self.list_of_subjects = list(self.subj_dict.keys())
        self.n_samples = len(self.list_of_subjects)

        self.use_same_subject = use_same_subject
        self.validator_mode = validator_mode

        # define iteration number
        self.steps = steps_per_epoch if steps_per_epoch is not None and not self.validator_mode else self.n_samples
        self.max_iter = int(self.steps / 2) if self.validator_mode and not self.use_same_subject else self.steps
        self.iter_idx = 0

        # outputs
        self.return_masks = return_masks
        self.return_clean_images = return_clean_images

        # initialise resize/rescale functions
        resize = augm_params["resize"]
        self.preproc_func = lambda x: preprocess(x, normalise=True, resize=resize, dtype='float32')
        self.preproc_func_labels = lambda x: preprocess(x, normalise=False, resize=resize, dtype='int32')

        # numpy seed
        self.rng = npr.RandomState(0)

        # load, resize, rescale images/labels (still numpy with size [H, W, D, C])
        self.samples = {}  # {'im basename': [im, mask]}
        for subj in self.list_of_subjects:
            self.samples[subj] = self.load_sample(self.subj_dict[subj])

        # get augmentation parameters if 1) in validation mode 2) using the same subject 3) not already given
        augment_discrete = []
        if self.validator_mode and self.use_same_subject:
            bias_sample_size = [math.ceil(size * augm_params["bias_scale"]) for size in resize]
            for _ in self.list_of_subjects:
                r = self.rng.uniform(-augm_params["rotation_range"], augm_params["rotation_range"], 3).tolist()
                t = self.rng.uniform(-augm_params["shift_range"], augm_params["shift_range"], 3).tolist()
                s = self.rng.uniform(1 - augm_params["scale_range"], 1 + augm_params["scale_range"], 3).tolist()
                sh = self.rng.uniform(-augm_params["shear_range"], augm_params["shear_range"], 6).tolist()
                if augm_params["max_noise_std"] > 0:
                    noise = self.rng.normal(0, self.rng.uniform(high=augm_params["max_noise_std"]), resize)
                else:
                    noise = None
                if augm_params["max_bias_std"] > 0:
                    bias = self.rng.normal(0, self.rng.uniform(high=augm_params["max_bias_std"]), bias_sample_size)
                else:
                    bias = None
                if augm_params["gamma_std"] > 0:
                    gamma = self.rng.normal(0, augm_params["gamma_std"])
                else:
                    gamma = None
                augment_discrete.append({"rotation": r,
                                         "translation": t,
                                         "scale": s,
                                         "shear": sh,
                                         "noise_field": noise,
                                         "bias_field": bias,
                                         "gamma": gamma})

        # initialise spatial/intensity augmenters
        self.spatial_augmenter = SpatialAugmenter(list_of_xfm_params=augment_discrete,
                                                  rotation_range=augm_params["rotation_range"],
                                                  shift_range=augm_params["shift_range"],
                                                  scale_range=augm_params["scale_range"],
                                                  shear_range=augm_params["shear_range"],
                                                  return_affine=True,
                                                  normalise=True)
        self.intensity_augmenter = IntensityAugmenter(list_of_params=augment_discrete,
                                                      max_noise_std=augm_params["max_noise_std"],
                                                      max_bias_std=augm_params["max_bias_std"],
                                                      bias_scale=augm_params["bias_scale"],
                                                      gamma_std=augm_params["gamma_std"])

        # output format params
        self.output_names = ["scan_moving", "scan_fixed", "xfm"]
        if self.return_masks:
            self.output_names += ["mask_moving", "mask_fixed"]
        if self.return_clean_images:
            self.output_names += ["clean_scan_moving", "clean_scan_fixed"]

    def load_sample(self, sample_tuple):
        sample = [self.preproc_func(sample_tuple[0])]  # [H, W, D, C]
        if self.return_masks:
            sample.append(self.preproc_func_labels(sample_tuple[1]))
        return sample

    def __next__(self):

        if self.iter_idx >= self.max_iter:
            self.iter_idx = 0  # reset at the end of every epoch
            raise StopIteration

        # get moving image [frame], possibly with its segmentation [frame, mask]
        idx = self.iter_idx if self.validator_mode else np.random.choice(self.n_samples)
        frame_mask_moving = copy.deepcopy(self.samples[self.list_of_subjects[idx]])  # frame size [H, W, D, C]

        # get fixed image [frame], possibly with its segmentation [frame, mask]
        if self.use_same_subject:
            frame_mask_fixed = copy.deepcopy(frame_mask_moving)
        else:
            idx = self.iter_idx + int(self.n_samples / 2) if self.validator_mode else np.random.choice(self.n_samples)
            frame_mask_fixed = copy.deepcopy(self.samples[self.list_of_subjects[idx]])  # frame size [H, W, D, C]

        # spatial augment (only when training or when validating with the same subject)
        if not self.validator_mode:
            frame_mask_moving, xfm_moving = self.spatial_augmenter.random_transform(*frame_mask_moving)
            frame_mask_fixed, xfm_fixed = self.spatial_augmenter.random_transform(*frame_mask_fixed)
        elif self.validator_mode and self.use_same_subject:
            xfm_moving = np.eye(4)
            frame_mask_fixed, xfm_fixed = self.spatial_augmenter.predefined_transform(idx, *frame_mask_fixed)
        else:
            xfm_moving = xfm_fixed = np.eye(4)
        xfm = xfm_fixed.astype('float32') @ np.linalg.inv(xfm_moving.astype('float32'))

        # get clean frames (ie not yet augmented for intensity)
        clean_frame_moving = frame_mask_moving[0]
        clean_frame_fixed = frame_mask_fixed[0]

        # intensity augment (only when training or when validating with the same subject)
        if not self.validator_mode:
            frame_mask_moving[0] = self.intensity_augmenter.random_transform(frame_mask_moving[0])
            frame_mask_fixed[0] = self.intensity_augmenter.random_transform(frame_mask_fixed[0])
        elif self.validator_mode and self.use_same_subject:
            frame_mask_fixed[0] = self.intensity_augmenter.predefined_transform(idx, frame_mask_fixed[0])

        # group outputs in dict
        output_dict = {}
        outputs = [np.rollaxis(frame_mask_moving[0], 3, 0).astype(np.float32),  # [C, H, W, D]
                   np.rollaxis(frame_mask_fixed[0], 3, 0).astype(np.float32),
                   xfm.astype(np.float32)]
        if self.return_masks:
            outputs += [np.rollaxis(frame_mask_moving[1], 3, 0).astype(np.float32),
                        np.rollaxis(frame_mask_fixed[1], 3, 0).astype(np.float32)]
        if self.return_clean_images:
            outputs += [np.rollaxis(clean_frame_moving, 3, 0).astype(np.float32),
                        np.rollaxis(clean_frame_fixed, 3, 0).astype(np.float32)]
        for name, output in zip(self.output_names, outputs):
            output_dict[name] = torch.tensor(output)  # now tensor (still of shape [C, H, W, D])

        self.iter_idx += 1
        return output_dict

    def __iter__(self):
        self.iter_idx = 0  # reset at the start of every epoch
        return self

    def __len__(self):
        return self.steps

    def next(self):
        return self.__next__()


class LoaderTesting(torch.utils.data.IterableDataset):
    """Implements a generator to feed the framework for rigid transform estimation between 2 images with testing data.
    This is the case where we have pairs that have been simulated beforehand, so here we take as inputs:
    moving images, fixed images and gt transforms"""

    def __init__(self,
                 subj_dict_moving,
                 subj_dict_fixed,
                 resize,
                 return_masks,
                 dict_xfm=None):
        """
        :param subj_dict_moving: dictionary of the form {'im basename': [path_im, path_mask]} as built by the function
        utils.build_subject_dict. This is for the moving images. The 'path_mask' entry is optional.
        :param subj_dict_fixed: same as above but for the fixed images
        :param resize: resize input images to this size with padding/cropping, list of image shape [H, W, D]
        :param return_masks: whether to return masks as additional volumes at each minibatch. These must be given in
        subj_dict_1 and subj_dict_2.
        :param dict_xfm: (optional) dictionary of the form {'im basename': path_gt_xfm}, where path_gt_xfm contains GT transforms
        to go from image 1 to image 2.
        """

        # input data
        self.subj_dict_moving = subj_dict_moving
        self.subj_dict_fixed = subj_dict_fixed
        self.dict_xfm = dict_xfm
        self.return_masks = return_masks
        self.list_of_subjects_fixed= list(self.subj_dict_fixed.keys())
        self.list_of_subjects_moving = list(self.subj_dict_moving.keys())
        assert len(self.list_of_subjects_fixed) == len(self.list_of_subjects_moving), \
            'not same number of subjects, had %d fixed and %d moving' % (len(self.list_of_subjects_fixed), len(self.list_of_subjects_moving))
        self.n_samples = len(self.list_of_subjects_fixed)
        self.subject_idx = -1

        # initialise resize/rescale functions
        self.preproc_func = lambda x: preprocess(x, normalise=True, resize=resize, dtype='float32')
        self.preproc_func_labels = lambda x: preprocess(x, normalise=False, resize=resize, dtype='int32')

        # output format params
        self.output_names = ["scan_moving", "scan_fixed"]
        if self.dict_xfm is not None:
            self.output_names += ["xfm"]
        if self.return_masks:
            self.output_names += ["mask_moving", "mask_fixed"]

    def load_sample(self, sample_tuple):
        sample = [self.preproc_func(sample_tuple[0])]  # [H, W, D, C]
        if self.return_masks:
            sample.append(self.preproc_func_labels(sample_tuple[1]))
        return sample

    def __next__(self):

        self.subject_idx += 1
        if self.subject_idx >= self.n_samples:
            raise StopIteration

        # load data for first and current datapoints
        frame_mask_moving = self.load_sample(self.subj_dict_moving[self.list_of_subjects_moving[self.subject_idx]])
        frame_mask_fixed = self.load_sample(self.subj_dict_fixed[self.list_of_subjects_fixed[self.subject_idx]])

        # group outputs in dict
        output_dict = {}
        outputs = [np.rollaxis(frame_mask_moving[0], 3, 0).astype(np.float32),  # [C, H, W, D]
                   np.rollaxis(frame_mask_fixed[0], 3, 0).astype(np.float32)]
        if self.dict_xfm is not None:
            outputs += [np.load(self.dict_xfm[self.list_of_subjects_fixed[self.subject_idx]]).astype(np.float32)]
        if self.return_masks:
            outputs += [np.rollaxis(frame_mask_moving[1], 3, 0).astype(np.float32),
                        np.rollaxis(frame_mask_fixed[1], 3, 0).astype(np.float32)]
        for name, output in zip(self.output_names, outputs):
            output_dict[name] = torch.tensor(output)  # now tensor (still of shape [C, H, W, D])

        return output_dict

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_samples

    def next(self):
        return self.__next__()


class LoaderTimeSeries(torch.utils.data.IterableDataset):
    """Implements a generator for feeding the testing data to the framework for rigid transform estimation between
    2 images. This is the case of real time-series, where the GT transforms might not been known.
    This takes as inputs lists of time frames, where we register every time frame to the first of the series"""

    def __init__(self,
                 subj_dict,
                 resize,
                 return_masks,
                 dict_xfm=None,
                 use_consecutive_frames=False):
        """
        :param subj_dict: dictionary of the form built in utils.build_subject_dict_time_series:
        {'time_series_1_dir': [[image_0, label_0], [image_1, label_1], ...]}
        :param resize: resize input images to this size with padding/cropping, list of image shape [H, W, D]
        :param dict_xfm: dictionary of the form built in utils.build_xfm_dict_time_series:
        {'im basename': [path_xfm_1, path_xfm_2, ...]} which contain N-1 GT transforms from image i to image 0
        :param use_consecutive_frames: whether to consider the reference as the first or the i-1 frame of the series
        """

        # input data
        self.subj_dict = subj_dict
        self.dict_xfm = dict_xfm
        self.return_masks = return_masks
        self.list_of_subjects = list(self.subj_dict.keys())
        self.subject_idx = 0
        self.frame_idx = 0
        self.use_consecutive_frames = use_consecutive_frames

        # initialise resize/rescale functions
        self.preproc_func = lambda x: preprocess(x, normalise=True, resize=resize, return_aff=True, dtype='float32')
        self.preproc_func_labels = lambda x: preprocess(x, normalise=False, resize=resize, dtype='int32')

        # output format params
        self.output_names = ["scan_moving", "scan_fixed", "aff"]
        if self.dict_xfm is not None:
            self.output_names.append("xfm_gt")
        if self.return_masks:
            self.output_names += ["mask_moving", "mask_fixed"]

    def load_sample(self, sample_tuple):
        sample = list(self.preproc_func(sample_tuple[0]))  # [H, W, D, C]
        if self.return_masks:
            sample.append(self.preproc_func_labels(sample_tuple[1]))
        return sample

    def load_labels_reg(self, sample_tuple):
        labels = self.preproc_func_labels(sample_tuple[0])
        return labels

    def __next__(self):

        self.frame_idx += 1
        if self.frame_idx == len(self.subj_dict[self.list_of_subjects[self.subject_idx]]):
            if self.subject_idx < (len(self.list_of_subjects) - 1):
                self.subject_idx += 1
                self.frame_idx = 1
                print('')
            else:
                raise StopIteration

        # load data for first and current datapoints
        if self.use_consecutive_frames:
            frame_idx_ref = self.frame_idx - 1
        else:
            frame_idx_ref = 0
        frame_mask_moving = self.load_sample(self.subj_dict[self.list_of_subjects[self.subject_idx]][self.frame_idx])
        frame_mask_fixed = self.load_sample(self.subj_dict[self.list_of_subjects[self.subject_idx]][frame_idx_ref])

        # group outputs in dict
        output_dict = {}
        outputs = [np.rollaxis(frame_mask_moving[0], 3, 0).astype(np.float32),  # [C, H, W, D]
                   np.rollaxis(frame_mask_fixed[0], 3, 0).astype(np.float32),
                   frame_mask_moving[1]]                                        # aff to save the images
        if self.dict_xfm is not None:
            xfm_gt = np.load(self.dict_xfm[self.list_of_subjects[self.subject_idx]][self.frame_idx])  # only n-1 xfm
            outputs.append(xfm_gt.astype(np.float32))
        if self.return_masks:
            outputs += [np.rollaxis(frame_mask_moving[2], 3, 0).astype(np.float32),
                        np.rollaxis(frame_mask_fixed[2], 3, 0).astype(np.float32)]
        for name, output in zip(self.output_names, outputs):
            output_dict[name] = torch.tensor(output)  # now tensor (still of shape [C, H, W, D])

        return output_dict

    def __iter__(self):
        return self

    def __len__(self):
        n_iterations = 0
        for subj in self.list_of_subjects:
            n_iterations += len(self.subj_dict[subj])
        return n_iterations

    def next(self):
        return self.__next__()
