# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE

import os
import sys
import random

import h5py
import numpy as np
import nibabel as nib

# search for utilities module under root dir
DIR_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DIR_ROOT = os.path.abspath(DIR_ROOT)

sys.path.insert(0, DIR_ROOT)
import utilities
sys.path.pop(0)

READER_LOG_LEVEL = "DEBUG"


class HDF5Generator:
    def __init__(self, path, keys=None, slice_dim=0, shuffle=False):
        logger = utilities.log.create_or_get_logger(
            self.__class__.__name__, level=READER_LOG_LEVEL
        )

        self._filename = str(path)
        self._keys = keys
        self._slice_dim = slice_dim
        self._shuffle = shuffle

        # basic checks
        if not os.path.exists(self._filename):
            msg = 'does not exist: ' + self._filename
            logger.error(msg)
            raise FileNotFoundError(msg)

        if self._keys is None:
            msg = '"keys" must be specified!'
            logger.error(msg)
            raise ValueError(msg)

        with h5py.File(self._filename, 'r') as hf:
            # log specified parameters and hdf5 file info
            logger.info("specified hdf5 file:")

            msg = "  " + str("filename").ljust(16)
            msg += " - " + str(self._filename)
            logger.info(msg)

            logger.debug("existing keys in specified hdf5 file:")
            for k in list(hf.keys()):
                msg = "  " + str(k).ljust(16) + " - " + str(hf[k].shape)
                logger.debug(msg)

            logger.info("specified keys:")
            for k in list(self._keys):
                msg = "  " + str(k).ljust(16) + " - " + str(hf[k].shape)
                logger.info(msg)

            logger.info("slice dimension: " + str(self._slice_dim))

            logger.info("shuffle data: " + str(self._shuffle))

            # check that all specified keys exist
            for k in list(hf.keys()):
                if k not in list(hf.keys()):
                    msg = "  " + 'key "' + str(k) + '" not found in hdf5 file!'
                    logger.error(msg)
                    raise ValueError(msg)

            # check that slicing dimension is valid
            sd_isvalid = True
            for k in self._keys:
                if self._slice_dim < 0:
                    sd_isvalid &= (abs(self._slice_dim) <= len(hf[k].shape))
                else:
                    sd_isvalid &= (self._slice_dim < len(hf[k].shape))

            if not sd_isvalid:
                msg = 'invalid slice dimension specified!'
                logger.error(msg)
                raise ValueError(msg)

            # check that datasets has same size in the slicing dimension
            nslices = [hf[k].shape[self._slice_dim] for k in self._keys]
            if len(set(nslices)) != 1:
                msg = 'specified keys correspond to datasets '
                msg += 'with different size in first dimension!'
                logger.error(msg)
                raise ValueError(msg)

            # check that datasets are not empty
            if nslices[0] == 0:
                msg = 'check that specified datasets are not empty'
                logger.error(msg)
                raise ValueError(msg)

            # store size in slice dimension as class attribute
            self._nslice = nslices[0]

    def __call__(self):
        with h5py.File(self._filename, 'r') as hf:
            # prepare slicers
            slicers = {}
            for k in self._keys:
                slicers[k] = [slice(None)] * len(hf[k].shape)

            # shuffle slice sequence if needed
            if self._shuffle:
                slice_indexes = list(range(self._nslice))
                random.shuffle(slice_indexes)
            else:
                slice_indexes = range(self._nslice)

            # loop through slices
            for islice in slice_indexes:
                # modify slicer for slice dimension
                for k in self._keys:
                    slicers[k][self._slice_dim] = slice(islice, islice + 1)

                # extract slice and yield
                s = [
                    np.squeeze(hf[k][tuple(slicers[k])], axis=self._slice_dim)
                    for k in self._keys
                ]
                yield tuple(s)

    def get_length(self):
        return self._nslice

    def get_dataset_shapes(self):
        with h5py.File(self._filename, 'r') as hf:
            shapes = {}
            for k in hf.keys():
                _shape = list(hf[k].shape)
                _shape.pop(self._slice_dim)

                shapes[k] = tuple(_shape)

        return shapes


class InterleavedGenerator:
    def __init__(self, generators, block_lengths=None, cyclic=False):
        logger = utilities.log.create_or_get_logger(
            self.__class__.__name__, level=READER_LOG_LEVEL
        )

        # make sure generators is a list
        if not isinstance(generators, (list, tuple)):
            msg = 'generators must be a list'
            logger.error(msg)
            raise ValueError(msg)

        # make sure block length is a list with same length as generators
        if block_lengths is None:
            block_lengths = [1] * len(generators)
        elif not isinstance(block_lengths, (list, tuple)):
            msg = 'block_lengths must be a list'
            logger.error(msg)
            raise ValueError(msg)

        if len(block_lengths) != len(generators):
            msg = 'block_lengths must be a list with same length as generators'
            logger.error(msg)
            raise ValueError(msg)

        # make sure all dataset shapes are the same
        dataset_shapes = None
        for generator in generators:
            if dataset_shapes is None:
                dataset_shapes = generator.get_dataset_shapes()
                continue

            # get shapes of current generator's datasets
            current_dataset_shapes = generator.get_dataset_shapes()

            # loop over dataset (i.e. keys)
            for k in dataset_shapes.keys():
                # check that the same dataset exist in the current generator
                if k not in current_dataset_shapes.keys():
                    msg = 'mismatched datasets between generator'
                    logger.error(msg)
                    raise ValueError(msg)

                # check that shape is same as first generator
                current_dataset_shape = np.array(current_dataset_shapes[k])
                dataset_shape = np.array(dataset_shapes[k])

                if not (current_dataset_shape == dataset_shape).all():
                    msg = 'mismatched dataset shapes between generator'
                    logger.error(msg)
                    raise ValueError(msg)

        self._generators = generators
        self._block_lengths = block_lengths
        self._cyclic = cyclic

    def __call__(self):
        logger = utilities.log.create_or_get_logger(self.__class__.__name__)

        # store class attributes as local variables
        generators = self._generators
        block_lengths = self._block_lengths
        cyclic = self._cyclic

        # initializing iterators and related class attributes
        iterators = [g() for g in generators]
        iterators_size = [g.get_length() for g in generators]
        iterators_pos = [0] * len(generators)
        iterators_finished = [False] * len(generators)

        while not all(finished for finished in iterators_finished):
            # loop over iterators
            for i in range(len(iterators)):
                # skip iterator if finished and cyclic is false
                if iterators_finished[i] and not cyclic:
                    continue

                # make block_lengths[i] yields with iterator
                for j in range(block_lengths[i]):
                    if iterators_pos[i] < iterators_size[i]:
                        # iterator has not reached its end; yield
                        iterators_pos[i] += 1
                        yield next(iterators[i])
                    else:
                        # iterator has reached its end
                        iterators_finished[i] = True

                        msg = "generator{:} depleted".format(i)
                        logger.debug(msg)

                        if all(finished for finished in iterators_finished):
                            # all iterators have finished, raise StopIteration
                            # or GeneratorExit with another yield
                            yield next(iterators[i])
                        elif (cyclic):
                            # not all iterators have finished,
                            # and cyclic is true;
                            # create new iterator and yield
                            iterators[i] = generators[i]()
                            iterators_size[i] = generators[i].get_length()
                            iterators_pos[i] = 0

                            iterators_pos[i] += 1
                            yield next(iterators[i])
                        else:
                            # not all iterators have finished,
                            # and cyclic is false;
                            # break to move on to another iterator

                            # try:
                            #     next(iterators[i])
                            # except:
                            #     msg = "generator{} depleted".format(i)
                            #     logger.debug(msg)

                            iterators[i] = None  # release memory
                            break

        # default fallback; should never reach this line
        logger.warn("reached forbidden line of code!")
        raise StopIteration()

    def get_length(self):
        logger = utilities.log.create_or_get_logger(self.__class__.__name__)

        msg = "get_length() not supported by InterleavedGenerator!"
        logger.error(msg)
        raise NotImplementedError(msg)

    def get_dataset_shapes(self):
        # all generators have same dataset_shapes (as checked in __init__)
        # hence, just use first generator
        return self._generators[0].get_dataset_shapes()


class NiftiGenerator:
    def __init__(
        self,
        path,
        keys=None,
        resample_orders=None,
        output_res=None,
        output_shape=None,
        slice_dim=None,
        shuffle=False
    ):
        logger = utilities.log.create_or_get_logger(
            self.__class__.__name__, level=READER_LOG_LEVEL
        )

        self._path = str(path)
        self._keys = keys
        self._resample_orders = resample_orders
        self._output_res = output_res
        self._output_shape = output_shape
        self._slice_dim = slice_dim
        self._shuffle = shuffle

        # basic checks
        if not os.path.exists(self._path):
            msg = 'does not exist: ' + self._path
            logger.error(msg)
            raise FileNotFoundError(msg)

        if self._keys is None:
            self._keys = ['.']
        elif not isinstance(self._keys, (list, tuple)):
            msg = 'keys must be a list or a tuple'
            logger.error(msg)
            raise ValueError(msg)
        else:
            self._keys = [str(s) for s in self._keys]

            for key in self._keys:
                subdir_fullpath = os.path.join(self._path, key)

                if not os.path.exists(subdir_fullpath):
                    msg = 'does not exist: ' + subdir_fullpath
                    logger.error(msg)
                    raise FileNotFoundError(msg)

        if self._resample_orders is None:
            self._resample_orders = [0] * len(self._keys)
        elif isinstance(self._resample_orders, (int, str)):
            self._resample_orders = [int(self._resample_orders)]
            self._resample_orders *= len(self._keys)
        elif isinstance(self._resample_orders, (list, tuple)):
            if len(self._resample_orders) != len(self._keys):
                msg = 'resample_orders must be same length as keys!'
                logger.error(msg)
                raise ValueError(msg)

            self._resample_orders = [int(i) for i in self._resample_orders]
        else:
            msg = 'resample_orders must either be a list/tuple '
            msg += 'or an integer'
            logger.error(msg)
            raise ValueError(msg)

        if self._output_res is not None:
            if not isinstance(self._output_res, (list, tuple)):
                msg = 'output_res must be a list or a tuple'
                logger.error(msg)
                raise ValueError(msg)
            else:
                self._output_res = list(self._output_res)

                for i, _ in enumerate(self._output_res):
                    if self._output_res[i] is not None:
                        self._output_res[i] = float(self._output_res[i])

                self._output_res = tuple(self._output_res)

        if self._output_shape is not None:
            if not isinstance(self._output_shape, (list, tuple)):
                msg = 'output_shape must be a list or a tuple'
                logger.error(msg)
                raise ValueError(msg)
            else:
                self._output_shape = [int(s) for s in self._output_shape]
                self._output_shape = tuple(self._output_shape)

        if self._slice_dim is not None:
            self._slice_dim = int(self._slice_dim)

    def __call__(self):
        logger = utilities.log.create_or_get_logger(self.__class__.__name__)

        # prepare a 2D list of fullpaths to nii, i.e.:
        # [
        #    [path_key0_nii0, path_key0_nii1, ...],
        #    [path_key1_nii0, path_key1_nii1, ...],
        #    ...
        # ]
        nii_key_lists = []
        for key in self._keys:
            dir_nii = os.path.join(self._path, key)

            nii_list = sorted(os.listdir(dir_nii))
            nii_list = [os.path.join(dir_nii, nii) for nii in nii_list]

            nii_key_lists.append(nii_list)

        # create a list of tuples of fullpaths to nii, i.e.:
        # [
        #    (path_key0_nii0, path_key1_nii0, ...),
        #    (path_key0_nii1, path_key1_nii1, ...),
        #    ...
        # ]
        nii_tuples = list(zip(*nii_key_lists))

        # make sure filename are the same
        for nii_tuple in nii_tuples:
            nii_basenames = [os.path.basename(nii) for nii in nii_tuple]

            if len(set(nii_basenames)) != 1:
                msg = "unexpected filename mismatch between keys"
                logger.error(msg)
                raise ValueError(msg)

        # shuffle tuple sequence if needed
        if self._shuffle:
            random.shuffle(nii_tuples)

        # loop over tuples
        for nii_tuple in nii_tuples:
            # load niis
            msg = "loading {:}".format(os.path.basename(nii_tuple[0]))
            logger.info(msg)

            nib_imgs = [nib.load(filepath) for filepath in nii_tuple]

            np_imgs = [nib_img.get_data() for nib_img in nib_imgs]
            # np_imgs = [np.squeeze(np_img.copy()) for np_img in np_imgs]

            for i, _ in enumerate(np_imgs):
                msg = "(shape: {:}) loaded image ({:})"
                msg = msg.format(np_imgs[i].shape, self._keys[i])
                logger.debug(msg)

            # resample to match output_res (if specified)
            if self._output_res is not None:
                for i, _ in enumerate(np_imgs):
                    msg = "(shape: {:}) resampling image ({:})"
                    msg = msg.format(np_imgs[i].shape, self._keys[i])
                    logger.debug(msg)

                    # obtain image resolution from nifti headers
                    nii_header = nib_imgs[i].header

                    # ndim = nii_header.structarr['dim'][0]
                    ndim = len(np_imgs[i].shape)
                    nii_res = nii_header.structarr['pixdim'][1:(1 + ndim)]

                    # resample image
                    np_imgs[i] = utilities.image.resample_image(
                        img_original=np_imgs[i],
                        res_original=nii_res,
                        res_target=self._output_res,
                        order=self._resample_orders[i]
                    )

                    msg = "(shape: {:}) resampled image ({:})"
                    msg = msg.format(np_imgs[i].shape, self._keys[i])
                    logger.debug(msg)

            # crop or pad image to match output_shape (if specified)
            if self._output_shape is not None:
                for i, _ in enumerate(np_imgs):
                    msg = "(shape: {:}) cropping image ({:})"
                    msg = msg.format(np_imgs[i].shape, self._keys[i])
                    logger.debug(msg)

                    # make sure slice_dim is non-negative
                    slice_dim = self._slice_dim
                    if slice_dim < 0:
                        slice_dim += len(np_imgs[i].shape)

                    # determine desired image shape
                    size_target = list(np_imgs[i].shape)

                    for j, _ in enumerate(size_target):
                        if j < slice_dim:
                            size_target[j] = self._output_shape[j]
                        elif j > slice_dim:
                            size_target[j] = self._output_shape[j - 1]

                    # crop or pad image to desired shape
                    np_imgs[i] = utilities.image.crop_or_pad_image(
                        img_original=np_imgs[i], size_target=size_target
                    )

                    msg = "(shape: {:}) cropped image ({:})"
                    msg = msg.format(np_imgs[i].shape, self._keys[i])
                    logger.debug(msg)

            # make sure shapes are same
            np_img_shapes = [np_img.shape for np_img in np_imgs]
            if len(set(np_img_shapes)) != 1:
                msg = "unexpected shape mismatch between niis"
                logger.error(msg)
                raise ValueError(msg)

            # prepare slicer and slice index
            np_img_shape = np_img_shapes[0]
            slicers = [slice(None)] * len(np_img_shape)
            slice_indexes = range(np_img_shape[self._slice_dim])

            # shuffle slice sequence if needed
            if self._shuffle:
                slice_indexes = list(slice_indexes)
                random.shuffle(slice_indexes)

            # slice and yield
            for islice in slice_indexes:
                slicers[self._slice_dim] = slice(islice, islice + 1, 1)

                s = [
                    np.squeeze(np_img[tuple(slicers)], axis=self._slice_dim)
                    for np_img in np_imgs
                ]
                yield tuple(s)

            # release memory
            np_imgs = []
            nib_imgs = []

    def get_length(self):
        logger = utilities.log.create_or_get_logger(self.__class__.__name__)

        msg = "get_length() not supported by NiftiGenerator!"
        logger.error(msg)
        raise NotImplementedError(msg)

    def get_dataset_shapes(self):
        if self._output_shape is not None:
            output_shape = tuple(self._output_shape)
        else:
            output_shape = None

        return {str(key): output_shape for key in self._keys}
