# Written by Chun Kit Wong and CIRC under MIT license:
# https://github.com/wong-ck/DeepSegment/blob/master/LICENSE

import gc
import os
import sys
# import time

import threading
if sys.version_info[0] == 2:
    from Queue import Queue
else:
    from queue import Queue

import numpy as np
import h5py
import nibabel as nib

# search for utilities module under root dir
DIR_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DIR_ROOT = os.path.abspath(DIR_ROOT)

sys.path.insert(0, DIR_ROOT)
import utilities
sys.path.pop(0)

WRITER_LOG_LEVEL = "DEBUG"


class HDF5Writer:
    def __init__(self, path, write_freq=10, resize_chunk=100, **kwargs):
        self._logger = utilities.log.create_or_get_logger(
            self.__class__.__name__, level=WRITER_LOG_LEVEL
        )

        if write_freq > resize_chunk:
            msg = "write_freq cannot be greater than resize_chunk!"
            self._logger.error(msg)
            raise ValueError(msg)

        self._filename = path
        self._write_freq = write_freq
        self._resize_chunk = resize_chunk

        self._h5file = None
        self._datasets = {}
        self._counters = {}
        self._caches = {}

        return

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        try:
            self.close()
        except:
            pass

        return

    def __del__(self):
        try:
            self.close()
        except:
            pass

        return

    def close(self):
        if self._h5file is not None:
            for k, _ in self._caches.items():
                self._flush_dataset(k)
                self._prune_dataset(k)

        self._close_file()

    def create_dataset(self, name, shape=None, dtype=None, data=None):
        if self._h5file is None:
            self._open_file()

        self._datasets[name] = self._h5file.create_dataset(
            name,
            shape=[1] + list(shape),
            maxshape=[None] + list(shape),
            dtype=dtype,
            data=data
        )

        self._counters[name] = 0
        self._caches[name] = []

    def write_data(self, data, name):
        self._caches[name].append(data)

        if (len(self._caches[name]) >= self._write_freq):
            self._flush_dataset(name)

    def _open_file(self):
        logger = self._logger

        if (self._filename is None) or (self._filename == ""):
            msg = "Output filename must be specified prior to opening file!"
            logger.error(msg)
            raise ValueError(msg)

        # create output directory if not exists
        _dirname = os.path.dirname(self._filename)
        if not os.path.exists(_dirname):
            os.makedirs(_dirname)

        self._h5file = h5py.File(self._filename, "w")
        self._datasets = {}
        self._counters = {}
        self._caches = {}

    def _close_file(self):
        if self._h5file is not None:
            self._h5file.close()

        self._h5file = None
        self._filename = ""
        self._datasets = {}
        self._counters = {}
        self._caches = {}

    def _prune_dataset(self, name):
        logger = self._logger

        dataset_size = self._datasets[name].shape
        prune_size = self._counters[name]
        if dataset_size[0] <= prune_size:
            return

        msg = "pruning {:} from {:} to {:}"
        msg = msg.format(name, dataset_size[0], prune_size)
        logger.debug(msg)

        self._datasets[name].resize(prune_size, axis=0)

    def _flush_dataset(self, name):
        logger = self._logger

        start = self._counters[name]
        end = start + len(self._caches[name])
        if (start == end):
            return

        msg = "flushing {:} from {:} to {:}".format(name, start, end)
        logger.debug(msg)

        # resize dataset if too small
        current_size = self._datasets[name].shape
        if end > current_size[0]:
            new_size = current_size[0] + self._resize_chunk
            self._datasets[name].resize(new_size, axis=0)

        self._datasets[name][start:end, ...] = self._caches[name]

        self._counters[name] = end
        self._caches[name] = []
        gc.collect()


class NiftiWriter:
    def __init__(
        self,
        path_output,
        path_refnii,
        input_res=None,
        input_shape=None,
        slice_dim=None,
        nchannels=None,
        resample_order=0,
        dtype=None,
        async_write=True,
        max_async_write=3,
        **kwargs
    ):
        self._logger = utilities.log.create_or_get_logger(
            self.__class__.__name__, level=WRITER_LOG_LEVEL
        )

        self._dir_output = path_output
        self._dir_refnii = path_refnii
        self._input_res = input_res
        self._input_shape = input_shape
        self._slice_dim = slice_dim
        self._nchannels = nchannels
        self._resample_order = resample_order
        self._dtype = dtype
        self._async_write = async_write
        self._max_async_write = max_async_write

        self._filepath_refniis = []
        for f in list(sorted(os.listdir(self._dir_refnii))):
            self._filepath_refniis += [os.path.join(self._dir_refnii, f)]

        self._counter_refnii = 0
        self._counter_slice = -1
        self._cache = None
        self._slicer = None
        self._nslices = 0
        self._shape_slice = None
        self._shape_pre_crop = None

        # create queue and background threads
        # for asynchronous writes
        self._writequeue = None
        if self._async_write:
            self._writequeue = Queue(maxsize=self._max_async_write)
            for thread_id in range(self._max_async_write):
                write_thread = threading.Thread(
                    target=NiftiWriter._async_write_worker_static,
                    args=(self._writequeue, thread_id)
                )
                write_thread.setDaemon(True)
                write_thread.start()

        return

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        try:
            self.close()
        except:
            pass

        return

    def __del__(self):
        try:
            self.close()
        except:
            pass

        return

    def close(self):
        logger = self._logger

        if self._async_write:
            if not self._writequeue.empty():
                msg = "waiting for writing threads to finish before closing"
                logger.info(msg)

            self._writequeue.join()

        if self._counter_refnii != len(self._filepath_refniis):
            msg = "writer closed before all reference niis are used"
            logger.warn(msg)

        self._cache = None

    def write_data(self, data):
        # check if new cache need to be prepared
        if self._counter_slice < 0:
            self._prepare_next_cache()

        # write slice to cache
        if self._slice_dim is not None:
            slice_dim = self._slice_dim

            if (self._nchannels is not None) and (slice_dim < 0):
                slice_dim -= 1

            self._slicer[slice_dim] = slice(
                self._counter_slice, self._counter_slice + 1, None
            )

        self._cache[tuple(self._slicer)] = np.reshape(data, self._shape_slice)

        # increment slice counter
        self._counter_slice += 1

        # if cache is full
        # - flush cache
        # - reset counters['slice'] to -1 (handled in _flush_cache())
        # - increment counters['refnii'] by 1 (handled in _flush_cache())
        if self._counter_slice >= self._nslices:
            self._flush_cache()

        return

    def _prepare_next_cache(self):
        logger = self._logger

        # check that current counter is not out of range
        if self._counter_refnii >= len(self._filepath_refniis):
            msg = "all reference niis had been used up!"
            logger.error(msg)
            raise IndexError(msg)

        # load nifti
        filepath_refnii = self._filepath_refniis[self._counter_refnii]
        nib_refnii = nib.load(filepath_refnii)

        # determine data type
        if self._dtype is None:
            dtype = nib_refnii.get_data_dtype()
        else:
            dtype = self._dtype

        # determine nii shape
        nii_dim = nib_refnii.header.structarr['dim'][0]
        nii_shape = tuple(nib_refnii.header.structarr['dim'][1:(1 + nii_dim)])

        # create empty image from nii shape
        np_img = np.zeros(shape=nii_shape, dtype=dtype)
        # np_img = np.squeeze(np_img)

        msg = "(shape: {:}) empty image created from {:}"
        msg = msg.format(nii_shape, os.path.basename(filepath_refnii))
        logger.debug(msg)

        # make sure slice_dim is non-negative
        slice_dim = self._slice_dim
        if (slice_dim is not None) and (slice_dim < 0):
            slice_dim += len(np_img.shape)

        # pre-process array to get correct shape
        # (1) resample to match input_res (if specified)
        if self._input_res is not None:
            # obtain image resolution from nifti headers
            ndim = len(np_img.shape)
            nii_res = nib_refnii.header.structarr['pixdim'][1:(1 + ndim)]

            # resample image
            np_img = utilities.image.resample_image(
                img_original=np_img,
                res_original=nii_res,
                res_target=self._input_res,
                order=0
            )

            msg = "(shape: {:}) resampled empty image".format(np_img.shape)
            logger.debug(msg)

        # (2) crop or pad image to match input_shape (if specified)
        self._shape_pre_crop = tuple(np_img.shape)
        if self._input_shape is not None:
            # determine desired image shape
            size_target = self._input_shape

            if slice_dim is not None:
                size_target = list(np_img.shape)
                for i, _ in enumerate(size_target):
                    if i < slice_dim:
                        size_target[i] = self._input_shape[i]
                    elif i > slice_dim:
                        size_target[i] = self._input_shape[i - 1]

            # crop or pad image to desired image shape
            np_img = utilities.image.crop_or_pad_image(
                img_original=np_img, size_target=size_target
            )

            msg = "(shape: {:}) cropped empty image".format(np_img.shape)
            logger.debug(msg)

        # expand dimension if nchannels is not None
        if self._nchannels is not None:
            np_img_shape = list(np_img.shape) + [self._nchannels]
            np_img = np.zeros(shape=tuple(np_img_shape), dtype=dtype)

            msg = "(shape: {:}) expanded empty image".format(np_img.shape)
            logger.debug(msg)

        # store empty image in cache
        self._cache = np_img

        # prepare slicer
        self._slicer = [slice(None)] * len(np_img.shape)

        # determine shape of each slice
        self._shape_slice = list(np_img.shape)
        if slice_dim is not None:
            self._shape_slice[slice_dim] = 1
        self._shape_slice = tuple(self._shape_slice)

        # determine nslice
        if slice_dim is not None:
            self._nslices = np_img.shape[slice_dim]
        else:
            self._nslices = 1

        # reset counters
        self._counter_slice = 0

        return

    def _flush_cache(self):
        logger = self._logger

        # obtain path to reference nii
        filepath_refnii = self._filepath_refniis[self._counter_refnii]

        # create output directory if not exists
        if not os.path.exists(self._dir_output):
            os.makedirs(self._dir_output)

        # write nii
        write_kwargs = {}
        write_kwargs['dir_output'] = self._dir_output
        write_kwargs['filepath_refnii'] = filepath_refnii
        write_kwargs['shape_pre_crop'] = self._shape_pre_crop
        write_kwargs['slice_dim'] = self._slice_dim
        write_kwargs['nchannels'] = self._nchannels
        write_kwargs['resample_order'] = self._resample_order
        write_kwargs['dtype'] = self._dtype

        if self._async_write:
            write_kwargs['np_img'] = self._cache.copy()
            self._writequeue.put(write_kwargs)

            msg = "added {:} to to asynchronous writing queue"
            msg = msg.format(os.path.basename(filepath_refnii))
            logger.info(msg)
        else:
            write_kwargs['np_img'] = self._cache
            NiftiWriter._write_nii_static(logger=logger, **write_kwargs)

        # invalidate counter_slice
        self._counter_slice = -1

        # increment counter_refnii
        self._counter_refnii += 1

        # empty cache and gc
        self._cache = None
        gc.collect()

        return

    @staticmethod
    def _async_write_worker_static(q, threadid):
        logger_name = "NiftiWriter_thd{:}".format(threadid)
        logger = utilities.log.create_or_get_logger(logger_name, level="DEBUG")

        while True:
            write_kwargs = q.get()
            NiftiWriter._write_nii_static(logger=logger, **write_kwargs)

            q.task_done()
        return

    @staticmethod
    def _write_nii_static(
        np_img,
        dir_output,
        filepath_refnii,
        shape_pre_crop,
        slice_dim,
        nchannels,
        resample_order,
        dtype,
        logger,
    ):
        # starttime = time.time()
        msg = "processing and writing to {:}"
        msg = msg.format(os.path.basename(filepath_refnii))
        logger.info(msg)

        # load reference nifti
        nib_refnii = nib.load(filepath_refnii)

        # extract nifti info
        nii_dim = nib_refnii.header.structarr['dim'][0]

        nii_shape = nib_refnii.header.structarr['dim'][1:1 + nii_dim]
        nii_affine = nib_refnii.affine

        # modify nifti info if nchannels is not None
        if nchannels is not None:
            nii_output_shape = tuple(list(nii_shape) + [nchannels])
        else:
            nii_output_shape = tuple(nii_shape)

        # reverse pre-processings to get correct shape
        # (1) undo crop or pad
        if shape_pre_crop is not None:
            msg = "(shape: {:}) uncropping image".format(np_img.shape)
            logger.debug(msg)

            if nchannels is not None:
                _size_original = list(shape_pre_crop) + [nchannels]
            else:
                _size_original = shape_pre_crop

            np_img = utilities.image.inverse_crop_or_pad_image(
                img_target=np_img,
                size_original=tuple(_size_original),
            )

            msg = "(shape: {:}) uncropped image".format(np_img.shape)
            logger.debug(msg)

        # (2) undo resample
        if np_img.shape != nii_output_shape:
            msg = "(shape: {:}) resizing image".format(np_img.shape)
            logger.debug(msg)

            np_img = utilities.image.resize_image(
                np_img,
                nii_output_shape,
                order=resample_order,
            )

            msg = "(shape: {:}) resized image".format(np_img.shape)
            logger.debug(msg)

        # cast type
        if dtype is None:
            dtype = nib_refnii.get_data_dtype()

        np_img = np_img.astype(dtype)

        # write nifti
        filename_outnii = os.path.basename(filepath_refnii)
        filename_outnii = os.path.join(dir_output, filename_outnii)

        nib.Nifti1Image(np_img, nii_affine).to_filename(filename_outnii)

        # endtime = time.time()
        msg = "finished writing to {:}"
        msg = msg.format(os.path.basename(filepath_refnii))
        # msg += "; time taken: {:.2f} seconds".format(endtime - starttime)
        logger.info(msg)

        return
