from typing import List, NoReturn, Tuple, Union, Optional

import numpy as np

from dataset.augmentors import BaseAugmentor
from dataset.commons import SubsetType, GroundTruthType
from dataset.labeltransform import BaseLabelTransform
from dataset.normalizers import BaseNormalizer
from dataset.template.basegroundtruth import BaseGroundTruth
from dataset.template.basesubset import BaseSubset, BaseSubsetBuilder
from dataset.wu2.groundtruth import AudioGroundTruth
from dataset.wu2.loaders import AudioWindowLoader, AudioWindowLoaderConf
from dataset.wu2.wu2metadata import WU2Metadata
from utilities.bufferutils import get_window_timestamp
from utilities.listutils import select_by_index
from utilities.numpyutils import ensure_numpy_2d_vector
from utilities.typingutils import is_typed_list


class AudioWindowSubset(BaseSubset):
    """
    A ``BaseSubset`` implementation that yields audio windows from binary files.

    Binary files are created with MATLAB, see ``src/main/matlab/dataset/wu2/generate_audio_files.m``.
    """

    def __init__(self, subset_type: SubsetType, batch_size: int,
                 session_idxs: List[int], ground_truth: BaseGroundTruth, conf: AudioWindowLoaderConf,
                 augmentor: Optional[BaseAugmentor] = None, label_transform: Optional[BaseLabelTransform] = None,
                 normalizer: Optional[BaseNormalizer] = None):
        """
        Create a loader of audio windows (from binary files).

        :param subset_type: The type of subset
        :param batch_size: Batch size
        :param session_idxs: The indices of the sessions that this loader will load windows from
        :param ground_truth: A ground-truth generator
        :param conf: Configuration for each session loader
        :param augmentor: Augmentor that is applied to each batch
        :param label_transform: A label transform that is applied to each batch
        :param normalizer: A normalizer that is applied to each sample of the batch (before augmentation)
        """
        assert isinstance(subset_type, SubsetType)
        assert isinstance(batch_size, int)

        # Setting length=0 temporarily, will compute its actual value and set it within the constructor
        super().__init__(subset_type, 0, batch_size, subset_type is SubsetType.TRAIN)

        assert is_typed_list(session_idxs, int)
        assert isinstance(ground_truth, BaseGroundTruth)
        assert isinstance(conf, AudioWindowLoaderConf)
        assert isinstance(augmentor, BaseAugmentor) or augmentor is None
        assert isinstance(label_transform, BaseLabelTransform) or label_transform is None
        assert isinstance(normalizer, BaseNormalizer) or normalizer is None

        # print(subset_type, 'session_idxs', session_idxs)

        self._session_idxs: List[int] = session_idxs
        self._ground_truth: BaseGroundTruth = ground_truth
        self._loader_conf: AudioWindowLoaderConf = conf
        self._augmentor: Optional[BaseAugmentor] = augmentor
        self._label_transform: Optional[BaseLabelTransform] = label_transform
        self._normalizer: Optional[BaseNormalizer] = normalizer

        self._should_augment: bool = subset_type is SubsetType.TRAIN and augmentor is not None
        self._should_normalize: bool = normalizer is not None

        self._loaders: List[AudioWindowLoader] = [AudioWindowLoader(idx, conf) for idx in session_idxs]
        self._loaders_session_idx = np.empty(max(session_idxs) + 1)
        self._loaders_session_idx.fill(np.nan)
        for i, session_idx in enumerate(session_idxs):
            self._loaders_session_idx[session_idx] = i

        self._window_idxs: np.ndarray = self._create_window_indices()
        self.length: int = self._window_idxs.shape[0]

        if self._shuffle:
            self.shuffle()

        # TODO - open is now called here. Perhaps change this to a better behavior
        self.open()

    def _create_window_indices(self) -> np.ndarray:
        """
        Creates a 2-D numpy array with three columns: [session index, loader index, window index].
        """
        x: List = []
        for i in range(len(self._session_idxs)):
            noof_windows: int = self._loaders[i].length
            x_i: np.ndarray = np.zeros((noof_windows, 3), np.int)

            session_idx: int = self._session_idxs[i]
            loader_idx: int = self._loaders_session_idx[session_idx]
            for j in range(noof_windows):
                x_i[j, 0] = session_idx
                x_i[j, 1] = loader_idx
                x_i[j, 2] = j
            x.append(x_i)

        return np.vstack(x)

    def open(self) -> NoReturn:
        for loader in self._loaders:
            loader.open()

    def close(self) -> NoReturn:
        for loader in self._loaders:
            loader.close()

    def shuffle(self):
        np.random.shuffle(self._window_idxs)

    def get_loader(self, session_idx: int) -> AudioWindowLoader:
        """
        Convenience method, equivalent to ``self._loaders[self._loaders_session_idx[session_idx]]``, but with checks.

        :param session_idx: The index of the session for which to obtain the loader
        :return: The loader
        """
        assert isinstance(session_idx, int)
        assert session_idx in self._session_idxs

        loader_idx: int = self._loaders_session_idx[session_idx]
        assert loader_idx is not None and loader_idx < len(self._loaders)

        return self._loaders[loader_idx]

    def __getitem__(self, batch_index: int) -> Tuple[np.ndarray, np.ndarray]:
        assert isinstance(batch_index, int)

        x: List = []
        y: List = []
        for i in range(self.batch_size):
            idx: int = batch_index * self.batch_size + i
            if idx >= self._window_idxs.shape[0]:
                continue

            session_idx: int = self._window_idxs[idx, 0]
            loader_idx: int = self._window_idxs[idx, 1]
            window_idx: int = self._window_idxs[idx, 2]

            x_i: np.ndarray = self._loaders[loader_idx].load(window_idx)
            if self._should_normalize:
                x_i = self._normalizer.normalize(x_i)
            x.append(x_i)

            t1, t2 = get_window_timestamp(self._loader_conf.wsize_sec, self._loader_conf.wstep_sec, window_idx)
            y.append(self._ground_truth.get_label(session_idx, t1, t2))

        np_x: np.ndarray = np.array(x)
        np_y: np.ndarray = np.array(y)
        np_y = ensure_numpy_2d_vector(np_y)

        if self._should_augment:
            np_x = self._augmentor.augment_batch(np_x)

        if self._label_transform is not None:
            np_y = self._label_transform.transform_batch(np_y)

        assert not np.any(np.isnan(np_x))
        assert not np.any(np.isnan(np_y))

        # print('batch index', batch_index)
        # print('batch quality', np.sum(np.abs(np_x[:self.batch_size, :] - np_x[self.batch_size:, :])))

        return np_x, np_y

    def get_y(self) -> np.ndarray:
        y: np.ndarray = np.zeros([self.length])

        for idx in range(self.length):
            session_idx: int = self._window_idxs[idx, 0]
            window_idx: int = self._window_idxs[idx, 2]
            t1, t2 = get_window_timestamp(self._loader_conf.wsize_sec, self._loader_conf.wstep_sec, window_idx)
            y[idx] = self._ground_truth.get_label(session_idx, t1, t2)

        if self._label_transform is not None:
            y = self._label_transform.transform_batch(y)

        assert not np.any(np.isnan(y))

        return y


class AudioWindowSubsetBuilder(BaseSubsetBuilder):
    def __init__(self, md: WU2Metadata, batch_size: int, gt_type: GroundTruthType, conf: AudioWindowLoaderConf,
                 augmentor: Optional[BaseAugmentor] = None, label_transform: Optional[BaseLabelTransform] = None,
                 normalizer: Optional[BaseNormalizer] = None):
        assert isinstance(md, WU2Metadata)
        assert isinstance(batch_size, int)
        assert isinstance(gt_type, GroundTruthType)
        assert isinstance(conf, AudioWindowLoaderConf)
        assert isinstance(augmentor, BaseAugmentor) or augmentor is None
        assert isinstance(label_transform, BaseLabelTransform) or label_transform is None
        assert isinstance(normalizer, BaseNormalizer) or normalizer is None

        self._md: WU2Metadata = md
        self._batch_size: int = batch_size
        self._ground_truth: BaseGroundTruth = AudioGroundTruth(md, gt_type)
        self._conf: AudioWindowLoaderConf = conf
        self._augmentor: Optional[BaseAugmentor] = augmentor
        self._label_transform: Optional[BaseLabelTransform] = label_transform
        self._normalizer: Optional[BaseNormalizer] = normalizer

    def get_partitions(self) -> List[str]:
        return self._md.partitions

    def _get_subsets(self, train_idxs: List[int], test_idxs: List[int], validation_idxs: List[int] = None) \
            -> Tuple[BaseSubset, BaseSubset, Union[BaseSubset, None]]:

        train_partitions: List[str] = select_by_index(self._md.partitions, train_idxs)
        train_session_idxs: List[int] = [self._md.session_idxs[i] for i, session_md in enumerate(self._md.session_md)
                                         if session_md.user_name in train_partitions]
        train: BaseSubset = AudioWindowSubset(SubsetType.TRAIN, self._batch_size, train_session_idxs,
                                              self._ground_truth, self._conf, self._augmentor, self._label_transform,
                                              self._normalizer)

        test_partitions: List[str] = select_by_index(self._md.partitions, test_idxs)
        test_session_idxs: List[int] = [self._md.session_idxs[i] for i, session_md in enumerate(self._md.session_md)
                                        if session_md.user_name in test_partitions]
        test: BaseSubset = AudioWindowSubset(SubsetType.TEST, self._batch_size, test_session_idxs, self._ground_truth,
                                             self._conf, self._augmentor, self._label_transform,
                                             self._normalizer)

        validation: Optional[BaseSubset]
        if validation_idxs is not None:
            validation_partitions: List[str] = select_by_index(self._md.partitions, validation_idxs)
            validation_session_idxs: List[int] = [self._md.session_idxs[i]
                                                  for i, session_md in enumerate(self._md.session_md)
                                                  if session_md.user_name in validation_partitions]
            validation = AudioWindowSubset(SubsetType.VALIDATION, self._batch_size, validation_session_idxs,
                                           self._ground_truth, self._conf, self._augmentor, self._label_transform,
                                           self._normalizer)
        else:
            validation = None

        return train, test, validation
