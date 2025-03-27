from dataset.commons import PartitionMode
from dataset.wu1.commons import LabelMode
from dataset.wu1.loaders import Loader, load_durations
from dataset.wu1.wu1metadata import WU1Metadata


class WU1Experiment:
    def __init__(self,
                 wu1metadata: WU1Metadata,
                 label_mode: LabelMode,
                 partition_mode: PartitionMode,
                 validation_items: int,
                 batch_size: int,
                 loader: Loader = None):
        # Input argument assertions
        assert isinstance(wu1metadata, WU1Metadata)
        assert isinstance(label_mode, LabelMode)
        assert isinstance(partition_mode, PartitionMode)
        assert isinstance(validation_items, int)
        assert isinstance(batch_size, int)
        assert isinstance(loader, Loader) or loader is None
        if loader is None:
            loader = Loader(wu1metadata)

        self.wu1md = wu1metadata
        self.label_mode = label_mode
        self.partition_mode = partition_mode
        self.validation_items = validation_items
        self.batch_size = batch_size
        self.loader = loader

        if partition_mode is PartitionMode.LOSO_SIMPLE:
            self.ids = wu1metadata.user_ids
        elif partition_mode is PartitionMode.LOFTO_SIMPLE:
            self.ids = wu1metadata.food_type_ids
        else:
            raise ValueError("Unsupported partition mode")

        self.validation_percentage = validation_items / len(wu1metadata.get_partition_ids(partition_mode))
        self._durations = None  # Postpone loading durations to avoid unnecessary delays during object instantiation.

    def get_durations(self):
        if self._durations is None:
            self._durations = load_durations(self.wu1md, self.loader, False)
        return self._durations
