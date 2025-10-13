import logging
import typing as tp

from torch.utils.data import Dataset

from datasets import load_dataset

logger = logging.getLogger(__name__)


PARTITION_SIZE = 1000


class PreferenceDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        partition_list: list[str],
        cache_dir: str,
        text_column: str = "caption",
        image_column: str = "image",
    ):
        self.raw_dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        self.partition_list = partition_list
        self.text_column = text_column
        self.image_column = image_column
        self.column_names = [text_column, image_column]

    def __getitem__(self, ind: int) -> dict[str, tp.Any]:
        partition_index = ind // PARTITION_SIZE
        item_index = ind % PARTITION_SIZE
        partition_name = self.partition_list[partition_index]

        row = self.raw_dataset[partition_name][item_index]

        image = (
            row["image1"]
            if row["votes_image1"] > row["votes_image2"]
            else row["image2"]
        )
        caption = row["prompt"]
        return {self.image_column: image, self.text_column: caption}

    def __len__(self):
        return PARTITION_SIZE * len(self.partition_list)
