import os
from PIL import Image
import webdataset as wds
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset


class CCSBUDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "image": sample[0],
            "answer": self.text_processor(sample[1]["caption"]),
        }


class CCSBUAlignDataset(CaptionDataset):

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = '{}.jpg'.format(ann["image_id"])
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = ann["caption"]

        return {
            "image": image,
            "answer": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }


class IMADDataset(CaptionDataset):

    def __getitem__(self, index):

        ann = self.annotation[index]

        img_file = '{}.jpg'.format(ann["image_id"])
        
        # Image 1 - ref
        image_path = os.path.join(self.vis_root, img_file)
        image1 = Image.open(image_path).convert("RGB")
        image1 = self.vis_processor(image1)
        
        # Image 2 - changed
        path2 = self.vis_root.replace("input_images", "output_images")
        image_path_2 = os.path.join(path2, img_file)
        image2 = Image.open(image_path_2).convert("RGB")
        image2 = self.vis_processor(image2)

        prediction = ann["prediction"]
        command = ann["command"]

        return {
            "image1": image1,
            "prediction": prediction,
            "image_id": self.img_ids[ann["image_id"]],
            "image2": image2,
            "command": command
        }