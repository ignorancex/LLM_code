import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS


class DataSample:
    def __init__(self, results):
        self.cams = []
        for camera_type, cam_item in results["cams"].items():
            self.cams.append(
                dict(
                    camera_type=camera_type,
                    sample_idx=results["sample_idx"],
                    sample_token=results["token"],
                    sample_data_token=cam_item["token"],
                    image_path=cam_item["image_path"],
                    bev_map_path=cam_item["bev_map_path"],
                    bev_shape=cam_item["bev_shape"],
                    resolution=cam_item["resolution"],
                    extents=cam_item["extents"],
                    cam2img=cam_item["cam2img"],
                    image_width=cam_item["image_width"],
                    names=cam_item["names"],
                    colormap=cam_item["colormap"],
                )
            )


@TRANSFORMS.register_module()
class LoadData(BaseTransform):
    def __init__(
        self,
        camera_types=[
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ],
        load_image=True,
        load_bev_map=True,
        test_mode=False,
    ):
        self.camera_types = camera_types
        self.load_image = load_image
        self.load_bev_map = load_bev_map
        self.test_mode = test_mode

    def transform(self, results: dict):
        results["cams"] = {cam: results["cams"][cam] for cam in self.camera_types}

        if self.load_image:
            results["image"] = []
        if self.load_bev_map:
            results["bev_map"] = []

        for cam_item in results["cams"].values():
            if self.load_image:
                image = Image.open(cam_item["image_path"])
                image = image.resize((704, 396))
                image = image.crop((0, 140, 704, 396))
                if not self.test_mode:
                    image = transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)(image)
                image = transforms.ToTensor()(image)
                image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
                results["image"].append(image)
            if self.load_bev_map:
                data = np.load(cam_item["bev_map_path"], allow_pickle=True)
                results["bev_map"].append(torch.tensor(data["bev_map"]))
                cam_item["bev_shape"] = tuple(data["bev_shape"].tolist())
                cam_item["resolution"] = data["resolution"].tolist()
                cam_item["extents"] = data["extents"].tolist()
                cam_item["cam2img"] = data["cam2img"].tolist()
                cam_item["image_width"] = data["image_width"].tolist()
                cam_item["names"] = data["names"].tolist()
                cam_item["colormap"] = data["colormap"].tolist()
        if self.load_image:
            results["image"] = torch.stack(results["image"])
        if self.load_bev_map:
            results["bev_map"] = torch.stack(results["bev_map"])

        inputs = dict()
        data_samples = results["sample_idx"]
        if self.load_image:
            inputs["image"] = results["image"]
        if self.load_bev_map:
            inputs["bev_map"] = results["bev_map"]
            data_samples = DataSample(results)
        return dict(data_samples=data_samples, inputs=inputs)
