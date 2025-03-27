import ast
import csv
import sys
import tqdm
import timm

import hydra
import shutil
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

from PIL import Image
from pathlib import Path
from omegaconf import DictConfig
from torchvision import transforms

from source.dataset import SlideDataset
from source.dataset_utils import collate_patch_filepaths, stain_norm

from models.resnet_lunit import resnet50
from models.vits_lunit import vit_small

@hydra.main(
    version_base="1.2.0", config_path="configs", config_name="default"
)
def main(cfg: DictConfig):

    features_dir = Path(cfg.output_dir_path, cfg.dataset.name, cfg.feature_extractor)
    slide_features_dir = Path(features_dir, "slide")
    patch_features_dir = Path(features_dir, "patch")

    if not cfg.resume:
        if features_dir.exists():
            print(f"{features_dir} already exists! deleting it...")
            shutil.rmtree(features_dir)
            print("done")
            features_dir.mkdir(parents=False)
            slide_features_dir.mkdir()
            if cfg.feature_extraction.save_patch_features:
                patch_features_dir.mkdir()
        else:
            features_dir.mkdir(parents=True, exist_ok=True)
            slide_features_dir.mkdir(parents=True, exist_ok=True)
            if cfg.feature_extraction.save_patch_features:
                patch_features_dir.mkdir(exist_ok=True)
    
    device = torch.device("cuda:" + str(cfg.gpu_id) if torch.cuda.is_available() else "cpu")
    print(f"Use {device} for feature extraction")

    if cfg.model.backbone == "resnet50":
        if "supervised-imagenet1k" in cfg.feature_extractor:
            feature_extractor = timm.create_model('resnet50', pretrained=True)
        elif "ssl-barlow_twins-imagenet1k" in cfg.feature_extractor:
            feature_extractor = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        elif "ssl-swav-imagenet1k" in cfg.feature_extractor:
            feature_extractor = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        elif "ssl-dino-imagenet1k" in cfg.feature_extractor:
            feature_extractor = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        feature_extractor.layer4 = nn.Identity() # remove the last layer so that the feature dimension is 1024 not 2048
        feature_extractor.fc = nn.Identity()
    
    elif cfg.model.backbone == "resnet50-lunit":
        if "ssl-barlow_twins-tcga-tulip" in cfg.feature_extractor:
            feature_extractor = resnet50(pretrained=True, progress=False, key='BT')
        elif "ssl-mocov2-tcga-tulip" in cfg.feature_extractor:
            feature_extractor = resnet50(pretrained=True, progress=False, key='MoCoV2')
        elif "ssl-swav-tcga-tulip" in cfg.feature_extractor:
            feature_extractor = resnet50(pretrained=True, progress=False, key='SwAV')
        feature_extractor.layer4 = nn.Identity()
        
    elif cfg.model.backbone == "vit_small16":
        if "supervised-imagenet1k" in cfg.feature_extractor:
            feature_extractor = timm.create_model('vit_small_patch16_224', pretrained=True)
        elif "supervised-imagenet21k" in cfg.feature_extractor:
            feature_extractor = timm.create_model('vit_small_patch16_224_in21k', pretrained=True)
        elif "ssl-dino-imagenet1k" in cfg.feature_extractor:
            feature_extractor = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        feature_extractor.head = nn.Identity()
    
    elif cfg.model.backbone == "vit_small16-lunit":
        feature_extractor = vit_small(pretrained=True, progress=False, key='DINO_p16', patch_size=16)
    
    elif cfg.model.backbone == "convnext_base":
        if "supervised-imagenet1k" in cfg.feature_extractor:
            feature_extractor = timm.create_model('convnext_base', pretrained=True)
        elif "supervised-imagenet21k" in cfg.feature_extractor:
            feature_extractor = timm.create_model('convnext_base_in22k', pretrained=True)
        feature_extractor.head = nn.Identity()
    
    elif cfg.model.backbone == "swin_base":
        if "supervised-imagenet1k" in cfg.feature_extractor:
            feature_extractor = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        elif "supervised-imagenet21k" in cfg.feature_extractor:
            feature_extractor = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=True)
        feature_extractor.head = nn.Identity()
    else:
        raise ValueError("Backbone is not supported")
    
    feature_extractor = feature_extractor.to(device)

    print("Model is loaded successfully")

    if cfg.feature_extraction.normalization == "imagenet":
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif cfg.feature_extraction.normalization == "lunit":
        mean, std = [0.70322989, 0.53606487, 0.66096631], [0.21716536, 0.26081574, 0.20723464]
    else:
        mean, std = None, None
    
    if mean is not None and std is not None:
        normalize = T.Normalize(mean=mean, std=std)
        test_transform = T.Compose([
            T.ToPILImage(),
            T.Resize(224),
            T.ToTensor(),
            normalize
        ])
    else:
        test_transform = None

    slide_data_path = cfg.dataset.slide_data_path
    slide_data = []
    if cfg.resume:
        with open(slide_data_path, 'r', newline='')as file:
            reader = csv.reader(file)
            for row in reader:
                row_list = ast.literal_eval(row[0])
                subset, class_, slide_stem = row_list
                slide_data.append([subset, class_, slide_stem])
    else:
        for subset in cfg.dataset.subsets:
            for class_ in sorted(cfg.dataset.classes):
                data_dir = f"{cfg.dataset.base_folder_path}/{subset}/{class_}"
                slides = [p for p in Path(data_dir).glob('*') if p.is_dir()]
                assert len(slides), f"There is no raw patches in {data_dir}"
                for slide in slides:
                    slide_data.append([subset, class_, slide.stem])

        with open(slide_data_path, 'w', newline='') as file:
            writer = csv.writer(file)
            for item in slide_data:
                writer.writerow([item])

    print(f"{len(slide_data)} slides found")

    slide_dataset = SlideDataset(slide_data, cfg.dataset.name, cfg.dataset.base_folder_path)
    loader = torch.utils.data.DataLoader(
        slide_dataset, batch_size=1, shuffle=False, collate_fn=collate_patch_filepaths
    )

    feature_extractor.eval()

    with tqdm.tqdm(
        loader,
        desc="Slide Encoding",
        unit="slide",
        ncols=80,
        position=0,
        leave=True,
        file=sys.stderr
    ) as t1:

        with torch.no_grad():

            for i, batch in enumerate(t1):

                idx, region_fps = batch
                subset, class_, slide_id = slide_data[idx][0], slide_data[idx][1], slide_data[idx][2]
                features = []

                with tqdm.tqdm(
                    region_fps,
                    desc=(f"{slide_data[idx]}"),
                    unit=" patch",
                    ncols=80 + len(slide_data[idx]),
                    position=1,
                    leave=False,
                ) as t2:
                    
                    # check if the raw patches exist
                    if len(t2) == 0:
                        with open(cfg.dataset.slide_missing_path, mode="a") as file:
                            file.write(f"{slide_data[idx]}\n")
                        continue

                    for fp in t2:

                        slide_id_x_y = Path(fp).stem
                        
                        img = np.array(Image.open(fp))

                        if cfg.feature_extraction.stain_norm_macenko:
                            img = stain_norm(img)

                        img = transforms.functional.to_tensor(img)  # [3, 256, 256]
                        img = transforms.Resize((cfg.dataset.patch_size, cfg.dataset.patch_size))(img) # make sure the inputs are in the desired shape

                        # apply transformations if needed
                        if test_transform is not None:
                            img = test_transform(img)

                        img = img.unsqueeze(0)  # [1, 3, 256, 256]
                        img = img.to(device, non_blocking=True)

                        if cfg.model.backbone in ["resnet50-lunit", "convnext_base"]:
                            feature = feature_extractor(img)
                            feature = F.avg_pool2d(feature, kernel_size=(cfg.model.kernel_size, cfg.model.kernel_size))
                            feature = feature.view(1, cfg.model.feats_dim)

                        elif cfg.model.backbone in ["resnet50", "vit_small16", "vit_small16-lunit", "swin_base"]:
                            feature = feature_extractor(img)
                            feature = feature.view(1, cfg.model.feats_dim)
                            
                        if cfg.feature_extraction.save_patch_features:
                            patch_saved_path = Path(patch_features_dir, subset, class_)
                            patch_saved_path.mkdir(parents=True, exist_ok=True)
                            torch.save(feature, patch_saved_path / f"{slide_id_x_y}.pt")
                        features.append(feature)

                stacked_features = torch.stack(features, dim=0).squeeze(1)
                slide_saved_path = Path(slide_features_dir, subset, class_)
                slide_saved_path.mkdir(parents=True, exist_ok=True)
                torch.save(stacked_features, slide_saved_path / f"{slide_id}.pt")

if __name__ == "__main__":

    main()