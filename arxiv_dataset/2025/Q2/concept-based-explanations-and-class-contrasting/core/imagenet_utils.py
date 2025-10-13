
import torch
import numpy as np
import os
import csv
import torch
import torchvision
import copy
from torchvision.datasets.folder import default_loader

from tqdm import tqdm

from pathlib import Path
import math


import torchvision
from torchvision.models import resnet50, resnet34, vit_b_32, vit_b_16


from core.config import PATH_TO_IMAGENET_TRAIN, PATH_TO_IMAGENET_VAL, PATH_VAL_SOLUTIONS, OUTPUT_PATH

DATA_RESIZE_SIZE = 224 #256 #480
DATA_CROP_SIZE = 224 #224 #480





def load_model(model_str):
    if model_str == "resnet50":
        return resnet50(pretrained=True).to("cuda").eval()
    elif model_str == "resnet34":
        return resnet34(pretrained=True).to("cuda").eval()
    elif model_str == "resnet50_robust":
        classifier_model = resnet50(pretrained=False)

        checkpoint = torch.load(...) #TODO: set path here

        state_dict_path = 'model'
        if not ('model' in checkpoint):
            state_dict_path = 'state_dict'

        new_state_dict = {}
        for key in checkpoint['model'].keys():
            if not "attacker" in key:
                if 'model' in key:
                    new_key = key[13:]
                    #print(new_key)
                    new_state_dict[new_key] = checkpoint['model'][key]

        classifier_model.load_state_dict(new_state_dict)

        return classifier_model.eval().to("cuda")
    elif ((model_str == "vit_b_16_dino") or (model_str == "vit_b_16_dino_ignore_last_token")):
        #classifier_model = vit_b_16(pretrained=False)
        classifier_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')

        #state_dict = torch.load("models/dino_vitbase16_pretrain.pth")

        #print(state_dict.keys())
        #raise RuntimeError

        #classifier_model.load_state_dict(state_dict)
        #raise RuntimeError
        return classifier_model.eval().to("cuda")
    elif model_str == "vit_b_8_dino":
        classifier_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
        return classifier_model.eval().to("cuda")
    elif model_str == "vit_b_32":
        return vit_b_32(pretrained=True).eval()#.to("cuda").eval()
    elif model_str == "vit_b_16":
        return vit_b_16(pretrained=True).eval()#.to("cuda").eval()




class ImageNetValDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_val, transform=None, transform_original=None, return_original_image=False):
        super().__init__()
        self.dataset_path = dataset_path_val
        self.transform = transform
        self.transform_original = transform_original
        self.return_original_image = return_original_image

        dataset_path_train = PATH_TO_IMAGENET_TRAIN#"/localdata/xai_derma/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train/"

        wnid_to_class = {}
        i = 0
        for folder_name in sorted(os.listdir(dataset_path_train)):
            wnid_to_class[folder_name] = i
            i += 1
        val_solutions_path = PATH_VAL_SOLUTIONS#"/localdata/xai_derma/imagenet-object-localization-challenge/LOC_val_solution.csv"

        self.image_id_to_class = {}
        with open(val_solutions_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=" ")
            for i, line in enumerate(reader):
                if i == 0:
                    continue
                image_id, wnid = line[0].split(",")
                self.image_id_to_class[image_id] = wnid_to_class[wnid]

        self.files = os.listdir(self.dataset_path)


    def __len__(self):
        #print("len dataset: {}".format(len(os.listdir(self.dataset_path))))
        return len(self.files)


    def __getitem__(self, index):
        filepath = self.files[index]
        class_idx = self.image_id_to_class[filepath.split(".")[0]]

        sample = default_loader(os.path.join(self.dataset_path, filepath))
        original_sample = copy.deepcopy(sample)
        if self.transform is not None:
            sample = self.transform(sample)
            #return self.transform(sample), class_idx

        if self.return_original_image:
            #print("return original sample")
            original_sample = self.transform_original(original_sample)
            return sample, class_idx, original_sample

        #print("do not return original sample")
        return sample, class_idx


class ImageNetTrainDataset(torchvision.datasets.ImageFolder):
    def __init__(self, transform_original, transform_self, **kwargs):
        super().__init__(**kwargs)
        self.transform_self = transform_self
        self.transform_original = transform_original

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        transformed_img = self.transform_self(img)
        original_img = self.transform_original(img)

        return transformed_img, target, original_img





def get_dataset(return_original_sample=False, use_train_ds=True):
    transform = [
        torchvision.transforms.Resize(DATA_RESIZE_SIZE),
        torchvision.transforms.CenterCrop(DATA_CROP_SIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    transform = torchvision.transforms.Compose(transform)


    transform_original_sample = [
        torchvision.transforms.Resize(DATA_RESIZE_SIZE),
        torchvision.transforms.CenterCrop(DATA_CROP_SIZE),
        torchvision.transforms.ToTensor(),
    ]
    transform_original_sample = torchvision.transforms.Compose(transform_original_sample)


    if return_original_sample:
        if use_train_ds:
            dataset = ImageNetTrainDataset(root=PATH_TO_IMAGENET_TRAIN, transform_self=transform, transform_original=transform_original_sample)
        else:
            dataset = ImageNetValDataset(transform=transform,
                                        dataset_path_val=PATH_TO_IMAGENET_VAL,
                                        return_original_image=True,
                                        transform_original=transform_original_sample)
    else:
        if use_train_ds:
            dataset = torchvision.datasets.ImageFolder(PATH_TO_IMAGENET_TRAIN, transform=transform)
        else:
            dataset = ImageNetValDataset(transform=transform,
                                    dataset_path_val=PATH_TO_IMAGENET_VAL)

    return dataset



class CroppedImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, patch_size=74):
        self.dataset = dataset
        self.patch_size = patch_size
        self.strides = int(0.8*self.patch_size)

        for data in self.dataset:
            break
        #print(len(data))
        self.len_individual_data = len(data)
        img = data[0]
        #print(img.shape)
        patches = torch.nn.functional.unfold(img.unsqueeze(dim=0), kernel_size=self.patch_size, stride=self.strides)
        patches = patches.transpose(1, 2).contiguous().view(-1, 3, self.patch_size, self.patch_size)
        self.len = len(patches) * len(self.dataset)
        self.patch_h = int(math.sqrt(len(patches)))


    def __len__(self):
        return self.len


    def __getitem__(self, index):
        dataset_idx = int(index / (self.patch_h * self.patch_h))

        idx_remain = index % (self.patch_h * self.patch_h)
        if self.len_individual_data == 3:
            transformed_img, target, original_img = self.dataset[dataset_idx]


            transformed_img = torch.nn.functional.unfold(transformed_img.unsqueeze(dim=0), kernel_size=self.patch_size, stride=self.strides)
            transformed_img = transformed_img.transpose(1, 2).contiguous().view(-1, 3, self.patch_size, self.patch_size)

            original_img = torch.nn.functional.unfold(original_img.unsqueeze(dim=0), kernel_size=self.patch_size, stride=self.strides)
            original_img = original_img.transpose(1, 2).contiguous().view(-1, 3, self.patch_size, self.patch_size)


            transformed_img = transformed_img[idx_remain]
            original_img = original_img[idx_remain]

            return transformed_img, target, original_img


        else:

            transformed_img, target = self.dataset[dataset_idx]

            transformed_img = torch.nn.functional.unfold(transformed_img.unsqueeze(dim=0), kernel_size=self.patch_size, stride=self.strides)
            transformed_img = transformed_img.transpose(1, 2).contiguous().view(-1, 3, self.patch_size, self.patch_size)

            transformed_img = transformed_img[idx_remain]

            return transformed_img, target





def get_dataset_excluding_class(model_name, class_idx):
    dataset = get_dataset(return_original_sample=False, use_train_ds=False)

    prediction_classes = np.load(os.path.join(OUTPUT_PATH, "predictions_imagenet/" + model_name + "_val.npy"))

    indices = np.where(prediction_classes != class_idx)[0]
    dataset = torch.utils.data.Subset(dataset, indices)
    #print(np.where(prediction_classes == class_idx)[0])

    return dataset




def get_dataset_for_class(model_name, class_idx, use_train_ds=False):
    dataset = get_dataset(return_original_sample=False, use_train_ds=use_train_ds)

    if use_train_ds:
        prediction_classes = np.load(os.path.join(OUTPUT_PATH, "predictions_imagenet/" + model_name + "_train.npy"))
    else:
        prediction_classes = np.load(os.path.join(OUTPUT_PATH, "predictions_imagenet/" + model_name + "_val.npy"))

    indices = np.where(prediction_classes == class_idx)[0]
    dataset = torch.utils.data.Subset(dataset, indices)
    #print(np.where(prediction_classes == class_idx)[0])

    return dataset


def generate_prediction_list(model_name, dataset, save_data_append="_val"):
    model = load_model(model_name).eval().to("cuda")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)

    predictions = []

    with torch.no_grad():
        for x, target in tqdm(dataloader, ascii=True):
            x = x.to("cuda")
            pred = torch.argmax(model(x), dim=1).cpu()
            predictions.append(pred)

    predictions = torch.cat(predictions, dim=0).numpy()

    out_folder = os.path.join(OUTPUT_PATH, "predictions_imagenet")

    Path(out_folder).mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(out_folder, model_name + save_data_append + ".npy"), predictions)


