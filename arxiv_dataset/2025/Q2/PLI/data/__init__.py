from functools import partial

import clip
import numpy as np
import torch
from PIL.Image import Image
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms, Resize

from data.fashionIQ import FashionIQDataset, FashionIQTestDataset, FashionIQTestQueryDataset
from data.cirr import CIRRDataset
from data.pretrain_data import ImageNetDataset, FashionIQCaptionsDataset, MSCOCOCaptionsDataset, Flickr30kCaptionsDataset, ImageNetMscocoDataset
from data.pretrain_data import LlavaDataset, ImageVaDataset
from data.circo import CIRCODataset

from lavis.models import load_model_and_preprocess
# preserve reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(42)


def train_dataset_factory(transforms, config):
    image_transform = transforms['image_transform']
    text_transform = transforms['text_transform']
    dataset_code = config['pretrain_dataset']

    if FashionIQDataset.code() in dataset_code:
        # concat subsets of FashionIQ      
        fashionIQ_datasets = [
            FashionIQDataset(split='train', clothing_type=clothing_type, img_transform=image_transform,
                             text_transform=text_transform)
            for clothing_type in FashionIQDataset.all_subset_codes()
        ]
        dataset = torch.utils.data.ConcatDataset(fashionIQ_datasets)
    elif CIRRDataset.code() in dataset_code:
        dataset = CIRRDataset(split='train', clothing_type=None, img_transform=image_transform,
                              text_transform=text_transform, mode="relative")
    # pretrain datasets
    elif ImageNetDataset.code() in dataset_code:
        dataset = ImageNetDataset(img_transform=image_transform, text_transform=text_transform)
    elif FashionIQCaptionsDataset.code() in dataset_code:
        # concat subsets of FashionIQ
        fashionIQ_caption_datasets = [
            FashionIQCaptionsDataset(split='train', clothing_type=clothing_type, img_transform=image_transform,
                                        text_transform=text_transform)
            for clothing_type in FashionIQCaptionsDataset.all_subset_codes()
        ]
        dataset = torch.utils.data.ConcatDataset(fashionIQ_caption_datasets)
    elif MSCOCOCaptionsDataset.code() in dataset_code:
        dataset = MSCOCOCaptionsDataset(split='train', img_transform=image_transform, text_transform=text_transform)
    elif Flickr30kCaptionsDataset.code() in dataset_code:
        dataset = Flickr30kCaptionsDataset(split='train', img_transform=image_transform, text_transform=text_transform)
    elif ImageNetMscocoDataset.code() in dataset_code:
        dataset = ImageNetMscocoDataset(split='train', img_transform=image_transform, text_transform=text_transform)
    elif LlavaDataset.code() == dataset_code:
        dataset = LlavaDataset(img_transform=image_transform, text_transform=text_transform)
    elif ImageVaDataset.code() == dataset_code:
        dataset = ImageVaDataset(img_transform=image_transform, text_transform=text_transform)
    else:
        raise ValueError("There's no {} dataset".format(dataset_code))

    return dataset


def test_dataset_factory(dataset_code, split='val', image_transform=None, text_transform=None):
    test_datasets = {}
    if FashionIQDataset.code() in dataset_code:
        for clothing_type in FashionIQDataset.all_subset_codes():
            test_datasets['fashionIQ_' + clothing_type] = {
                "samples": FashionIQTestDataset(split=split, clothing_type=clothing_type,
                                                img_transform=image_transform, text_transform=text_transform),
                "query": FashionIQTestQueryDataset(split=split, clothing_type=clothing_type,
                                                   img_transform=image_transform, text_transform=text_transform)
            }
    elif CIRRDataset.code() in dataset_code:
        test_datasets[CIRRDataset.code()] = {
            "samples": CIRRDataset(split=split, clothing_type=None, img_transform=image_transform,
                                   text_transform=text_transform, test_split="samples"),
            "query": CIRRDataset(split=split, clothing_type=None, img_transform=image_transform,
                                 text_transform=text_transform, test_split="query")
        }
    elif CIRCODataset.code() in dataset_code:
        test_datasets[CIRCODataset.code()] = {
            "samples": CIRCODataset(split=split, img_transform=image_transform, mode="classic",
                                      text_transform=text_transform, test_split="samples"),
            "query": CIRCODataset(split=split, img_transform=image_transform, mode="relative",
                                    text_transform=text_transform, test_split="query")
        }
    if len(test_datasets) == 0:
        raise ValueError("There's no {} dataset".format(dataset_code))

    return test_datasets


def train_dataloader_factory(dataset, config, collate_fn=None, drop_last=False):
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 16)
    shuffle = config.get('shuffle', True)
    # drop_last = batch_size == 32

    return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                      collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g, drop_last=drop_last)


def test_dataloader_factory(datasets, config, collate_fn=None):
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 16)
    shuffle = False

    return {
        'query': DataLoader(datasets['query'], batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                            collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g),
        'samples': DataLoader(datasets['samples'], batch_size, shuffle=shuffle, num_workers=num_workers,
                              pin_memory=True,
                              collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g)
    }


def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def transform_factory(configs: dict):
    image_transforms, text_transforms, text_processors = None, None, None
    if "blip" in configs["image_encoders"]:
        blip_model, image_processors, text_processors = load_model_and_preprocess(configs["blip_model_name"],
                                                                                  configs["blip_model_type"])

        image_transforms = {
            'train': image_processors['eval'],
            'val': image_processors['eval']
        }
        if configs["image_transforms"] == "target":
            from data.utils import targetpad_transform
            preprocess = targetpad_transform(target_ratio=1.25, dim=224)
            image_transforms = {
                'train': preprocess,
                'val': preprocess
            }
    if "blip" in configs["text_encoders"]:
        def tokenize(text):
            text_input = blip_model.tokenizer(text, return_tensors="pt", padding='max_length', truncation=True,
                                              max_length=blip_model.max_txt_len)
            return text_input['input_ids']

        text_transforms = {
            'train': transforms.Compose([text_processors['eval'], tokenize]),
            'val': transforms.Compose([text_processors['eval'], tokenize])
        }
    if "clip" in configs["image_encoders"]:
        clip_model, preprocess = clip.load(configs["clip_image_model"], device="cpu")
        
        if configs["image_transforms"] == "target":
            from data.utils import targetpad_transform
            preprocess = targetpad_transform(target_ratio=1.25, dim=clip_model.visual.input_resolution)

        image_transforms = {
            'train': preprocess,
            'val': preprocess
        }
    if "clip" in configs["text_encoders"]:
        # from functools import partial
        tokenizer = partial(clip.tokenize, truncate=True)
        text_transforms = {
            'train': transforms.Compose([tokenizer]),
            'val': transforms.Compose([tokenizer])
        }
    return image_transforms, text_transforms
def create_test_dataloaders(configs):
    """
    To uncouple the data module,
    we simply return the raw images and text that our model needs, without any preprocessing,
    and model pipeline will do the preprocessing.
    """
    # default image transform following CLIP
    img_transforms = transforms.Compose([Resize(configs["img_size"], interpolation=BICUBIC),
                                         transforms.ToTensor()])
    if configs["clip_image_model"] == "None":
        img_transforms = None # BLIP-2 pipeline will do the image transform
    test_datasets = test_dataset_factory(configs['dataset'], image_transform=img_transforms)
    # test_dataloaders = {key: test_dataloader_factory(datasets=value, config=configs, collate_fn=collate_fn) for
    #                     key, value in test_datasets.items()}
    test_dataloaders = test_datasets

    return test_dataloaders


def create_train_eval_dataloaders(configs):
    image_transforms, text_transforms = transform_factory(configs)

    train_dataset = train_dataset_factory(
        transforms={'image_transform': image_transforms['train'], 'text_transform': text_transforms['train']},
        config=configs)
    # train on multiple GPUs, drop_last=True
    train_dataloader = train_dataloader_factory(dataset=train_dataset, config=configs, collate_fn=collate_fn, drop_last=True)

    eval_dataset = test_dataset_factory(configs['dataset'], split='val', image_transform=image_transforms['val'],
                                        text_transform=text_transforms['val'])
    eval_dataloaders = {key: test_dataloader_factory(datasets=value, config=configs, collate_fn=collate_fn) for
                        key, value in eval_dataset.items()}
    return train_dataloader, eval_dataloaders