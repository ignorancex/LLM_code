
"""
Main function to pretrain VLMs using
an assembly dataset and vision-text modalities.
"""

import argparse

import torch
torch.multiprocessing.set_start_method('spawn', force=True)

from cxrvlms.pretraining.data.dataloader import get_loader
from cxrvlms.pretraining.data.transforms import augmentations_pretraining
from cxrvlms.modeling.model import VLMModel

from local_data.constants import *


def process(args):

    # Set data for training
    datalaoders = get_loader(dataframes_path=args.dataframes_path, data_root_path=args.data_root_path,
                             datasets=args.datasets, batch_size=args.batch_size, num_workers=args.num_workers,
                             banned_categories=args.banned_categories, caption=args.caption, cache=args.cache,
                             norm=args.norm, size=args.size)

    # Init VLM model
    model = VLMModel(vision_type=args.architecture, out_path=args.out_path + args.exp_id + '/',
                     from_checkpoint=False, vision_pretrained=True, learning_criteria=args.learning_criteria,
                     projection=args.project_features, norm_features=args.norm_features,
                     apply_t_scaling=args.apply_t_scaling, learn_t_scaling=args.learn_t_scaling,
                     lambda_dlilp=args.lambda_dlilp)

    # Training
    model.fit(datalaoders, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, scheduler=args.scheduler,
              warmup_epoch=args.warmup_epoch, store_num=args.store_num, transforms=augmentations_pretraining)


def main():
    parser = argparse.ArgumentParser()

    # Folders, data, etc.
    parser.add_argument('--data_root_path', default=PATH_DATASETS)
    parser.add_argument('--dataframes_path', default=PATH_DATAFRAME_PRETRAIN)
    parser.add_argument('--datasets', default="CheXpert-train-frontal,MIMIC-CXR-2-train-frontal",
                        help='MIMIC-CXR-2-train-frontal,CheXpert-train-frontal,PadChest-train-frontal'
                             'PadChest-train-frontal',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--banned_categories', default=[])
    parser.add_argument('--out_path', default=PATH_RESULTS_PRETRAIN, help='output path')
    parser.add_argument('--exp_id', default='[ID_vlms_n]', help='output path')

    # Prompts setting and augmentation hyperparams
    parser.add_argument('--caption', default="A radiology image of [CLS]")

    # Dataloader setting
    parser.add_argument('--cache', default=False, type=bool, help='memory_cache')

    # Training options
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    parser.add_argument('--scheduler', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--warmup_epoch', default=1, type=int, help='number of warmup epochs')
    parser.add_argument('--size', default=224, type=int, help='image size')
    parser.add_argument('--norm', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--store_num', default=5, type=int)
    parser.add_argument('--project_features', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--norm_features', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--learning_criteria', default="dlilp", help="unicl-clip-unimodal-dlilp")
    parser.add_argument('--apply_t_scaling', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--learn_t_scaling', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--lambda_dlilp', default=0.1, type=float, help='Relative importance in text branch from DLILP')

    # Architecture and pretrained weights options
    parser.add_argument('--architecture', default='resnet_v2', help='resnet_v1 -- resnet_v2 - convnext')

    # Resources
    parser.add_argument('--num_workers', default=0, type=int, help='workers number for DataLoader')

    args, unknown = parser.parse_known_args()
    process(args=args)


if __name__ == "__main__":
    main()