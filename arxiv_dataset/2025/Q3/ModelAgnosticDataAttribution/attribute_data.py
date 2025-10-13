import torch
import torch.nn as nn
import numpy as np
import argparse
import torch.nn.functional as F
import torchvision.transforms as T
from data import CelebDataset, get_subset_celeba_attr
from torch.utils.data import DataLoader
from models import ResNet18
from tqdm import tqdm
from utils import get_patch_predictions


# take attribute as command line argument
parser = argparse.ArgumentParser(description='Caclulate attributions')
parser.add_argument('attribute', default='Blond_Hair', type=str,
                    help='The attribute to calculate attributions for')
# argument for using a subset or no
parser.add_argument('--num_classes', default=2, type=int)
parser.add_argument('--n_regions', default=2401, type=int)
parser.add_argument('--model_path', default='models/model.pt', type=str)
parser.add_argument('--save_path', default='attributions.npy', type=str)
parser.add_argument('--partition', type=str, choices=['train', 'val', 'test', 'custom'], default='val',
                    help='The partition to calculate attributions for, when custom, will create a subset based on extra attribute')
parser.add_argument('--extra_attribute', type=str, default='Smiling')
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()


# ensure reproducibility
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)


transforms = T.Compose([
    T.Resize((128, 128), antialias=True),
    T.ConvertImageDtype(torch.float32),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if args.partition == 'custom':
    subset_indices, _ = get_subset_celeba_attr(
        'data/celeba/list_attr_celeba.csv', 'data/celeba/list_eval_partition.csv', target_attribute=args.attribute, hidden_attribute=args.extra_attribute)
    dataset = CelebDataset(target=args.attribute,
                           transforms=transforms, partition=0)
    dataset = torch.utils.data.Subset(dataset, subset_indices)
else:
    mapping = {'train': 0, 'val': 1, 'test': 2}
    dataset = CelebDataset(target=args.attribute,
                        transforms=transforms, partition=mapping[args.partition])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
attribution_model = ResNet18(
    num_classes=args.num_classes, region=True, n_regions=args.n_regions).to(device)
attribution_model.load_state_dict(torch.load(args.model_path))
attribution_model.eval()

to_save = []
for i in tqdm(range(len(dataset)), total=len(dataset), desc="Calculating attributions"):
    image, target = dataset[i]
    image = image.squeeze(0).to(device)

    # get the predictions
    predictions = get_patch_predictions(
        image, 32, attribution_model,  args.num_classes)
    to_save.append(predictions)

torch.save(to_save, args.save_path)
