# Library imports
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import argparse

# File imports
from loss_networks import FeatureExtractor, extractor_collector
from workspace_path import home_path

# Logging, checkpointing, and data directories
log_dir = home_path / 'logs'
checkpoint_dir = home_path / 'checkpoints'
image_dir = home_path / 'images'


def show_feature_map(loss_network_path, image_ids):
    '''
    Takes a path to a network and a list of strings identifying an image.
    Finds the first image with those strings in its path and displays the
    feauture maps of the network for that image.
    Send terminal inputs to generate next feature map.
    Args:
        loss_network_path (str): Path to a network with feature extraction
        image_ids ([str]): List of strings used to identify an image
    '''

    # Transforms to get image into a preferable format for Torchvision
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])

    # Load loss network
    loss_network = torch.load(loss_network_path)

    # Find the image and augs from ids
    images = []
    for dir in image_dir.iterdir():
        if not dir.is_dir() or dir.stem == 'ref':
            continue
        for image in dir.iterdir():
            if all([i in str(image) for i in image_ids]):
                images.append(image)
        if len(images) > 0:
            break
    
    # Load images
    images = torch.cat([
        image_transform(
            Image.open(i).convert(mode='RGB')
        ).unsqueeze(dim=0) for i in images
    ], dim=0)

    # Get feature maps
    maps = loss_network(images)
    cn_maps = [F.normalize(y) for y in maps]
    for i, (y, cn_y) in enumerate(zip(maps, cn_maps)):
        sz = y.size()
        m = y.permute(1,2,0,3).reshape(sz[1],sz[2],sz[0]*sz[3])
        cn_m = cn_y.permute(1,2,0,3).reshape(sz[1],sz[2],sz[0]*sz[3])
        maps[i] = torch.cat([m, cn_m], dim=1)

    # Show feature maps
    to_pil = torchvision.transforms.ToPILImage()
    for ys in maps:
        for y in ys:
            to_pil(y.detach()).resize((768,512),resample=Image.BOX).show()
            input()
    

def run_feature_mapping():
    '''
    Visualizes the feature maps of the image identified by input strings
    for the selected network.
    Send inputs through the terminal to generate the next feature map.
    '''
    
    # Create parser and parse input
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image',
        type=str,
        default=['3ring','color'],
        nargs='+',
        help='Selects the first image '
    )
    parser.add_argument(
        '--network',
        type=str,
        default='alexnet',
        choices=['alexnet','squeezenet','vgg16'],
        help='Network for which to generate feature maps'
    )
    args = parser.parse_args()

    # Find or create the feature extraction networks
    alexnet_path = extractor_collector(
        FeatureExtractor,
        architecture = 'alexnet',
        layers = [1,4,7,9,11],
        pretrained=True,
        frozen = True,
        flatten_layer = False,
        normalize_in = False,
    )
    squeezenet_path = extractor_collector(
        FeatureExtractor,
        architecture = 'squeezenet1_1',
        layers = [1,4,7,9,10,11,12],
        pretrained=True,
        frozen = True,
        flatten_layer = False,
        normalize_in = False,
    )
    vgg16_path = extractor_collector(
        FeatureExtractor,
        architecture = 'vgg16',
        layers = [3,8,15,22,29],
        pretrained=True,
        frozen = True,
        flatten_layer = False,
        normalize_in = False,
    )
    path = vgg16_path
    if args.network == 'alexnet':
        path = alexnet_path
    elif args.network == 'squeezenet':
        path = squeezenet_path

    # Visualize the feature maps
    show_feature_map(path, args.image)


# When this file is executed independently, execute the main function
if __name__ == "__main__":
    run_feature_mapping()