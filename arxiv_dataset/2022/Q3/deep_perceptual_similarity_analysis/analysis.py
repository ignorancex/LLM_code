# Library imports
import torch
import torchvision.transforms as transforms
from PIL import Image
import csv

# File imports
from loss_networks import FeatureExtractor, \
    ExtractionModifier, extractor_collector
from experiment import DistanceMetric


from workspace_path import home_path
log_dir = home_path / 'logs'
checkpoint_dir = home_path / 'checkpoints'
image_dir = home_path / 'images'


def run_analysis(loss_network_path, metric_fun, spatial=False, simple=False):
    '''
    Checks the the given feature extraction network as a similarity metric
    using the given comparison method. Dire
    Args:
        loss_network_path (str): Path to the loss network to use as a metric
        metric_fun (str): Which similarity fun to use "spatial", "sort", "mean"
        spatial (bool): Whether to add spatial with metric_fun
        simple (bool): Whether to skip using channel-wise normalization
    '''

    # Create the metric to use
    loss_network = torch.load(loss_network_path)
    architecture = loss_network.architecture
    if metric_fun != 'spatial':
        loss_network = ExtractionModifier(
            extractor=loss_network,
            modification=metric_fun,
            return_policy= 'concat' if spatial else 'new',
            shape_policy='keep'
        )
    metric = DistanceMetric(loss_network, channel_norm=not simple)

    # Transformation to get images suitable for Torchvision
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])

    # Name this experiment
    name = architecture + '_' + metric_fun
    if spatial:
        name = name + '+spatial'
    if simple:
        name = name + '_simple'

    # Collect the reference images to compare against
    refs = []
    for ref in (image_dir/'ref').iterdir():
        image = Image.open(ref).convert(mode='RGB')
        refs.append(image_transform(image).unsqueeze(dim=0))
    assert len(refs)==7, 'Wrong number of reference images found.'

    # Iterate over all image pairs and check the metric on them and refs
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    with open(log_dir / 'results.csv', 'a', newline='') as save_file:
        writer = csv.writer(save_file, delimiter=' ')
        for dir in image_dir.iterdir():
            if not dir.is_dir() or dir.stem == 'ref':
                continue
            dir_results = []
            dir_dists = []
            for image in dir.iterdir():
                if '_' in image.stem:
                    continue
                org = image_transform(
                    Image.open(image).convert(mode='RGB')
                ).unsqueeze(dim=0)
                for compare_image in dir.iterdir():
                    if not (
                        image.stem in compare_image.stem and '_' in compare_image.stem
                    ):
                        continue
                    aug = image_transform(
                        Image.open(compare_image).convert(mode='RGB')
                    ).unsqueeze(dim=0)
                    aug_dist = metric(org, aug).item()
                    ref_dists = [metric(org, ref).item() for ref in refs]
                    result = [
                        architecture,
                        ('spatial+' if spatial else '') + metric_fun,
                        'no_channel_norm' if simple else 'yes_channel_norm',
                        '/'.join(image.parts[-2:]),
                        '/'.join(compare_image.parts[-2:]),
                        aug_dist 
                    ] + ref_dists
                    writer.writerow(result)
                    dir_results.append(
                        sum([aug_dist>ref_dist for ref_dist in ref_dists])==0
                    )
                    dir_dists.append(aug_dist)
            if dir.stem == 'color_stain':
                print(
                    dir.stem,
                    name,
                    sum([
                        dir_dists[i]>dir_dists[i+1] for i in range(0,len(dir_dists),2)
                    ]),
                    int(len(dir_dists)/2)
                )
            else:
                print(dir.stem, name, sum(dir_results), len(dir_results))


def run_analysis_aggregation():
    '''
    Runs the similarity metric analysis for three networks, five methods
    for calculating deep perceptual similarity (spatial, mean, sort,
    mean+spatial, sort+spatial), and with and without channel-wise normalization. 
    '''

    # Get paths to feature extraction networks
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
    paths = [alexnet_path, squeezenet_path, vgg16_path]
    
    #Run analysis for each possible combination
    metric_funs = ['spatial', 'mean', 'sort']
    for simple in [True,False]:
        for metric_fun in metric_funs:
            for path in paths:
                run_analysis(path, metric_fun, simple=simple)
                if metric_fun != 'spatial':
                    run_analysis(path, metric_fun, spatial=True, simple=simple)


# When this file is executed independently, execute the main function
if __name__ == "__main__":
    run_analysis_aggregation()