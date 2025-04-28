import argparse

import torch
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import yaml
import matplotlib.pyplot as plt

from source.dataloaders.EchocardiographicVideoDataset import EchoClassesDataset
from source.helpers.various import concatenating_YAML_via_tags, plot_dataset_classes
from source.models.nets_misc import SimpleVideoClassifier, train_loop, test_loop

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Specify config.yml')
    args = parser.parse_args()

    yaml.add_constructor('!join', concatenating_YAML_via_tags)  ## register the tag handler

    with open(args.config, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    # Define some static transforms, i.e. transforms that apply to the entire dataset.
    # These transforms are not augmentation.
    if config['use_pretransform_im_size']:
        pretransform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=config['pretransform_im_size']),
            transforms.ToTensor(),
        ])
    else:
        pretransform = None

    # define some transforms for data augmentation: they have all random parameters that
    # will change at each epoch.
    if config['use_augmentation']:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),  # in degrees
            transforms.RandomEqualize(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),  # this normalizes in
        ])
    else:
        transform = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EchoClassesDataset(main_data_path=config['main_data_path'],
                                 participant_videos_list=config['participant_videos_list'],
                                 participant_path_json_list=config['participant_path_json_list'],
                                 crop_bounds_for_us_image=config['crop_bounds_for_us_image'],
                                 number_of_frames_per_segment_in_a_clip=config[
                                     'number_of_frames_per_segment_in_a_clip'],
                                 sliding_window_length_in_percentage_of_frames_per_segment=config[
                                     'sliding_window_length_in_percentage_of_frames_per_segment'],
                                 device=device,
                                 max_background_duration_in_secs=config['max_background_duration_in_secs'],
                                 pretransform=pretransform,
                                 transform=transform,
                                 use_tmp_storage=True,
                                 )


    print(f'Number of clips: {len(dataset)} ')

    # Plotting all clips of the Echo classes
    plot_dataset_classes(dataset, config)


    print(f'Loaded EchoClassesDataset with {len(dataset)} clips ')
    data_clip00 = dataset[0]
    train_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=True)
    print(type(train_dataloader))

    ## Setting up model
    data_size = tuple(data_clip00[0].shape)
    print(type(data_size))
    print(data_size)
    model_net = SimpleVideoClassifier(data_size)
    model_net.to(device)
    print(model_net)

    optimizer = torch.optim.Adam(model_net.parameters())  # use default settings
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(config['max_epochs']):
        print(f'Epoch {epoch + 1}\n-------------------------------')
        train_loop(train_dataloader, model_net, loss_function, optimizer, device)
        test_loop(test_dataloader, model_net, loss_function, device)

    print(f'Done!')
