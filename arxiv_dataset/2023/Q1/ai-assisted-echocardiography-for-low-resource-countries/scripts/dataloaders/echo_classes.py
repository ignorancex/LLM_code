import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
import os

from source.dataloaders.EchocardiographicVideoDataset import EchoClassesDataset
from source.models.ViewClassifiers import SimpleVideoClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Specify config.yml')
    args = parser.parse_args()

    with open(args.config, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)

    # Define some static transforms, i.e. transforms that apply to the entire dataset.
    # These transforms are not augmentation.
    pretransform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=config['pretransform_im_size']),
        transforms.ToTensor(),  # this normalizes in
    ])

    # define some transforms for data augmentation: they have all random parameters that
    # will change at each epoch.
    if config['use_augmentation']:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5), # in degrees
            transforms.RandomEqualize(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),  # this normalizes in
        ])
    else:
        transform=None

    #device = torch.device("cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = EchoClassesDataset(main_data_path=config['main_data_path'],
                                 participant_videos_list=config['participant_videos_list'],
                                 participant_path_json_list=config['participant_path_json_list'],
                                 crop_bounds_for_us_image=config['crop_bounds_for_us_image'],
                                 number_of_frames_per_segment_in_a_clip=config['number_of_frames_per_segment_in_a_clip'],
                                 sliding_window_length_in_percentage_of_frames_per_segment=config['sliding_window_length_in_percentage_of_frames_per_segment'],
                                 device=device,
                                 max_background_duration_in_secs=config['max_background_duration_in_secs'],
                                 pretransform=pretransform,
                                 transform=transform,
                                 use_tmp_storage=True,
                                 )


    ## USAGE

    number_of_clips = len(dataset)
    print(f'Plotting {number_of_clips} clips  and frames: ')
    print(config['number_of_frames_per_segment_in_a_clip'])
    labelnames = ('B', '4') #('BKGR', '4CV')

    print(f'Number of clips: {len(dataset)}')
    print(f'Load two clips: ')
    clip_index_a = 0  # this must be within the dataset length
    clip_index_b = 15  # this must be within the dataset length
    data_a = dataset[clip_index_a]
    data_b = dataset[clip_index_b]

    print('Display the two clips:')
    labelnames = ('BKGR', '4CV')
    plt.figure()
    for f in range(data_a[0].shape[1]):
        plt.subplot(2, data_a[0].shape[1], f+1)
        plt.imshow(data_a[0][0, f, ...].cpu().data.numpy(), cmap='gray')
        plt.axis('off')
        plt.title('{} {}'.format(labelnames[data_a[1]], f))
    for f in range(data_b[0].shape[1]):
        plt.subplot(2, data_b[0].shape[1], f+data_b[0].shape[1]+1)
        plt.imshow(data_b[0][0, f, ...].cpu().data.numpy(), cmap='gray')
        plt.axis('off')
        plt.title('{} {}'.format(labelnames[data_b[1]], f))
    plt.show()

    # Dataloader that will serve the batches over the epochs
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=True)

    # TODO in #34
    # Do a loop as if we were training a model
    data_size = tuple(data_a[0].shape)
    print(type(data_size))
    net = SimpleVideoClassifier(data_size)
    net.to(device)
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-03) # use default settings
    loss_function = nn.CrossEntropyLoss()
    #loss_function = nn.BCEWithLogitsLoss()


    plt.figure()
    subplot_index = 0
    for clip_index_i in range(len(dataset)):
        print(f'   Clip number: {clip_index_i}')
        data_idx = dataset[clip_index_i]
        print(f'   Random index in the segment clip: {data_idx[2]} of n_available_frames {data_idx[3]}')

        for frame_i in range(data_idx[0].shape[1]):
            plt.subplot(number_of_clips, data_idx[0].shape[1], subplot_index+1)
            plt.imshow(data_idx[0][0, frame_i, ...].cpu().data.numpy(), cmap='gray')
            # plt.ylabel('{}'.format( clip_index_i  ) )
            plt.axis('off')
            plt.title('{}:f{}'.format(labelnames[data_idx[1]], frame_i))

            subplot_index +=1

    plt.show()


    # TODO in #34
    training = False
    if training == True:
        # Dataloader that will serve the batches over the epochs
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=True)

        # Do a loop as if we were training a model
        data_size = tuple(data_idx[0].shape)
        print(type(data_size))
        net = SimpleVideoClassifier(data_size)
        net.to(device)
        print(net)

        optimizer = torch.optim.Adam(net.parameters()) # use default settings
        loss_function = nn.CrossEntropyLoss()

        losses = []
        for epoch in range(config['max_epochs']):
            running_loss = 0
            for step, data in enumerate(dataloader):
                clip = data[0]
                label = data[1].to(device)

                out = net(clip)

                loss = loss_function(out, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.detach().item()

            running_loss /= len(dataloader)
            losses.append(running_loss)

            print('{:03.0f} {:.5f}'.format(epoch, running_loss))

        plt.figure()
        plt.plot(losses,'-')
        plt.xlabel('Epochs')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Training')
        plt.show()

    # save the model to disk
    os.makedirs('{}/{}/models'.format(config['output_path'], net.get_name()), exist_ok=True)
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
    }, '{}/{}/models/model.pth'.format(config['output_path'], net.get_name()))

    plt.figure()
    plt.plot(losses,'-')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Training')
    plt.show()

