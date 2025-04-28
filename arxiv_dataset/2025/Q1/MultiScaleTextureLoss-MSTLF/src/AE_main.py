"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import matplotlib.pyplot as plt

from options.train_options import TrainOptions
from data import create_dataset
from models.autoencoder_perceptual import *
import torch

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options (train dataset)
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = Autoencoder().apply(weights_init_normal)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Number of epochs
    num_epochs = 100
    loss = []

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0

        for idx, input_batch in enumerate(dataset):

            # Forward pass
            encoded_input, decoded_output = model(input_batch['img'])
            print(input_batch['img'].shape)
            print(decoded_output.shape)

            # Compute the loss
            loss = criterion(decoded_output, input_batch['img'])

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx == 1 or idx == 300 or idx == 800 or idx == 1200 or idx == 1450 or idx == 1800 or idx == 2000 or idx == 2200 or idx == 2500 or idx == 2700 or idx == 3200 or idx == 3250:
                plt.imshow(input_batch['img'][4, 0, :, :].detach(), cmap="gray")
                plt.savefig(f'./autoencoder/input_{idx}.png')
                plt.imshow(decoded_output[4, 0, :, :].detach(), cmap="gray")
                plt.savefig(f'./autoencoder/decoded_output_{idx}.png')

            total_loss += loss.item()

        # Print average loss for the epoch
        average_loss = total_loss / len(dataset)

        loss.append(average_loss)
        plt.plot(loss)
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('./autoencoder/training_loss_plot.png')

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

    torch.save(model.state_dict(), f"/autoencoder/autoencoder.pth")
    print('Training finished!')

