import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
from model import ModifiedVGG19
from augment import get_augmentation
from augment import test_augmentation
import matplotlib.pyplot as plt
save_dir="/home/mmoradi6/DDPM-06.30/DDPM-Pytorch-main/case_study/Classification model/checkpoints/"
image_dir="/home/mmoradi6/DDPM-06.30/DDPM-Pytorch-main/case_study/images/"
evaluation_dir="/home/mmoradi6/DDPM-06.30/DDPM-Pytorch-main/case_study/evaluation/"

# Define the training function
def train(model, dataloader,dataloader_test, criterion, optimizer, num_epochs, phase):
    all_loss=[]
    all_test_loss=[]
    x_axis=[iii for iii in range(1,101)]
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        test_running_loss=0.0
        for inputs, targets in dataloader:
            cl=['0', '100', '33', '66']
            # print("test1",targets)
            # print("test2",[iii.item() for iii in targets])
            targets=torch.tensor([int(cl[ii]) for ii in targets])
            # print("test3",targets)
            inputs, targets = inputs.cuda(), targets.cuda()
            targets=targets.to(dtype=torch.float32)/100
            optimizer.zero_grad()
            outputs = model(inputs)
            # print("outputs",outputs.dtype)
            # print("target",targets.dtype)
            loss = criterion(outputs, targets)
            # print("loss",len(dataloader))
            running_loss=running_loss+loss.detach().item()
            loss.backward()
            optimizer.step()
        #evaliuation
        for test_inputs, test_targets in dataloader_test:
            cl=['100']
            print("test_targets",test_targets)
            test_targets=torch.tensor([int(cl[ii]) for ii in test_targets])
            test_inputs, test_targets = test_inputs.cuda(), test_targets.cuda()
            test_targets=test_targets.to(dtype=torch.float32)/100
            test_outputs = model(test_inputs)
            test_loss = criterion(test_outputs, test_targets)
            test_running_loss=test_running_loss+test_loss.detach().item()
            del test_loss



        print(f'Epoch {epoch+1}/{num_epochs} [{phase}]: Trainin Loss = {running_loss/len(dataloader)}, Test Loss = {test_running_loss/len(dataloader_test)}')
        all_loss.append(running_loss/len(dataloader))
        all_test_loss.append(test_running_loss/len(dataloader_test))
        # Save the model
        torch.save(model.state_dict(), save_dir+phase+str(epoch+1)+".pth")
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.plot(x_axis, all_loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE train loss")
    ax.set_title(phase+":train")
    fig.savefig('/home/mmoradi6/DDPM-06.30/DDPM-Pytorch-main/case_study/Classification model/checkpoints/'+phase+"train.png")   # save the figure to file
    plt.close(fig) 
    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.plot(x_axis, all_test_loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE test loss")
    ax.set_title(phase+":test")
    fig.savefig('/home/mmoradi6/DDPM-06.30/DDPM-Pytorch-main/case_study/Classification model/checkpoints/'+phase+"test.png")   # save the figure to file
    plt.close(fig) 

if __name__ == "__main__":
    # Load data with augmentations
    transform = get_augmentation()
    dataset = datasets.ImageFolder(root=image_dir, transform=transform)
    
    # print("classes",dataset.classes)
    # print("Class to index mapping:", dataset.class_to_idx)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True,drop_last=True)
    test_transform=test_augmentation()
    test_data=datasets.ImageFolder(root=evaluation_dir,transform=test_transform)
    dataloader_test=DataLoader(test_data, batch_size=16, shuffle=True,drop_last=True)
    print("Class to index mapping:", test_data.class_to_idx)
    # Load model
    model = ModifiedVGG19().cuda()

    # Criterion for regression (Mean Squared Error)
    criterion = nn.MSELoss()

    # Training Stage 1: Train only fully connected layers
    for param in model.features.parameters():
        param.requires_grad = False  # Freeze the convolutional layers

    optimizer = optim.Adam(model.fc.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=1e-8)

    print("Training Stage 1: Training only FC layers")
    train(model, dataloader,dataloader_test, criterion, optimizer, num_epochs=100, phase='FC Only')

    # Training Stage 2: Train all layers
    for param in model.parameters():
        param.requires_grad = True  # Unfreeze all layers

    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

    print("Training Stage 2: Training all layers")
    train(model, dataloader,dataloader_test, criterion, optimizer, num_epochs=100, phase='All Layers')

