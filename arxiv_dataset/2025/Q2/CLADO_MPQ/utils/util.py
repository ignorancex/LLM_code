import time,torch,torchvision,os
import torchvision.transforms as transforms
import numpy as np
from utils.layer import *

def get_loader(dset,batch_size,test_size=None,test_batch_size=500):
    if dset == 'MNIST':
        transf = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        # MNIST Dataset (Images and Labels)
        train_dataset = torchvision.datasets.MNIST(root =os.path.abspath('data'),train=True,transform=transf, download=True)
        test_dataset = torchvision.datasets.MNIST(root =os.path.abspath('data'),train=False,transform=transf, download=False)
        if test_size is not None and test_size < 10000:
            np.random.seed(42)
            sampled_index=np.random.choice(10000,test_size)
            test_dataset.data = torch.tensor(np.array(test_dataset.data)[sampled_index])
            test_dataset.targets = torch.tensor(np.array(test_dataset.targets)[sampled_index])


        # Dataset Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = True)
        test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size=test_batch_size, shuffle = False)

    elif dset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = torchvision.datasets.CIFAR10(root='data', train=False,download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,shuffle=False)
    
    elif dset == 'CIFAR100':
        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])
        train_dataset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = torchvision.datasets.CIFAR100(root='data', train=False,download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,shuffle=False)
        
    
    else:
        print('unrecogonized dataset')
        exit(1)

    return train_loader,test_loader

def generate_variation(model,noise_std=0):
    if noise_std > 0:
        for m in model.modules():
            if isinstance(m,qLinear) or isinstance(m,qConv2d):
                m.generate_variation(noise_std)
            
            
            
def evaluate(val_loader,model,noise_std=0,repeat=1,device='cuda',imgSize=32,
         imgFlat=False,debug=False,lossfunc=torch.nn.CrossEntropyLoss()):
    model.eval()
    loss_list = []
    acc_list = []
    start = time.time()
    for test in range(repeat):
        if noise_std > 0:
            generate_variation(model,noise_std=noise_std)
        # performance on testset
        correct = 0
        total = 0
        accumulative_loss = 0
        count = 0

        for t_images, t_labels in val_loader:
            count += 1
            if imgFlat:
                t_images = t_images.view(-1,imgSize**2)
            t_images = t_images.to(device)
            t_outputs = model(t_images)
            t_labels = t_labels.to(device)
            t_loss = lossfunc(t_outputs,t_labels)
            accumulative_loss += t_loss.data.item()
            _, t_predicted = torch.max(t_outputs.data, 1)
            total += t_labels.size(0)
            correct += (t_predicted == t_labels).sum()
        acc = (correct.data.item()/ total)
        loss_list.append(accumulative_loss/count)
        acc_list.append(acc)

        if debug:
            print("test %s/%s [%s batches %.4f seconds]:"%(test+1,repeat,count,time.time()-start))
            start = time.time()
            print("loss %.4f acc %.4f"%(accumulative_loss/count,acc))
    end = time.time()
    loss_list = np.array(loss_list)
    acc_list = np.array(acc_list)
    # statistics
    qtl_loss = np.quantile(loss_list,0.95)
    mean_loss = loss_list.mean()
    qtl_acc = np.quantile(acc_list,0.05)
    mean_acc = acc_list.mean()

    return {'mean_acc':mean_acc,'qtl_acc':qtl_acc,'mean_loss':mean_loss,'qtl_loss':qtl_loss,
            'test time':end-start,'acc_list':acc_list,'loss_list':loss_list}