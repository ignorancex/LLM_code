import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
import argparse
from itertools import chain

from src.dataloader import make_data_loader
from src.trainers import eval_model, first_eval, train_student_kd
from src.trainers import train_baseline
from src.wrappers import Student, Teacher
from src.model_loader import load_model
from src.prune_scheduler import AgpPruningRate
from src.losses import  SinkhornLoss

parser = argparse.ArgumentParser()


parser.add_argument('-ep', '--epochs', type=int, default=20)
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
parser.add_argument('-a', '--alpha', type=float, default=1.0)
parser.add_argument('-n', '--normalized', type=bool, default=False)
parser.add_argument('-bs', '--batch_size', type=int, default=16)
parser.add_argument('-ts', '--target_sparsity', type=float, default=0.3)
parser.add_argument('-lp', '--log_path', type=str, default='/tmp/')
parser.add_argument('-ikl', '--inner_knowledge_loss', type=str, choices=['l2', 'REM'], default='l2')

parser.add_argument('-ar', '--arch', type=str,
                    choices=['vgg16', 'resnet18'], default='vgg16')

parser.add_argument('-mp', '--model_path', type=str,
                    default='models/baseline/cifar10/vgg16.pth',
                    help='Path to model weights')

parser.add_argument('-ds', '--dataset', type=str, default='cifar10',
                    choices=[
                        'cifar10', 'cifar100', 'food101',
                        'imagenette', 'imagewoof', 'imagenet'
                        ],
                    help='Dataset for training')

parser.add_argument('-ps', '--prune_strat', type=str,
                    choices=['struct_mag', 'ik_struct_mag', 'ik_ss_struct_mag',
                             'finegrain_mag', 'ik_finegrain_mag'],
                    default='ik_struct_mag')

args = parser.parse_args()

epochs = args.epochs
loss = args.inner_knowledge_loss
lr = args.learning_rate
bs = args.batch_size
target_sparsity = args.target_sparsity
arch = args.arch
strat = args.prune_strat
log_path = args.log_path
alpha = args.alpha
normalized = args.normalized
model_path = args.model_path
dataset = args.dataset
freq = 2
prune_end = epochs // 2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataloaders, dataset_sizes = make_data_loader(bs, dataset=dataset)


student, teacher = load_model(path=model_path, dataset=dataset, model=arch)
student = Student(student)
teacher = Teacher(teacher)

student = student.to(device)
teacher = teacher.to(device)

criterion = nn.CrossEntropyLoss()

if loss == 'l2':
    inner_criterion = nn.MSELoss()
else:
    inner_criterion = SinkhornLoss()

if loss != 'l2':
    pass_strat = strat + '/{}'.format(loss)
else:
    pass_strat = strat

model_path = '{}/{}/{}/lr_{}/alpha_{}/sparsity_{}/'.format(
                                            dataset, arch, pass_strat,
                                            lr, alpha, target_sparsity)

if loss != 'l2':
    model_path = model_path + '/{}/'.format(loss)


model_id = model_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = './models/' + model_path

if normalized and 'ik' in strat:
    lr = lr / alpha

optimizer_ft = optim.SGD(student.parameters(), lr=lr, momentum=0.9)

exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft,
                                            milestones=[
                                                np.ceil(epochs*.33),
                                                np.ceil(epochs*.66)],
                                            gamma=.1)


if not os.path.exists(save_path):
    os.makedirs(save_path)

writer = SummaryWriter(log_dir=log_path + model_id)


print('acc pre pruning')
class_acc = first_eval(student, criterion, dataloaders,
                       dataset_sizes, writer, epoch=0)


prune_sch = AgpPruningRate(.1, target_sparsity, 1, prune_end, freq)
prune_layers = list(chain(student.model.features, student.model.classifier[:-1]))

for epoch in range(1, epochs + 1):

    if epoch % freq == 1 and epoch <= prune_end:
        target = prune_sch.step(epoch)
        print(f'pruning {target * 100}% sparsity')
        if epoch > 1 and epoch < prune_end:
            for i, layer in enumerate(prune_layers):
                if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                    prune.remove(layer, "weight")
        for i, layer in enumerate(prune_layers):
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                if "struct" in strat:
                    prune.ln_structured(layer, name="weight",
                                        amount=float(target), n=1, dim=0)
                elif 'finegrain' in strat:
                    prune.l1_unstructured(layer, name='weight',
                                          amount=float(target))

                layer_spar = float(torch.sum(layer.weight == 0))
                layer_spar /= float(layer.weight.nelement())
                print(f"Sparsity in layer {i}: {layer_spar: 3f}")
    elif epoch > prune_end:
        print("All done pruning")

    if strat == 'ik_ss_struct_mag':
        train_student_kd(
            student, teacher, criterion,
            inner_criterion, dataloaders, dataset_sizes,
            optimizer_ft, exp_lr_scheduler, writer,
            epoch, phase='ss_train')

    if 'ik' in strat:
        train_student_kd(
                student, teacher, criterion,
                inner_criterion, dataloaders, dataset_sizes,
                optimizer_ft, exp_lr_scheduler, writer,
                epoch, phase='train', alpha=alpha)

    else:
        train_baseline(
                student, criterion, dataloaders,
                dataset_sizes, optimizer_ft, exp_lr_scheduler,
                writer, epoch, phase='train')

    epoch_acc = eval_model(
                    student, criterion, dataloaders,
                    dataset_sizes, writer, epoch=epoch,
                    class_acc=class_acc)


writer.close()
torch.save(student.state_dict(), './models/' + model_id + '.pt')
