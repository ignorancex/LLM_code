import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

def get_model(args):
    if args.model == 'resnet18':
        model = models.resnet18(pretrained=False)
        if args.datasets == 'cifar10':
            model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.model == 'cnn':
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
                self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.fc1 = nn.Linear(320, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, x):
                x = F.relu(F.max_pool2d(self.conv1(x), 2))
                x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
                x = x.view(-1, 320)
                x = F.relu(self.fc1(x))
                x = F.dropout(x, training=self.training)
                x = self.fc2(x)
                return F.log_softmax(x, dim=0)
        model = Net()
    elif args.model == 'vgg13':
        model = models.vgg13(pretrained = False)
        if args.datasets == 'cifar10':
            model.classifier[6] = nn.Linear(4096, 10)

    return model