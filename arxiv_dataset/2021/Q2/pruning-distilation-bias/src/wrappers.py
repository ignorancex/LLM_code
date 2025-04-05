import torch
from torchvision import models

class Student(torch.nn.Module):
    def __init__(self, model):
        super(Student, self).__init__()
        self.model = model
        self.activation = []
        self.sethooks()

    def forward(self, x):
        self.activation = []
        y_ = self.model(x)
        return self.activation, y_

    def get_activation(self, name, i):
        def hook(model, input, output):
            self.activation.append(output)
        return hook

    def sethooks(self):
        self.activation = []
        if isinstance(self.model, models.vgg.VGG):
            for i, layer in enumerate(self.model.features.children()):
                if isinstance(layer, torch.nn.MaxPool2d):
                    self.model.features[i].register_forward_hook(
                        self.get_activation('MaxPool2d', i))
        elif isinstance(self.model, models.resnet.ResNet):
            self.model.layer1.register_forward_hook(self.get_activation('layer1', 1))
            self.model.layer2.register_forward_hook(self.get_activation('layer2', 1))
            self.model.layer3.register_forward_hook(self.get_activation('layer3', 1))
            self.model.layer4.register_forward_hook(self.get_activation('layer4', 1))
        else:
            print("Check arch argument")



class Teacher(torch.nn.Module):
    def __init__(self, model):
        super(Teacher, self).__init__()
        self.model = model
        self.activation = []
        self.sethooks()

    def forward(self, x, is_feat=False):
        self.activation = []
        y_ = self.model(x)

        if is_feat:
            return self.activation, y_
        else:
            return y_

    def get_activation(self, name, i):
        def hook(model, input, output):
            self.activation.append(output)
        return hook

    def sethooks(self):
        self.activation = []
        if isinstance(self.model, models.vgg.VGG):
            for i, layer in enumerate(self.model.features.children()):
                if isinstance(layer, torch.nn.MaxPool2d):
                    self.model.features[i].register_forward_hook(
                        self.get_activation('MaxPool2d', i))
        elif isinstance(self.model, models.resnet.ResNet):
            self.model.layer1.register_forward_hook(self.get_activation('layer1', 1))
            self.model.layer2.register_forward_hook(self.get_activation('layer2', 1))
            self.model.layer3.register_forward_hook(self.get_activation('layer3', 1))
            self.model.layer4.register_forward_hook(self.get_activation('layer4', 1))
        else:
            print("Check arch argument")
        

class Dummy(torch.nn.Module):
    def __init__(self, model):
        super(Dummy, self).__init__()
        self.model = model
        self.activation = []

    def forward(self, x):
        self.activation = []
        y_ = self.model(x)
        return self.activation, y_
