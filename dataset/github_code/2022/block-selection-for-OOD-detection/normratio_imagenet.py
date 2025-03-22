import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from utils import *
import resnet
import wrn
import vgg

parser = argparse.ArgumentParser()
parser.add_argument('--net','-n', default = 'resnet50', type=str)
parser.add_argument('--gpu', '-g', default = '0', type=str)
parser.add_argument('--save_path', '-s', default='.', type=str)
args = parser.parse_args()

def forward_feature_resnet50(model, x):
    features = []
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    for i in range(3):
        x = model.layer1[i](x)
        features.append(x)

    for i in range(4):
        x = model.layer2[i](x)    
        features.append(x)

    for i in range(6):
        x = model.layer3[i](x)
        features.append(x)
    
    for i in range(3):
        x = model.layer4[i](x)
        features.append(x)
    return features

def forward_feature_vgg16(model, x):
    layers = [64, 'r', 64, 'r', "M", 128, 'r', 128, 'r', "M", 256, 'r', 256, 'r', 256, 'r', "M", 512, 'r', 512, 'r', 512, 'r', "M", 512, 'r', 512, 'r', 512, 'r', "M"]
    features = []

    for i, layer in enumerate(layers):
        x = model.features[i](x)
        if layer == 'M':
            features.append(x)
    return features

def forward_feature_mobilenetv3(model, x):
    features = []
    for i, layer in enumerate(model.features):
        # print(layer, type(layer).__name__)
        x = model.features[i](x)
        features.append(x)
    return features

def calculate_layer(model, train_loader, jigsaw_loader, device):
    model.eval()
    if type(model).__name__ == 'ResNet':  
        forward_features = forward_feature_resnet50
        num_blocks = 16
    elif type(model).__name__ == 'VGG':  
        forward_features = forward_feature_vgg16
        num_blocks = 5
    elif type(model).__name__ == 'MobileNetV3': 
        forward_features = forward_feature_mobilenetv3
        num_blocks = 17

    norm_pred_ori = dict()
    norm_pred_jigsaw = dict()
    for i in range(num_blocks):
        norm_pred_ori[i] = []
        norm_pred_jigsaw[i] = []

    print(type(model).__name__, len(train_loader), len(jigsaw_loader))
    with torch.no_grad():
        for batch_idx, (data1, data2) in enumerate(zip(train_loader, jigsaw_loader)):
            if batch_idx > 100: # For fast calculation
                break
            x = torch.cat([data1[0], data2[0]], dim=0).to(device)

            features =  forward_features(model, x)  

            for i in range(num_blocks):        
                norm = torch.norm(F.relu(features[i]), dim=[2,3]).mean(1)
                norm_ori = norm[:len(data1[0])]
                norm_jigsaw = norm[len(data1[0]):]

                norm_pred_ori[i].append(norm_ori)
                norm_pred_jigsaw[i].append(norm_jigsaw)

    for i in range(num_blocks):
        norm_pred_ori[i] = torch.cat(norm_pred_ori[i], dim=0)
        norm_pred_jigsaw[i] = torch.cat(norm_pred_jigsaw[i], dim=0)

        print('NormRatio-Block{}: {}'.format(i, (norm_pred_ori[i]/norm_pred_jigsaw[i]).mean())) 

        
def eval():
    config = read_conf('conf/imagenet.json')
    device = 'cuda:'+args.gpu
    dataset_path = config['id_dataset']
    batch_size = config['batch_size']
    
    train_loader, jigsaw_loader = get_imagenet_jigsaw(dataset_path, batch_size)

   
    if 'resnet50' == args.net:
        model = torchvision.models.resnet50(pretrained=True, num_classes=1000)
    if 'vgg16' == args.net:
        model = torchvision.models.vgg16(pretrained=True, num_classes=1000)
    if 'mobilenetv3' == args.net:
        model = torchvision.models.mobilenet_v3_large(pretrained=True, num_classes=1000)
    model.to(device)
    model.eval()
    
    calculate_layer(model, train_loader, jigsaw_loader, device)


if __name__ =='__main__':
    eval()