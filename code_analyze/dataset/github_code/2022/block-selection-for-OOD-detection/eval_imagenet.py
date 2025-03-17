import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import argparse

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--net','-n', default = 'resnet50', type=str)
parser.add_argument('--gpu', '-g', default = '0', type=str)
parser.add_argument('--save_path', '-s', default='.', type=str)
parser.add_argument('--method' ,'-m', default = 'featurenorm', type=str)
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


def calculate_norm(model, loader, device):
    #FeatureNorm from the selected block
    if type(model).__name__ == 'ResNet':  
        forward_features = forward_feature_resnet50
    elif type(model).__name__ == 'VGG':  
        forward_features = forward_feature_vgg16
    elif type(model).__name__ == 'MobileNetV3': 
        forward_features = forward_feature_mobilenetv3

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            x = inputs.to(device)         
            # ResNet
            features = forward_features(model, x)
            features = features[model.sblock]

            # Norm calculation
            norm = torch.norm(F.relu(features), dim=[2, 3]).mean(1)
            predictions.append(norm)
    predictions = torch.cat(predictions).to(device)
    return predictions            

def calculate_msp(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            x = inputs.to(device)         
            x = model(x)
            x = torch.softmax(x, dim=1).max(dim=1).values
            predictions.append(x)
    predictions = torch.cat(predictions).to(device)
    return predictions   

if args.method == 'msp':
    calculate_score = calculate_msp
elif args.method == 'featurenorm':
    calculate_score = calculate_norm


def OOD_results(preds_id, model, loader, device, method, file):  
    #image_norm(loader)
    preds_ood = calculate_score(model, loader, device).cpu()

    print(torch.mean(preds_ood), torch.mean(preds_id))
    show_performance(preds_id, preds_ood, method, file=file)
    
def eval():
    device = 'cuda:'+args.gpu
    num_classes = 1000

    if 'resnet50' == args.net:
        model = torchvision.models.resnet50(pretrained=True, num_classes=1000)
        model.sblock = 14
    if 'vgg16' == args.net:
        model = torchvision.models.vgg16(pretrained=True, num_classes=1000)
        model.sblock = 4
    if 'mobilenetv3' == args.net:
        model = torchvision.models.mobilenet_v3_large(pretrained=True, num_classes=1000)
        model.sblock = 16
    model.to(device)
    model.eval()

    config = read_conf('conf/imagenet.json')
    
    _, valid_loader = get_imagenet(config['id_dataset'], 32)

    f = open('{}/{}_result.txt'.format(args.save_path, args.net), 'w')
    valid_accuracy = validation_accuracy(model, valid_loader, device)
    print(valid_accuracy)
    f.write('Accuracy for ValidationSet: {}\n'.format(str(valid_accuracy)))

    preds_in = calculate_score(model, valid_loader, device).cpu()
    OOD_results(preds_in, model, get_ood('./OOD_for_ImageNet/iNaturalist', for_imagenet=True), device, args.method+'-SVHN', f) # iNaturalist
    OOD_results(preds_in, model, get_ood('./OOD_for_ImageNet/SUN', for_imagenet=True), device, args.method+'-SUN', f) # SUN
    OOD_results(preds_in, model, get_ood('./OOD_for_ImageNet/Places', for_imagenet=True), device, args.method+'-PLACES', f) # PLACES
    OOD_results(preds_in, model, get_ood('./OOD_for_ImageNet/dtd/images', for_imagenet=True), device, args.method+'-Textures', f) #TExtures
    f.close()


if __name__ =='__main__':
    eval()