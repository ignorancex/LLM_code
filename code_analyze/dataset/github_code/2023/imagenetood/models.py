import torchvision
from torchvision.models import *


def generate_torchvision_models(list_of_models):
    list_of_models['torchvision-model-0'] = lambda num_classes : torchvision.models.densenet121(
        DenseNet121_Weights(DenseNet121_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-1'] = lambda num_classes : torchvision.models.densenet201(
        DenseNet201_Weights(DenseNet201_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-2'] = lambda num_classes : torchvision.models.densenet161(
        DenseNet161_Weights(DenseNet161_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-3'] = lambda num_classes : torchvision.models.densenet169(
        DenseNet169_Weights(DenseNet169_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-4'] = lambda num_classes : torchvision.models.resnet152(
        ResNet152_Weights(ResNet152_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-5'] = lambda num_classes : torchvision.models.resnet101(
        ResNet101_Weights(ResNet101_Weights.IMAGENET1K_V1)
    )  
    list_of_models['torchvision-model-6'] = lambda num_classes : torchvision.models.resnext50_32x4d(
        ResNeXt50_32X4D_Weights(ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-7'] = lambda num_classes : torchvision.models.resnext101_32x8d(
        ResNeXt101_32X8D_Weights(ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
    ) 
    list_of_models['torchvision-model-8'] = lambda num_classes : torchvision.models.wide_resnet101_2(
        Wide_ResNet101_2_Weights(Wide_ResNet101_2_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-9'] = lambda num_classes : torchvision.models.wide_resnet50_2(
        Wide_ResNet50_2_Weights(Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-10'] = lambda num_classes : torchvision.models.regnet_x_32gf(
        RegNet_X_32GF_Weights(RegNet_X_32GF_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-11'] = lambda num_classes : torchvision.models.regnet_x_16gf(
        RegNet_X_16GF_Weights(RegNet_X_16GF_Weights.IMAGENET1K_V1)
    )
    list_of_models['torchvision-model-12'] = lambda num_classes : torchvision.models.resnet50(
        ResNet50_Weights(ResNet50_Weights.IMAGENET1K_V1)
    )
    
model_list = {}

generate_torchvision_models(model_list)
