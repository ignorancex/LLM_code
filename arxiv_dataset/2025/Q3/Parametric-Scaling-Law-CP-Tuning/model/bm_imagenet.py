import torchvision.models as models

def build_model_imagenet(model_name):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "densenet121":
        model = models.densenet121(weights="IMAGENET1K_V1", progress=True)
    elif model_name == "vgg16":
        model = models.vgg16_bn(weights="IMAGENET1K_V1", progress=True)
    else:
        raise ValueError("This models is not supported.")
    model.eval()
    return model







