from featup.core.dinov2 import DINOv2UpFeatBackbone


def get_model(name):
    """ Get a model with specified name """
    if name == "dinov2":
        return DINOv2UpFeatBackbone.load_backbone(pretrained=True, use_norm=True)
    else:
        raise ValueError("Invalid model")
