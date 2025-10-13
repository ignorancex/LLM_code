from models.image_encoders.blip_image_encoder import BlipImageEncoder
from models.image_encoders.clip_image_encoder import ClipImageEncoder


def image_encoder_factory(configs: dict):
    model_code = configs['image_encoders']
    if model_code == BlipImageEncoder.code():
        return BlipImageEncoder(configs['blip_model_name'], configs['blip_model_type'])
    elif model_code == ClipImageEncoder.code():
        return ClipImageEncoder(configs['clip_image_model'])
    raise ValueError("There's no image encoder matched with {}".format(model_code))