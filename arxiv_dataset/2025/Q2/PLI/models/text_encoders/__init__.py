from models.text_encoders.blip_text_encoder import BlipTextEncoder
from models.text_encoders.clip_text_encoder import ClipTextEncoder

def text_encoder_factory(configs: dict):
    model_code = configs['text_encoders']
    if model_code == BlipTextEncoder.code():
        return BlipTextEncoder(configs['blip_model_name'], configs['blip_model_type'])
    elif model_code == ClipTextEncoder.code():
        return ClipTextEncoder(configs['clip_text_model'])

    raise ValueError("There's no text encoder matched with {}".format(model_code))