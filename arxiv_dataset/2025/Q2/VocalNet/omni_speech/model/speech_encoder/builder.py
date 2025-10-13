from .speech_encoder import WhisperWrappedEncoder
def build_speech_encoder(config):
    speech_encoder_type = getattr(config, 'speech_encoder_type', None)
    if speech_encoder_type is None:
        raise ValueError('speech_encoder_type must be specified in the configuration.')
    if "whisper" == speech_encoder_type.lower():
        return WhisperWrappedEncoder.load(config)

    raise ValueError(f'Unknown speech encoder: {speech_encoder_type}')

