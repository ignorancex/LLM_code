
from .speech_generator_ar_mtp import SpeechGeneratorARMTP

def build_speech_generator(config):
    generator_type = getattr(config, 'speech_generator_type')
    if 'ar_mtp' in generator_type:
        return SpeechGeneratorARMTP(config)
    raise ValueError(f'Unknown generator type: {generator_type}')
