import torch
import numpy as np
from transformers import SpeechT5FeatureExtractor


# [V2A] SuperModel including all networks to be trained for enabling gradient accumulation and DeepSpeed
class TrainingModel(torch.nn.Module):
    def __init__(self, 
                 unet_svg, 
                 unet_a, 
                 unet_v, 
                ):
        super().__init__()

        self.unet_svg = unet_svg
        self.unet_a = unet_a
        self.unet_v = unet_v
    
    def forward(self, x, timesteps, context, class_labels, xtype=["audio", "video"], x_con=[None, None], t_con=[None, None], block_connection=None):
        noise_pred = self.unet_svg([self.unet_a, self.unet_v], x, timesteps, context, class_labels, xtype, x_con=x_con, t_con=t_con, block_connection=block_connection)
        return noise_pred
    

class logmel_extractor():
    def __init__(self, sampling_rate):
        self.base_extractor = SpeechT5FeatureExtractor(sampling_rate=sampling_rate,
                                                num_mel_bins=64,
                                                hop_length=160*1000//sampling_rate, 
                                                win_length=1024*1000//sampling_rate, 
                                                fmin=0, 
                                                fmax=8000)
        self.sampling_rate = sampling_rate

    def __call__(self, audio, duration_per_sample):
        mel_features_dict = self.base_extractor(audio_target=audio,
                                        sampling_rate=self.sampling_rate, 
                                        max_length=self.sampling_rate//160*duration_per_sample,
                                        truncation=True)
        mel_features = 10 ** mel_features_dict["input_values"]
        mel_features = np.log(np.clip(mel_features, 1e-5, None))
        # mel_features = 10 ** torch.from_numpy(mel_features_dict["input_values"])
        # mel_features = torch.log(torch.clamp(mel_features.to(accelerator.device, dtype=weight_dtype), min=1e-5)).unsqueeze(1)

        return mel_features
