# Adopted from https://github.com/ddlBoJack/SLAM-LLM/blob/main/src/slam_llm/models/encoder.py

import json
import torch
import torch.nn as nn
import numpy as np
import whisper
import pdb

from safetensors.torch import load_file
from omni_speech.model.speech_tokenizer.modeling_whisper import WhisperVQEncoder,WhisperVQConfig

import sys
import os
sys.path.insert(0, os.path.abspath('/root/SpeechLLMs/omni_speech/model/speech_encoder'))
from funasr import AutoModel
from typing import Tuple


class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_config):

        def replace_layer_norm(module):
            from whisper.model import LayerNorm
            for name, child in module.named_children():
                if isinstance(child, LayerNorm):
                    old_params = child.state_dict()
                    new_layer_norm = nn.LayerNorm(child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine)
                    new_layer_norm.load_state_dict(old_params)
                    setattr(module, name, new_layer_norm)
                else:
                    replace_layer_norm(child)


        from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

        whisper = pipeline("automatic-speech-recognition", model_config.speech_encoder, torch_dtype=torch.bfloat16, device="cpu",chunk_length_s=30, batch_size=256)
        encoder = whisper.model.get_encoder()
        
        replace_layer_norm(encoder)
        return encoder

class WhisperModelLoader:
    @classmethod
    def load(cls, model_config):
        # whisper_model = WhisperForConditionalGeneration.from_pretrained(model_config.speech_encoder).model.encoder.to(torch.bfloat16).to('cpu')
        
        Whisper_config = WhisperVQConfig.from_pretrained(model_config.speech_encoder)
        whisper_model = WhisperVQEncoder(config=Whisper_config).eval().to('cpu')
        pretrained_state_dict = load_file(model_config.speech_encoder + '/model.safetensors')
        whisper_model.load_state_dict(pretrained_state_dict)
        whisper_model.to(torch.bfloat16)
        
        return whisper_model





