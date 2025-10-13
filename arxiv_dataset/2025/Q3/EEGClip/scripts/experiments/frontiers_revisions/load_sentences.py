from cProfile import label

import numpy as np
from EEGClip.clip_models import EEGClipModel
import torch
import json
import configs.preprocess_config as preprocess_config
from transformers import AutoModel, AutoTokenizer

#eegclipmodel = EEGClipModel()
eegclipmodel = EEGClipModel.load_from_checkpoint(
            preprocess_config.model_paths["eegclip"],
            strict=False,
        )


EEGEncoder = torch.nn.Sequential(
    eegclipmodel.eeg_encoder, eegclipmodel.eeg_projection
)

for param in EEGEncoder.parameters():
        param.requires_grad = False

print(EEGEncoder)

text_projection = eegclipmodel.text_projection
text_projection.eval()

text_encoder_name = "mixedbread-ai/mxbai-embed-large-v1"
label_name = "pathological"



with open(preprocess_config.zc_sentences_emb_dict_path, "r") as f:
            zc_sentences_emb_dict = json.load(f)
print(zc_sentences_emb_dict[text_encoder_name])
emb_dict = zc_sentences_emb_dict[text_encoder_name][label_name]
s0, s1 = (
                    emb_dict["s0"],
                    emb_dict["s1"],
                )

s0 = torch.tensor(s0, dtype=torch.float32)
# put on GPU if available
if torch.cuda.is_available():
    s0 = s0.cuda()

print(f"Sentence 0 shape: {s0.shape}")



s0 = text_projection(s0.unsqueeze(0))

print(f"Sentence 0 after projection: {s0.shape}")