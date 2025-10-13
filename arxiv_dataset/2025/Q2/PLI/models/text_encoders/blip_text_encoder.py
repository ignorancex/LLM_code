from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from models.text_encoders.abc import AbstractBaseTextEncoder
from lavis.models import load_model_and_preprocess

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class BlipTextEncoder(AbstractBaseTextEncoder):
    def __init__(self, blip_model_name, model_type):
        super().__init__()
        blip_model, _, _ = load_model_and_preprocess(blip_model_name, model_type=model_type, device=device,
                                                     is_eval=True)
        self.tokenizer = blip_model.tokenizer # tokenize is finish in text_transforms.py
        self.text_encoder = blip_model.text_encoder
        self.text_proj = blip_model.text_proj
        self.max_txt_len = blip_model.max_txt_len
        self.feature_size = self.text_proj.in_features
        self.temp = blip_model.temp
        # save memory
        del blip_model
        gc.collect()

    def forward(self, input_ids, image_embeds=None):
        input_ids = input_ids.squeeze()
        if image_embeds is None:  # return text features
            text_output = self.text_encoder(
                input_ids,
                return_dict=True,
                mode="text",
            )
            text_embeds = text_output.last_hidden_state

            text_features = self.text_proj(text_embeds)[:, 0, :]  # [cls] token
            # text_features = F.normalize(text_features, dim=-1)

            return text_features

        # multimodal
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        attention_mask = torch.tensor(input_ids != self.tokenizer.pad_token_id, dtype=torch.long).to(image_embeds.device)
        input_ids[:, 0] = self.tokenizer.enc_token_id

        output = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        multimodal_embeds = output.last_hidden_state  # [batch_size, max_txt_len, embed_dim]

        return multimodal_embeds[:, 0, :]

    @classmethod
    def code(cls):
        return "blip_text_encoder"

    def layer_shapes(self):
        return {'output_dim': self.text_proj.out_features}
