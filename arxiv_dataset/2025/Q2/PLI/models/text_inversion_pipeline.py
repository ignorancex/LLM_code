import torch
import torch.nn as nn
import torch.nn.functional as F
from models.clip_text_encoder_pipeline import ClipTextEncoderPipeline
from models.clip_image_encoder_pipeline import ClipImageEncoderPipeline
# from lavis.models import load_model_and_preprocess
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import os


device = 'cuda' if torch.cuda.is_available() else 'cpu'
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(os.path.join(ROOT_DIR, "blip2-flan-t5-xl"))


class TextInversionPipeline(nn.Module):
    def __init__(self, text_encoder_name, image_encoder_name, blip_model_name, model_type):
        super().__init__()
        self.text_encoder = ClipTextEncoderPipeline(text_encoder_name)
        self.image_encoder = ClipImageEncoderPipeline(image_encoder_name)

        # self.blip2, self.vis_processors, _ = load_model_and_preprocess(blip_model_name, model_type=model_type, device=device, is_eval=True)
        # text inversion model by blip2 from transformers
        self.blip2 = Blip2ForConditionalGeneration.from_pretrained(os.path.join(ROOT_DIR, blip_model_name), local_files_only=True)
        self.processor = Blip2Processor.from_pretrained(os.path.join(ROOT_DIR, blip_model_name), local_files_only=True)

        # control the strength of the inversion text matching
        self.inversion_bias = 0.1

    def forward(self, image, text=None):
        clip_img_features = F.normalize(self.image_encoder(image), dim=-1)
        if text is not None: # reference image
            # transformer style inversion text
            img = self.processor(images=image, return_tensors="pt").to(device)
            generate_ids = self.blip2.generate(**img)
            inversion_text = self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
            # img = self.vis_processors["eval"](image).unsqueeze(0).to(device)
            # inversion_text = self.blip2.generate(img) # list of string
            clip_txt_features = F.normalize(self.text_encoder(inversion_text), dim=-1) * self.inversion_bias
            return clip_img_features + clip_txt_features + F.normalize(self.text_encoder(text), dim=-1)
        return clip_img_features
