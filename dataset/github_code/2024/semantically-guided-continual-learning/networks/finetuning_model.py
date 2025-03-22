import torch

from clip import tokenize
from .BaseModel import BaseIncModel


class CustomCLIP(BaseIncModel):
    def __init__(self, classes_names, clip_model):
        super().__init__(classes_names, clip_model)
        self.classes_names = classes_names
        self.model = clip_model
        self.text_features_per_task = []

    @torch.no_grad()
    def build_classifier_from_text(self, template, device, num_old, num_cur):
        orig_mode = self.training
        self.eval()
        cur_classes = self.classes_names[num_old:num_cur]
        tokenized_text = torch.cat([tokenize(template.format(c)) for c in cur_classes])
        cur_text_features = self.model.encode_text(tokenized_text.to(device))
        cur_text_features = cur_text_features / cur_text_features.norm(dim=1, keepdim=True)
        self.text_features_per_task.append(cur_text_features)
        self.train(orig_mode)

    def encode_image(self, images):
        return self.model.encode_image(images)

    def encode_text(self, text):
        return self.model.encode_text(text)

    def forward(self, images, text):
        return self.model.forward(images, text)
