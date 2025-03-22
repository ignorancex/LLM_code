import torch
import numpy as np
import torch.nn as nn

from clip import tokenize
from .BaseModel import BaseIncModel


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


class CustomCLIP(BaseIncModel):
    def __init__(self, classes_names, clip_model):
        super().__init__(classes_names, clip_model)
        self.classes_names = classes_names
        self.text_encoder = TextEncoder(clip_model)
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre
        self.vision_transformer = clip_model.visual.transformer
        self.ln_post = clip_model.visual.ln_post
        self.logit_scale = clip_model.logit_scale
        self.proj = clip_model.visual.proj
        self.text_features_per_task = []

    @torch.no_grad()
    def encode_current_text(self, template, device, num_old, num_cur):
        orig_mode = self.training
        self.eval()
        cur_classes = self.classes_names[num_old:num_cur]
        cur_text = torch.cat([tokenize(template.format(c)) for c in cur_classes])
        cur_text_features = self.encode_text(cur_text.to(device))
        cur_text_features = cur_text_features / cur_text_features.norm(dim=1, keepdim=True)
        self.text_features_per_task.append(cur_text_features)
        self.train(orig_mode)

    @torch.no_grad()
    def generate_soft_labels(self, num_cur, device, softness):
        orig_mode = self.training
        self.eval()
        cur_classes = self.classes_names[:num_cur]
        tokenized_labels = torch.cat([tokenize(f"{c}") for c in cur_classes])
        text_features = self.encode_text(tokenized_labels.to(device))
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        sim_matrix = text_features @ text_features.t()
        sim_matrix = sim_matrix.float()
        soft_labels = torch.exp(softness * sim_matrix) / torch.sum(torch.exp(softness * sim_matrix), dim=0)
        self.train(orig_mode)
        return soft_labels

    def calculate_inter_task_similarity(self, targets, device, prev_text_features):
        batched_labels = np.array(self.classes_names)[targets]
        tokenized_labels = torch.cat([tokenize(f"{c}") for c in batched_labels])
        batched_text_features = self.encode_text(tokenized_labels.to(device))
        batched_text_features = batched_text_features / batched_text_features.norm(dim=1, keepdim=True)
        sim_matrix = batched_text_features @ prev_text_features.t()
        return sim_matrix

    def encode_image(self, x, proj=True):
        x = self.conv1(x.type(self.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.vision_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if proj:
            x = x @ self.proj
        return x

    def encode_text(self, text):
        return self.text_encoder(text)

    def forward(self, images, text):
        image_features = self.encode_image(images)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def freeze_text_branch(self):
        for k, v in self.named_parameters():
            if 'text' in k:
                v.requires_grad = False

    def forward_features(self, x):
        return self.encode_image(x, proj=False)

    @torch.no_grad()
    def task_aware_pred(self, image_features, task_id):
        self.eval()
        taw_text_features = self.text_features_per_task[task_id]
        logit_scale = self.logit_scale.exp()
        taw_predictions = logit_scale * image_features @ taw_text_features.t()
        return taw_predictions

    @torch.no_grad()
    def task_agnos_pred(self, image_features):
        self.eval()
        tag_text_features = torch.cat(self.text_features_per_task, dim=0)
        logit_scale = self.logit_scale.exp()
        tag_predictions = logit_scale * image_features @ tag_text_features.t()
        return tag_predictions

    @torch.no_grad()
    def compute_mean(self, loader):
        device = next(self.parameters()).device
        self.eval()
        features = []
        labels = []
        for i, (images, captions, targets) in enumerate(loader):
            feature = self.forward_features(images.to(device))
            if feature.shape[0] == loader.batch_size:
                labels.append(targets.numpy())
                features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))

        prototype = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
        return prototype, class_label
