from abc import ABC

from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from PIL import Image
from evaluators.abc import AbstractBaseEvaluator
from models import BLIP2Pipeline
from trainers.pretrainer import encode_with_pseudo_tokens
import torch
import clip


class PretrainerEvaluator(AbstractBaseEvaluator, ABC):
    def __init__(self, models: dict, dataloaders, configs, top_k=(1, 10, 50)):
        super().__init__(models, dataloaders, top_k)
        self.generator = torch.Generator(device=self.device).manual_seed(42)
        self.configs = configs

        self.image_encoder = self.models['image_encoder']
        self.text_encoder = self.models['text_encoder'] # Usually, text encoder can fuse image and text for reference images and texts
        self.compositor = self.models['compositor'] # So compositor usually follows text encoder
        self.compositor_code = configs['compositor'] if 'compositor' in configs else "None"

        self.img_weight = configs['img_weight'] if 'img_weight' in configs else 0.5
    def _extract_image_features(self, images):
        # assert output shape is (bs, embed_size)
        tar_img_emb = self.image_encoder(images, proj=True) # (batch_size, n, hidden_size)
        tar_features = tar_img_emb # [cls] token, (batch_size, hidden_size)
        return tar_features


    def _extract_composed_features(self, images, modifiers):
        # assert output shape is (bs, embed_size)
        ref_img_emb = self.image_encoder(images, proj=True) # (batch_size, n, hidden_size)
        if self.compositor_code == 'Combiner': # multigpu may not work
            composed_features = self.compositor.combine_features(ref_img_emb, self.text_encoder(modifiers))
        elif self.compositor_code == 'phi':
            extimated_tokens = self.compositor(ref_img_emb)
            prompt = "a photo of $"  # The $ is a special token that will be replaced with the pseudo tokens
            tokenized_prompt = clip.tokenize([prompt] * len(images)).to(self.device)
            composed_features = encode_with_pseudo_tokens(self.text_encoder, tokenized_prompt, extimated_tokens)
        else:
            composed_features = F.normalize(self.text_encoder(modifiers), dim=-1) + F.normalize(ref_img_emb, dim=-1) * self.img_weight
            composed_features = self.compositor(composed_features)
        return composed_features

    def extract_test_features_and_attributes(self):
        """
        override this method from AbstractBaseEvaluator
                Returns: (1) torch.Tensor of all test features, with shape (N_test, Embed_size)
                        (2) list of test attributes, Size = N_test
        """

        dataloader = tqdm(self.test_samples_dataloader)
        all_test_attributes = []
        all_test_features = []
        with torch.no_grad():
            # Use Pipeline, dataloader is dataset, not batch
            for batch_idx, (test_images, test_attr) in enumerate(dataloader):
                test_images = test_images.to(self.device)

                features = self._extract_image_features(test_images).cpu()
                features = features.reshape(features.shape[0], -1)

                all_test_features.extend(features)
                all_test_attributes.extend(test_attr)

        return torch.stack(all_test_features), all_test_attributes

    def extract_query_features_and_attributes(self):
        """
        override this method from AbstractBaseEvaluator
            Returns: (1) torch.Tensor of all query features, with shape (N_query, Embed_size)
                    (2) list of target attributes, Size = N_query
            """

        dataloader = tqdm(self.test_query_dataloader)  # Use Pipeline, dataloader is dataset, not batch
        all_target_attributes = []
        all_ref_attributes = []
        all_composed_query_features = []

        with torch.no_grad():
            for batch_idx, (ref_images, ref_attribute, modifiers, target_attribute, len_modifiers) in enumerate(
                    dataloader):
                ref_images = ref_images.to(self.device)
                modifiers = modifiers.to(self.device)

                composed_features = self._extract_composed_features(ref_images, modifiers).cpu() # (batch_size, hidden_size)
                composed_features = composed_features.view(composed_features.shape[0], -1)

                all_composed_query_features.extend(composed_features)
                all_target_attributes.extend(target_attribute) # target_attribute is a list of str, so extend
                all_ref_attributes.extend(ref_attribute)

        return torch.stack(all_composed_query_features), all_target_attributes, all_ref_attributes

    @classmethod
    def code(cls) -> str:
        return 'pretrainer_evaluator'
