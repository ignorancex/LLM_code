from abc import ABC

from torch import nn
from tqdm import tqdm
from PIL import Image
from evaluators.abc import AbstractBaseEvaluator
from models import BLIP2Pipeline
import torch


class BLIP2Evaluator(AbstractBaseEvaluator, ABC):
    def __init__(self, models, dataloaders, configs, top_k=(1, 10, 50)):
        super().__init__(models, dataloaders, top_k)
        self.generator = torch.Generator(device=self.device).manual_seed(42)
        self.configs = configs

    def _extract_image_features(self, images):
        # assert output shape is (bs, embed_size)
        tar_img_emb = self.models(images)
        return tar_img_emb


    def _extract_composed_features(self, images, modifiers):
        # assert output shape is (bs, embed_size)
        composed_features = self.models(images, modifiers)
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
                # without img_transform, is PIL image
                # test_images = test_images.to(self.device)

                features = self._extract_image_features(test_images).cpu()
                features = features.reshape(features.shape[0], -1)

                all_test_features.extend(features)
                all_test_attributes.append(test_attr)

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
                # without img_transform and txt_transform, is PIL image and str
                # ref_images = ref_images.to(self.device)
                # modifiers, len_modifiers = modifiers.to(self.device), len_modifiers.to(self.device)

                composed_features = self._extract_composed_features(ref_images, modifiers).cpu()
                composed_features = composed_features.view(1, -1)
                all_composed_query_features.extend(composed_features)
                all_target_attributes.append(target_attribute)
                all_ref_attributes.append(ref_attribute)

                # test
                # if batch_idx > 50:
                #     break

        return torch.stack(all_composed_query_features), all_target_attributes, all_ref_attributes

    @classmethod
    def code(cls) -> str:
        return 'blip2_evaluator'
