from abc import ABC

from torch import nn
from tqdm import tqdm
from PIL import Image
from evaluators.abc import AbstractBaseEvaluator
from models import StableDiffusionImg2ImgPipeline
import torch


class DiffusionEvaluator(AbstractBaseEvaluator, ABC):
    def __init__(self, models, dataloaders, configs, top_k=(1, 10, 50)):
        super().__init__(models, dataloaders, top_k)
        self.generator = torch.Generator(device=self.device).manual_seed(42)
        self.configs = configs

    def _extract_image_features(self, images):
        # extract test image features
        # clip image encoder pipeline is in the models folder, directly use it
        if "clip_image_model" in self.models:
            image_features = self.models["clip_image_model"](images)
            return image_features.reshape(image_features.shape[0], -1)

        # StableDiffusionImg2ImgPipeline is in the models directory
        if "diffusion" in self.models:
            image_features = self._diffusion_process(self.models["diffusion"], images, modifiers="image")
            return image_features.reshape(image_features.shape[0], -1)

        raise NotImplementedError("Not support other models for now")

    def _extract_composed_features(self, images, modifiers):
        composed_features = None
        if "clip_text_model" in self.models:
            text_features = self.models["clip_text_model"](modifiers)
            image_features = self.models["clip_image_model"](images)
            composed_features = image_features + text_features

        elif "diffusion" in self.models:
            diffusion_model: StableDiffusionImg2ImgPipeline = self.models["diffusion"]
            composed_features = self._diffusion_process(diffusion_model, images, "this clothes " + modifiers)
            if "clip_image_model" in self.models:
                # when use clip, composed_features is post processed image list
                composed_features = self.models["clip_image_model"](composed_features[0])

        if composed_features is None:
            raise NotImplementedError("Only support StableDiffusionImg2ImgPipeline model for now")
        return composed_features.reshape(composed_features.shape[0], -1)

    # extract diffusion process in a method
    def _diffusion_process(self, models: StableDiffusionImg2ImgPipeline, images, modifiers=None):
        features = models(prompt=modifiers, image=images, generator=self.generator, strength=self.configs['strength'],
                         num_inference_steps=self.configs['num_inference_steps'],
                         guidance_scale=self.configs['guidance_scale'], output_type=self.configs['output_type']).images
        return features

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
                ref_images = ref_images.to(self.device)
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
        return 'diffusion_evaluator'
