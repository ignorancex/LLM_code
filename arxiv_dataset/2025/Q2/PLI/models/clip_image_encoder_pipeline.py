import clip
import torch

import torch.nn as nn
from PIL.Image import Image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class ClipImageEncoderPipeline(nn.Module):
    def __init__(self, clip_model_name='RN50.pt'):
        super().__init__()
        clip_model, pil_preprocess = clip.load("./models/clip/" + clip_model_name, device=device)
        clip_model.float().cuda().eval()
        self.model = clip_model.visual
        self.output_dim = clip_model.visual.output_dim
        self.dtype = clip_model.dtype
        self.input_resolution = clip_model.visual.input_resolution

        self.pil_preprocess = pil_preprocess # TODO: CLIP can accept raw PIL image not tensor, same as BLIP-2, Diffusion
        # pipeline accept dataset image is tensor, not PIL image
        self.preprocess = Compose([
            Resize(self.input_resolution, interpolation=BICUBIC),
            CenterCrop(self.input_resolution),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])


    def forward(self, image):
        processed_image = torch.tensor(self.preprocess(image)).to(device)
        with torch.no_grad():
            image_features = self.model(processed_image.reshape(-1,3,224,224))
        return image_features
