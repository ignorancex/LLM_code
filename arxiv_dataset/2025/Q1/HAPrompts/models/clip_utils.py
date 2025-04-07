from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


# setup for clip image encoder
def _convert_image_to_rgb(image):
    return image.convert("RGB")

def custom_clip_transform(n_px, mean=None, std=None):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073) if mean is None else mean, 
            (0.26862954, 0.26130258, 0.27577711) if std is None else std)
    ])