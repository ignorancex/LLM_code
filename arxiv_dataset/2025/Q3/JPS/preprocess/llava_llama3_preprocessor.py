from .base_preprocessor import BasePreprocessor
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torch

image_mean = (
    0.48145466,
    0.4578275,
    0.40821073
)

image_std = (
    0.26862954,
    0.26130258,
    0.27577711
)

class LlavaLlama3Preprocessor(BasePreprocessor):
    def __init__(self, config):
        self.config = config
    
    def preprocess(self, data, pipeline):
        for item in data:
            item['target'] = item['target'].replace('Sure,', '\nI am happy to help you with that!', 1).replace('here', 'Here', 1)
            # item['target'] = item['target'] + ':\nStep 1:'
            item['origin_question'] = self.config['jb_prompt'].format(attack_target=item['target'], attack_question=item['origin_question'])
            if 'rephrased_question' in item:
                item['rephrased_question'] = self.config['jb_prompt'].format(attack_target=item['target'], attack_question=item['rephrased_question'])
            if item['image'] is not None:
                item['image'] = self.load_image(item['image'], pipeline)
        
        return data

    def load_image(self,image_file, pipeline, input_size=448, max_num=12):
        if isinstance(image_file, str):
            image = Image.open(image_file).convert('RGB')
        else:
            image = image_file.convert('RGB')

        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.ToTensor(),
            T.Normalize(mean=image_mean, std=image_std)
        ])

        pixel_values = transform(image).unsqueeze(0).to(pipeline['model'].dtype).cuda()
        return pixel_values