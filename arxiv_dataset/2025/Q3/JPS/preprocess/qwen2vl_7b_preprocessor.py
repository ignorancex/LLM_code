from .base_preprocessor import BasePreprocessor
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torch

class Qwen2VL7BPreprocessor(BasePreprocessor):
    def __init__(self, config):
        self.config = config
    
    def preprocess(self, data, pipeline):
        for item in data:
            item['target'] = item['target'].replace('Sure,', 'I am happy to help you with that.', 1).replace('here', 'Here', 1)
            # item['target'] = item['target'] + ':\nStep 1:'
            # item['target'] = item['target'].replace('Sure, ', '', 1).replace('here', 'Here', 1)
            item['origin_question'] = self.config['jb_prompt'].format(attack_target=item['target'], attack_question=item['origin_question'])

            if 'rephrased_question' in item:
                item['rephrased_question'] = self.config['jb_prompt'].format(attack_target=item['target'], attack_question=item['rephrased_question'])

        return data