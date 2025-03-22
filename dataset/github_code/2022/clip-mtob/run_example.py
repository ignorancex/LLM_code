import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from bert_tokenizer import BertTokenizerWarper
import modeling_clip_helper_mtob, modeling_clip_helper_ref

class CLIPWarper(object):

    tokenizer = BertTokenizerWarper(max_len=77, checkpoint_dir='ERNIE-BASE-EN-2.0')
    transform_test = Compose([
        Resize(224, interpolation=Image.BICUBIC), CenterCrop(224),
        lambda image: image.convert("RGB"), ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])

    def __init__(self, model) -> None:
        if model is modeling_clip_helper_mtob.CLIP:
            self.model = modeling_clip_helper_mtob.CLIP(
                embed_dim=512, image_resolution=224, vision_layers=12, 
                vision_width=768, vision_patch_size=16, output_dim=32, 
                n_class_tokens=16, context_length=77, vocab_size=30522, 
                transformer_width=512, transformer_heads=8, transformer_layers=12).eval()
            self.model.load_state_dict(torch.load('vit-b16_mt-ob-32-16_e32_f32k_2.pt'))
            
        elif model is modeling_clip_helper_ref.CLIP:
            self.model = modeling_clip_helper_ref.CLIP(
                embed_dim=512, image_resolution=224, vision_layers=12, 
                vision_width=768, vision_patch_size=16, output_dim=512, 
                n_class_tokens=1, context_length=77, vocab_size=30522, 
                transformer_width=512, transformer_heads=8, transformer_layers=12).eval()
            self.model.load_state_dict(torch.load('vit-b16_e32_f32k.pt'))
        else:
            raise NotImplementedError
        
    def forward(self, list_of_text, one_image):
        input_ids = []
        attention_mask = []
        for init_text in list_of_text:
            text = CLIPWarper.tokenizer(init_text)
            input_ids.append(text['input_ids'])
            attention_mask.append(text['attention_mask'])

        input_ids = torch.from_numpy(np.stack(input_ids))
        attn = torch.from_numpy(np.stack(attention_mask))
        
        img = Image.open(one_image)
        img = CLIPWarper.transform_test(img).unsqueeze(0)

        return self.model.forward(img, (input_ids, attn))[0]

if __name__ == '__main__':
    clip_mtob = CLIPWarper(modeling_clip_helper_mtob.CLIP)
    clip_ref = CLIPWarper(modeling_clip_helper_ref.CLIP)

    text = ['cat', 'dog', 'asfasdf', 'a photo of cat', '23234234124', 'cccaaattt',
                        'a photo of nothing', 'catcatcatcatcat']
    image = 'Cat03.jpeg'

    print(clip_mtob.forward(text, image))
    print(clip_ref.forward(text, image))