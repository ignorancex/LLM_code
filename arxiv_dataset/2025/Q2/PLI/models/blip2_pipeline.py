import torch
import torch.nn as nn
import torch.nn.functional as F
# lavis is conflict with torch transformers > 4.27.0
# from transformers import Blip2Processor, Blip2Model
from lavis.models import load_model_and_preprocess, Blip2Qformer
from peft import LoraConfig, get_peft_model, PeftModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'
class BLIP2Pipeline(nn.Module):
    def __init__(self, bilp2_model_name, model_type):
        """
        Args:
            bilp2_model_name: str, name of the BLIP2 model
            model_type (str): type of the model.
        """
        super().__init__()
        # lavis blip2 model
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name=bilp2_model_name, model_type=model_type, device=device, is_eval=True)


        # transformer provided blip2 model without get_input_embeddings(), word embedding is not available
        # self.processor = Blip2Processor.from_pretrained(bilp2_model_name)
        # self.model: Blip2Model = Blip2Model.from_pretrained(bilp2_model_name)
    def forward(self, image, text=None):
        """
        Args:
            image:
            text:
            mode: in ["multimodal", "image", "text"], default multimodal return multi-modal features, else return uni-modal features
        Returns: BlipOutputFeatures
        """
        # lavis blip2 feature extraction
        img = self.vis_processors["eval"](image).unsqueeze(0).to(device)
        # -----------------
        # Use multimodal features
        if text is None:
            img_feature = self.model.extract_features({"image": img, "text_input": "a photo of"}, mode="multimodal").multimodal_embeds  # (N, 32, 768)
            img_feature = self.model.vision_proj(img_feature)
            return F.normalize(img_feature[:, 0, :], dim=-1)
        txt = self.txt_processors["eval"](text)
        # Use multimodal features
        composed_features = self.model.extract_features({"image": img, "text_input": txt}, mode="multimodal").multimodal_embeds  # (N, 32, 768)
        composed_features = self.model.vision_proj(composed_features)
        return F.normalize(composed_features[:, 0, :], dim=-1)
        # -----------------

        img_feature = self.model.extract_features({"image": img}, mode="image").image_embeds_proj  # (N, 32, 768)
        if text is None:
            # TODO: transform tar_img_emb
            # return img_feature.mean(dim=1) # bad performance
            return img_feature[:, 0, :]
        txt = self.txt_processors["eval"](text)
        # TODO: transform composed_features
        text_features = self.model.extract_features({"image": img, "text_input": txt}, mode="text").text_embeds_proj # (N, 12, 768)
        # [cls] token, uni-modal features has been normalized
        composed_features = img_feature[:, 0, :] + text_features[:, 0, :]
        return composed_features
        # img = processor(images=image, return_tensors="pt", padding=True)
        # if text is not None:
        #     text = processor(text=text, return_tensors="pt", padding=True)
        #     return self.model.get_qformer_features(**img, **text)
        # return self.model.get_image_features(**img)

# load model test
if __name__ == "__main__":
    from pathlib import Path
    import requests
    from PIL import Image
    # ==========================
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    # prompt = "Question: how many cats are there? Answer:"
    # # run this script should modify the lavis/models/blip2.py PREFIX_DIR = "./"
    # model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_feature_extractor",
    #                                                                   model_type="pretrain", is_eval=True,
    #                                                                   device=device)
    # print(model)
    # image = vis_processors["eval"](image).unsqueeze(0).to(device)
    # text_input = txt_processors["eval"](prompt)
    # sample = {"image": image, "text_input": [text_input]}
    # features_multimodal = model.extract_features(sample)
    # print(features_multimodal.multimodal_embeds.shape)
    # features_image = model.extract_features(sample, mode="image")
    # features_text = model.extract_features(sample, mode="text")
    # print(features_image.image_embeds.shape)
    # # torch.Size([1, 32, 768])
    # print("text features shape:", features_text.text_embeds.shape)
    # ==========================
    # model = Blip2Model.from_pretrained("./blip2-flan-t5-xl")
    # processor = Blip2Processor.from_pretrained("./blip2-flan-t5-xl")
    # print(model)
    #
    #

    # print(processor(text=["a photo of a cat"], images=image, return_tensors="pt", padding=True))
    # print(processor(images=image, return_tensors="pt", padding=True))

    #
    # inputs = processor(images=image, text=prompt, return_tensors="pt")
    #
    # # decoder_input_ids can't be None. ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds
    # outputs = model(**inputs, decoder_input_ids=inputs.input_ids)
    # print(outputs)
    # run this script should modify the lavis/models/blip2.py PREFIX_DIR = "./"
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # model_t5, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_t5",
    #                                                                      model_type="pretrain_flant5xl", is_eval=True,
    #                                                                      device=device)
    model_t5, vis_processors, txt_processors = load_model_and_preprocess(name="blip2_t5_instruct",
                                                                         model_type="flant5xl", is_eval=True,
                                                                         device=device)
    image = Image.open("/data/chenjy/code/TGIR/Diffusion4TGIR/data/fashionIQ/image_data/dress/B00BPYP69K.jpg")
    model_t5 = model_t5.to(device)
    print(model_t5)
    lora_config = LoraConfig(
        target_modules=["q", "k"],
        init_lora_weights=False # # to initiate with random weights
    )
    model = get_peft_model(model_t5, lora_config)
    model.print_trainable_parameters()
    # show lora weight
    for name, param in model.named_parameters():
        # 名字转小写
        if "lora" in str(name).lower():
            print(name, param)
            break

    # save model
    model.save_pretrained("blip2_flant5xl_peft")
    # load model
    config = LoraConfig.from_pretrained("blip2_flant5xl_peft")
    model = PeftModel.from_pretrained(model_t5, "blip2_flant5xl_peft")
    for name, param in model.named_parameters():
        if "lora" in str(name).lower():
            print(name, param)
            break

    prompt = "I have an image. Given an instruction with 'Instruction:' to edit the image, carefully generate a finally composed description of new edited image. Instruction: 'shorter and tighter with more blue and white'. Write a detailed description of the edited image:"
    img = vis_processors["eval"](image).unsqueeze(0).to(device)

    print(model_t5.generate({"image": img, "prompt": prompt}, use_nucleus_sampling=True, num_captions=3))