import os
from data.pretrain_data import ImageNetDataset
from data.fashionIQ import FashionIQDataset
import torch
from set_up import set_up_gpu, fix_random_seed_as
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import json


def imagenet_caption(configs):
    random_seed = fix_random_seed_as(configs['random_seed'])
    set_up_gpu(configs['device_idx'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    dataset = ImageNetDataset()

    ROOT_DIR = "./models"
    blip2 = Blip2ForConditionalGeneration.from_pretrained(os.path.join(ROOT_DIR, "blip2-flan-t5-xl"),
                                                               local_files_only=True)
    processor = Blip2Processor.from_pretrained(os.path.join(ROOT_DIR, "blip2-flan-t5-xl"), local_files_only=True)
    blip2.to(device)

    for idx, batch in tqdm(enumerate(dataset)):
        image, name = batch['image'], batch['image_name']
        # check path exist and text is not empty
        if os.path.exists(f"/data/chenjy/code/TGIR/Diffusion4TGIR/data/ImageNet1K/captions/{str(name).replace('JPEG', 'txt')}"):
            if os.path.getsize(f"/data/chenjy/code/TGIR/Diffusion4TGIR/data/ImageNet1K/captions/{str(name).replace('JPEG', 'txt')}") != 0:
                continue
        img = processor(images=image, return_tensors="pt").to(device)
        generate_ids = blip2.generate(**img)
        text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
        # save text
        with open(f"/data/chenjy/code/TGIR/Diffusion4TGIR/data/ImageNet1K/captions/{str(name).replace('JPEG', 'txt')}", "w") as f:
            f.write(text)

def fashionIQ_caption(configs):
    random_seed = fix_random_seed_as(configs['random_seed'])
    set_up_gpu(configs['device_idx'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(os.environ['CUDA_VISIBLE_DEVICES'], device, random_seed)



    ROOT_DIR = "./models"
    blip2 = Blip2ForConditionalGeneration.from_pretrained(os.path.join(ROOT_DIR, "blip2-flan-t5-xl"),
                                                          local_files_only=True)
    processor = Blip2Processor.from_pretrained(os.path.join(ROOT_DIR, "blip2-flan-t5-xl"), local_files_only=True)
    blip2.to(device)

    # use json file to save text and id
    for clothing_type in FashionIQDataset.all_subset_codes():
        print("caption for", clothing_type)
        dataset = FashionIQDataset(clothing_type=clothing_type) # duplicate data
        caption = []
        for idx, batch in tqdm(enumerate(dataset), total=len(dataset)):
            reference_img, target_img, modifier, _, ref_id, targ_id = batch
            # print(reference_img, target_img, modifier, ref_id, targ_id)
            img = processor(images=reference_img, return_tensors="pt").to(device)
            generate_ids = blip2.generate(**img)
            text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
            caption.append({"candidate": ref_id, "captions": text})

        # save json
        with open(f"/data/chenjy/code/TGIR/Diffusion4TGIR/data/fashionIQcaptions/cap.{clothing_type}.train.json", "w") as f:
            json.dump(caption, f, indent=4)

    # remove duplicate data
    # test json file
    with open("/data/chenjy/code/TGIR/Diffusion4TGIR/data/fashionIQcaptions/cap.dress.train.json", "r") as f:
        data = json.load(f)
    print(data[:10])
    # find that duplicate data, 0 and 1 is the same, we need all odd index
    for closs in ["dress", "shirt", "toptee"]:
        with open(f"/data/chenjy/code/TGIR/Diffusion4TGIR/data/fashionIQcaptions/cap.{closs}.train.json", "r") as f:
            data = json.load(f)
        new_data = []
        for i in range(0, len(data), 2):
            new_data.append(data[i])
        print(len(new_data), len(data))
        with open(f"/data/chenjy/code/TGIR/Diffusion4TGIR/data/fashionIQcaptions/cap.{closs}.train.json", "w") as f:
            json.dump(new_data, f, indent=4)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_idx', type=str, default='2')
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()
    configs = args.__dict__
    # imagenet_caption(configs)
    fashionIQ_caption(configs)