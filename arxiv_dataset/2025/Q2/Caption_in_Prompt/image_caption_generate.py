from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image
from os import listdir
from os.path import join
from utils import save_list
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="imagenet", help="dataset")
parser.add_argument("--dir_path", type=str, default="/data/ImageNet/train", help="")
parser.add_argument("--save_path", type=str, default="imagenet_caption", help="path for saving image captions")
parser.add_argument("--caption_model", type=str, default="blip2", help="models used for image caption") # vit-gpt2, blip, blip2
parser.add_argument("--batch_size", type=int, default=65, help="batch size for image caption") # 128 for ViT-GPT2, 65 for blip2
args = parser.parse_args()

# Print Args
print("--------args----------")
for k in list(vars(args).keys()):
    print("%s: %s" % (k, vars(args)[k]))
print("--------args----------\n")

#vit-gpt2, blip2-opt-2.7b, blip-base, blip-large

dataset = args.dataset
dir_path = args.dir_path
save_path = args.save_path
caption_model = args.caption_model
batch_size = args.batch_size


if caption_model == 'vit-gpt2':
    pretrained_path = "nlpconnect/vit-gpt2-image-captioning"
elif caption_model == 'blip-base':
    pretrained_path = "Salesforce/blip-image-captioning-base"
elif caption_model == 'blip-large':
    pretrained_path = "Salesforce/blip-image-captioning-large"
elif caption_model == 'blip2':
    pretrained_path = "Salesforce/blip2-opt-2.7b"
else:
    raise ValueError("Wrong caption model name")

if caption_model == 'vit-gpt2':
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
elif caption_model == 'blip2':
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dir_folder_name = listdir(dir_path)

max_length = 16
num_beams = 1
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def predict_step_vit_gpt(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


def predict_step_blip(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)
    
    inputs = processor(images=images, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, **gen_kwargs)
    preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

if caption_model == 'vit-gpt2':
    predict_step = predict_step_vit_gpt
else:
    predict_step = predict_step_blip


for idx, class_name in enumerate(dir_folder_name):
    print("Caption %s-th class."%{idx})
    class_folder_path = join(dir_path, class_name)
    image_name_list = listdir(class_folder_path)
    image_path_list = [join(class_folder_path, f) for f in image_name_list]
    
    # slipt the image_path_list for saving memory
    image_path_list = [image_path_list[i:i + batch_size] for i in range(0, len(image_path_list), batch_size)]
    
    caption_list = []
    for batch_image_path_list in image_path_list:
        batch_caption_list = predict_step(batch_image_path_list)
        caption_list.extend(batch_caption_list)
    
    # save caption list
    save_list(caption_list, join(save_path, class_name))