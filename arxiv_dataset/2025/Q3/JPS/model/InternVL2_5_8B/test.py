import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import sys
sys.path.append('/workshop/crm/checkpoint/InternVL2_5-8B')
from conversation import get_conv_template

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
path = '/workshop/crm/checkpoint/InternVL2_5-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# set the max number of tiles in `max_num`
# pixel_values = load_image('/workshop/crm/checkpoint/InternVL2_5-8B/examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
generation_config = dict(max_new_tokens=1024, do_sample=False)

# batch inference, single image per sample (单图批处理)
# pixel_values1 = load_image('/workshop/crm/checkpoint/InternVL2_5-8B/examples/image1.jpg', max_num=6).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('/workshop/crm/checkpoint/InternVL2_5-8B/examples/image2.jpg', max_num=6).to(torch.bfloat16).cuda()
# num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
pixel_values = load_image('/workshop/crm/checkpoint/InternVL2_5-8B/examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('/workshop/crm/checkpoint/InternVL2_5-8B/examples/image2.jpg', max_num=6).to(torch.bfloat16).cuda()
num_patches_list = [pixel_values.size(0)]

questions = ['<image>\nWhat is the content of this image?'] * len(num_patches_list)
responses = model.batch_chat(tokenizer, pixel_values,
                             num_patches_list=num_patches_list,
                             questions=questions,
                             generation_config=generation_config)
for question, response in zip(questions, responses):
    print(f'User: {question}\nAssistant: {response}')

IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'
img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

if pixel_values is not None:
    image_bs = pixel_values.shape[0]
    print(f'dynamic ViT batch size: {image_bs}')

queries = []
for idx, num_patches in enumerate(num_patches_list):
    question = questions[idx]
    if pixel_values is not None and '<image>' not in question:
        question = '<image>\n' + question
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    image_size = model.config.force_image_size or model.config.vision_config.image_size
    patch_size = model.config.vision_config.patch_size
    num_image_token = int((image_size // patch_size) ** 2 * (model.config.downsample_ratio ** 2))
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
    query = query.replace('<image>', image_tokens, 1)
    queries.append(query)

tokenizer.padding_side = 'left'
model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
input_ids = model_inputs['input_ids'].to(model.device)
attention_mask = model_inputs['attention_mask'].to(model.device)
# import pdb;pdb.set_trace()
eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
image_flags = torch.ones(image_bs)
generation_output = model.generate(
    pixel_values=pixel_values,
    input_ids=input_ids,
    attention_mask=attention_mask,
    # image_flags=image_flags,
    **generation_config
)
print(tokenizer.decode(generation_output[0],skip_special_tokens=True))
# import pdb;pdb.set_trace()