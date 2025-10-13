from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import torchvision.transforms as transforms
from PIL import Image

# # default: Load the model on the available device(s)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "/workshop/crm/checkpoint/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/workshop/crm/checkpoint/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda:0",
)

# default processer
processor = AutoProcessor.from_pretrained("/workshop/crm/checkpoint/Qwen2-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

images_tensor = torch.zeros((3, 448, 448))
to_pil = transforms.ToPILImage()
image_obj = to_pil(images_tensor)
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": '/workshop/crm/data/MM-SafetyBench/img/01-Illegal_Activitiy/SD_TYPO/3.jpg',
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]

# # Preparation for inference
# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )
# image_inputs, video_inputs = process_vision_info(messages)
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )

# inputs = inputs.to(model.device)
# temporal_patch_size = processor.image_processor.temporal_patch_size
# patch_size = processor.image_processor.patch_size
# merge_size = processor.image_processor.merge_size
# grid_t, grid_h, grid_w = inputs['image_grid_thw'][0]
# patches = inputs['pixel_values'].reshape(
#     grid_t, grid_h // merge_size, grid_w // merge_size, 
#     merge_size, merge_size, 3, 
#     temporal_patch_size, patch_size, patch_size
# )
 
# patches = patches.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
# patches = patches.reshape(grid_t * temporal_patch_size, 3, grid_h * patch_size, grid_w * patch_size)
# image_mean=(
#     0.48145466,
#     0.4578275,
#     0.40821073
# )
# image_std=(
#     0.26862954,
#     0.26130258,
#     0.27577711
# )
# image_mean = torch.tensor(image_mean)
# image_std = torch.tensor(image_std)
# import pdb;pdb.set_trace()
# result_image = patches[0].cpu() * image_std[:,None,None] + image_mean[:,None,None]
# result_image = result_image * 255
# result_image = result_image.byte()
# img = Image.fromarray(result_image.permute(1, 2, 0).cpu().numpy(), 'RGB')
# img.save('./result.png')

# import pdb;pdb.set_trace()
# import time
# start_time=time.time()
# # Inference: Generation of the output
# generated_ids = model.generate(**inputs, max_new_tokens=128)
# generated_ids_trimmed = [
#     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
# print(output_text)
# end_time=time.time()
# print(end_time-start_time)

messages1 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": '/workshop/crm/data/MM-SafetyBench/img/01-Illegal_Activitiy/SD_TYPO/3.jpg',
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

messages2 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_obj,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

messages3 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_obj,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    },
    {
        "role":"assistant",
        "content":[
            {"type":"text","text":"This is a picture of a cat."}
        ]
    }
]

messages4 = [
    {
        "role": "user",
        "content":[]
    },
]
# messages4[0]['content']=[
#     {
#         "type": "image",
#         "image": image_obj,
#     },
#     {"type": "text", "text": 'test'},
# ]
messages5 = [
   {
        "role": "user",
        "content":[]
    },
]
messages = [ messages2, messages3, messages4,messages5]

# Preparation for batch inference
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    for msg in messages
]

# texts = [text.rstrip("<|im_end|>\n") for text in texts]

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")
import pdb;pdb.set_trace()
generated_id = model(**inputs)

# Batch Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
import pdb;pdb.set_trace()
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_texts = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_texts)