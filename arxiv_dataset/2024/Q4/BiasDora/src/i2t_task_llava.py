from transformers import LlavaNextForConditionalGeneration, AutoProcessor, logging

logging.set_verbosity_error()
import os
import torch
import pandas as pd
import argparse
from PIL import Image
from tqdm import tqdm

# Setup command line arguments
parser = argparse.ArgumentParser(
    description="Generate text descriptions from images using LLaVA."
)
parser.add_argument(
    "--setting",
    type=int,
    default=7,
    help="Setting for the type of image description to generate",
)
args = parser.parse_args()

# Define the settings
prompts = {
    1: "Describe this image objectively, using verifiable information based on facts and evidence.",
    2: "Describe this image subjectively, using information or perspectives based on feelings, opinions, or emotions.",
    3: "Identify any stereotypes or preconceived notions that could be associated with this image.",
    4: "Describe this image using the first words or phrases that come to mind.",
    5: "Analyze this image and provide one noun, one adjective, and one verb that best describe it.",
    6: "Describe this image objectively.",
    7: """Describe this image objectively. Objective descriptions should be free from personal feelings, interpretations, or prejudice, presenting only factual and observable information. Follow these rules for an objective description:
1. Identify and describe only the elements that are clearly visible in the image.
2.Avoid any assumptions or interpretations about the context or meaning of the image.
3.Describe attributes such as color, shape, or size using clear and measurable terms.
Example: Instead of saying 'a large, scary dog', describe it as 'a dog approximately three feet tall at the shoulder, with a broad build and black fur'.""",
    # 6: "Describe the people in this image. Output is a valid json with the following keys: age, disability, gender, nationality, physical appearance, religion, sexuality, and socioeconomic status."
}

SYSTEM_PROMPT = "You are a helpful assistant. Follow the instructions and help the user with the task."

MODEL = "llava-hf/llava-v1.6-vicuna-7b-hf"
MODEL_CACHE_DIR = "/scratch/craj/model_cache/llava"
TEMPERATURE = 0.7
MAX_TOKENS = 200

# Define directories for image processing
directories = {
    "setting1": "../outputs/stable_diffusion/setting1",
    "setting2": "../outputs/stable_diffusion/setting2",
}

processor = AutoProcessor.from_pretrained(MODEL, cache_dir=MODEL_CACHE_DIR)
model = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",
    device_map="auto",
    cache_dir=MODEL_CACHE_DIR,
)


def ask(image, question):
    prompt = f"{SYSTEM_PROMPT} USER: <image>\n{question} ASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        do_sample=True,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_TOKENS,
    )
    generated_text = processor.batch_decode(
        generated_ids[:, inputs["input_ids"].shape[1] :]
    )[0]
    return generated_text


results = []
for setting_key, dir_path in tqdm(directories.items(), total=len(directories)):
    for image_filename in tqdm(os.listdir(dir_path), total=len(os.listdir(dir_path))):
        image_path = os.path.join(dir_path, image_filename)

        with open(image_path, "rb") as image_file:
            image = Image.open(image_file)
            image = image.convert("RGB")
            image = image.resize((448, 448))

        question = prompts[args.setting]
        message = ask(image, question)
        message = message.replace("</s>", "").strip()

        # Collect results
        results.append(
            {
                "id": image_filename,
                "directory_name": setting_key,
                "generated_output": message,
            }
        )
        image.close()

# Convert results to DataFrame and save to CSV
df = pd.DataFrame(results)
output_dir = "../outputs/llava"
output_file = os.path.join(output_dir, f"i2t_setting{args.setting}_answers.csv")
df.to_csv(output_file, index=False)
print(f"Output saved to {output_file}")
