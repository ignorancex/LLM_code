import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load model directly
from transformers import AutoProcessor, LlavaNextForConditionalGeneration, AutoConfig, AutoModelForCausalLM
from PIL import Image
import pandas as pd
import os
from datasets import load_dataset
import argparse
from tqdm import tqdm

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--type", type=str, default="vivid")
arg_parser.add_argument("--concept", type=str, default="car")
args = arg_parser.parse_args()

TYPE = args.type
CONCEPT = args.concept
df = pd.read_csv("results/dalle_images.csv")

df = df[df["concept"] == CONCEPT]
df = df[df["type"] == TYPE]

dataset = df.to_dict(orient="records")
print(f"Number of {TYPE} images for {CONCEPT}: {len(dataset)}")

# Load model and processor
MODEL_NAME = "llava-v1.6-vicuna-7b-hf"
MODEL_CACHE_DIR = f"/projects/{MODEL_NAME}"

processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf", cache_dir=MODEL_CACHE_DIR).to("cuda")

def ask(image, question):
    prompt = f"USER: <image>\n{question} ASSISTANT:"

    # Process the image and text
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

    # Generate response
    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=128,
    )

    generated_text = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:])[0]

    # print("Predicted:", generated_text)
    return generated_text

results = []
for example in tqdm(dataset):
    path = example["image_path"]
    concept = example["concept"]
    country = example["country"]
    type = example["type"]

    id = path
    image = Image.open(path)
    if image.mode in ("RGBA", "P", "L", "LA"):
        image = image.convert("RGB")
    image = image.resize((448, 448))
    question = "Which geographical subregion is this image from? Strictly follow the United Nations geoscheme for subregions. Make an educated guess. Answer in one to three words."
    answer = ask(image, question)
    answer = answer.replace("</s>", "").strip()
    results.append({
        "id": id,
        "model": "llava-v1.6-vicuna-7b-hf",
        "split": f"{TYPE}_{CONCEPT}",
        "response": answer,
        "true_country": example["country"],
        "concept": example["concept"],
        "type": example["type"]
    })
    image.close()

df = pd.DataFrame(results)
print(df.head())
df.to_csv(f"results/dalle_eval/llava/{TYPE}/{CONCEPT}.csv", index=False)
print(f"Results saved to results/dalle_eval/llava/{TYPE}/{CONCEPT}.csv")
