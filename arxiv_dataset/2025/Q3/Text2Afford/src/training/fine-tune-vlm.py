import pandas as pd
import pickle
from datasets import Image
import datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image as image_loader
import io
from transformers import AutoProcessor, Blip2ForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration
from peft import LoraConfig, get_peft_model
import torch
from tqdm import tqdm

import train


feature = Image()

data_frame = pd.read_csv("ECCV_affordance_data.tsv", sep="\t")

# {"Object": [img1, img2,...,img5]}
with open('ECCV_object_images_map.pkl', 'rb') as f:
    ECCV_object_images_map = pickle.load(f)


labels = data_frame.columns.to_list()[2:]
data_dict = []
# ## Generate a new condition question

incontext_examples = "Question: Can human Play the game? Answer: Yes\n\nQuestion: Can human TypeOn the horse? Answer: No\n\nQuestion: Can human Ride the car? Answer: Yes\n\n"

for idx, rows in data_frame.iterrows():
    # print(rows['Object'])
    template = "{}Question: Can {} be used for {} by Human? Answer: "

    for lab in labels:
        if rows[lab] == 0:
            answer = "No"
        else:
            answer = "Yes"

        obj = rows['Object'].strip().replace(" ", "_")
        for key in ECCV_object_images_map.keys():
            if obj in key:
                break

        img_list = ECCV_object_images_map[key]
        dict = {
            "Sentence": rows['Sentence'],
            "Question": template.format(incontext_examples, rows['Object'], lab),
            "Answer": answer,
            "Image_1": feature.encode_example(img_list[0]),
            "Image_2": feature.encode_example(img_list[1]),
            "Image_3": feature.encode_example(img_list[2]),
            "Image_4": feature.encode_example(img_list[3]),
            "Image_5": feature.encode_example(img_list[4]),
        } 
        data_dict.append(dict)

df = pd.DataFrame(data_dict)
data = datasets.Dataset.from_pandas(df)

# Set a seed for reproducibility
seed = 42

# Split the dataset into training and validation sets
train_valid = data.train_test_split(test_size=0.1, seed=seed)
train_data = train_valid["train"]
val_data = train_valid["test"]
print(data)
print(train_valid)
print(val_data)

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Convert PIL image to numpy array
        # image1 = np.array(Image.open(io.BytesIO(item["image_1"]['bytes'])))
        # image2 = np.array(Image.open(io.BytesIO(item["image_2"]['bytes'])))

        # resize image 
        # image1 = image_loader.open(io.BytesIO(item["Image_1"]['bytes']))
        resize_dim = (384, 384)
        image1 = np.array(image_loader.open(io.BytesIO(item["Image_1"]['bytes'])).resize(resize_dim))
        # image2 = np.array(image_loader.open(io.BytesIO(item["Image_2"]['bytes'])).resize(resize_dim))
        # image3 = np.array(image_loader.open(io.BytesIO(item["Image_3"]['bytes'])).resize(resize_dim))
        # image4 = np.array(image_loader.open(io.BytesIO(item["Image_4"]['bytes'])).resize(resize_dim))
        # image5 = np.array(image_loader.open(io.BytesIO(item["Image_5"]['bytes'])).resize(resize_dim))
        # # Concatenate two numpy array
        # print(image1.shape)
        # print(image2.shape)
        # print(image3.shape)
        # print(image4.shape)
        # print(image5.shape)
        # image = np.concatenate((image1, image2, image3, image4, image5), axis=1)
        # # Convert numpy array to PIL image
        # image = Image.fromarray(image)


        encoding = self.processor(images=image1, padding="max_length", return_tensors="pt")
        # remove batch dimension
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        encoding["sentence"] = item["Sentence"]
        encoding["question"] = item["Question"]
        encoding["answer"] = item["Answer"]
        return encoding

def collate_fn(batch):
    # pad the input_ids and attention_mask
    processed_batch = {}
    for key in batch[0].keys():
        if key == "pixel_values":
            processed_batch[key] = torch.stack([example[key] for example in batch])
        elif key == "answer":
            text_labels = processor.tokenizer(
                [example["answer"] for example in batch], padding="max_length", return_tensors="pt", max_length=4
            )
            processed_batch["labels"] = text_labels["input_ids"]
        else:
            text_inputs = processor.tokenizer(
                [example["question"] for example in batch], padding="max_length", truncation=True, return_tensors="pt", max_length=128
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
    return processed_batch


# processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
# model = Blip2ForConditionalGeneration.from_pretrained("ybelkada/blip2-opt-2.7b-fp16-sharded", device_map="auto", load_in_8bit=True)

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b", device_map="auto", load_in_8bit=True)
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")

device = "cuda" if torch.cuda.is_available() else "cpu"


# Let's define the LoraConfig
config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj"]
)

model = get_peft_model(model, config)
model.print_trainable_parameters()

train_dataset = ImageCaptioningDataset(train_data, processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8, collate_fn=collate_fn)

val_dataset = ImageCaptioningDataset(val_data, processor)
val_dataloader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)


# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    avg_train_loss = train.train_model(model, train_dataloader, optimizer, device)
    print(f"Average Training Loss: {avg_train_loss}")

    # Validation
    avg_val_loss = train.evaluate_model(model, val_dataloader, device)
    print(f"Average Validation Loss: {avg_val_loss}")


# model.train()

# for epoch in tqdm(range(30)):
#   print("Epoch:", epoch)
#   for idx, batch in enumerate(train_dataloader):
#     input_ids = batch["input_ids"].to(device)
#     pixel_values = batch["pixel_values"].to(device, torch.float16)
#     labels = batch["labels"].to(device)

#     outputs = model(input_ids=input_ids,
#                     pixel_values=pixel_values,
#                     labels=labels)
    
#     loss = outputs.loss

#     # print(input_ids)
#     # print(pixel_values)
#     print("Loss:", loss.item())

#     loss.backward()

#     optimizer.step()
#     optimizer.zero_grad()


model.save_pretrained("fine-tuned-blip2") 
# model.push_to_hub("my_awesome_peft_model") also works


model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"

image = image_loader.open("0.png")

inputs = processor(image, text="Question: Can Human Feed a Helicopter? Answer Yes or No: ", return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=10)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)