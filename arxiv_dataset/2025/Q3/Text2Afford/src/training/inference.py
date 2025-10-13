from transformers import Blip2ForConditionalGeneration, AutoProcessor
from peft import PeftModel, PeftConfig
import torch
from PIL import Image
import numpy as np
import pickle
import pandas as pd
import os

peft_model_id = "fine-tuned-blip2"
config = PeftConfig.from_pretrained(peft_model_id)

model = Blip2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, device_map="auto")
model = PeftModel.from_pretrained(model, peft_model_id)

print(config.base_model_name_or_path)
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")


model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"


# image_type = 'retrieval'
# # image_type = 'generation'

# filename = "../data/ECCV_affordance_data.tsv"
# filename = "../data/toloka_annotated_data.tsv"
# filename = "../data/final_annotated_data.tsv"
# # filename = "../data/Daivik_annotated.tsv"

# data = pd.read_csv(filename, sep='\t')

# # print(data.Sentence.loc[0])


# label_names = list(data.columns[2:])
# oov_class_map = {"lookthrough": "looking", "siton": "sitting", "pourfrom": "pouring", "writewith": "writing", "typeon": "typing"}

# labels_lower = [x.lower() for x in label_names]
# classes = []
# for lab in labels_lower:
#     if lab not in oov_class_map.keys():
#         classes.append(lab)
#     else:
#         classes.append(oov_class_map[lab])



# def predict_affordance_proba(model, sentence, object_name, image_path_list):

#     # Empty dict for the probability scores of affordances
#     affordance_proba = {}

#     for affordance in classes:
#         prompt = f"Context: {sentence}\n\nQuestion: Can human {affordance} the {object_name}?\n\nAnswer: "
#         # prompt = f"Context: {sentence}. Question: Can the action '{affordance}' be performed on {object_name} by human?\nOPTIONS:\n-YES \n-NO"

#         image = Image.open(image_path_list[0])

#         inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

#         generated_ids = model.generate(**inputs, max_new_tokens=4)
#         generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

#         if 'yes' in generated_text.lower():
#             affordance_proba[affordance] = 1
#         else:
#             affordance_proba[affordance] = 0

    
#     return affordance_proba



# gt_affordance = {}
# correct = 0
# wrong = 0

# ## Initialize to calculate Mean Average Precision
# ground_truth_classes = []
# predicted_classes = []

# for ids, rows in data.iterrows():
# # predicted_affordances = predict_affordance_proba(model, sentence, object)
#     if ids >= 200:
#         break
#     positive_classes = []
#     negative_classes = []
#     sentence = rows[0]
#     object = rows[1]

#     image_path = f'../data/test_images/generation/{image_type}/{ids}'
#     image_path = f'../data/test_images/{image_type}/{ids}'

#     image_files = os.listdir(image_path)
#     image_path_list = [os.path.join(image_path, image_file) for image_file in image_files]

#     predicted_affordances = predict_affordance_proba(model, sentence, object, image_path_list)
#     sorted_predicted_affordances = dict(sorted(predicted_affordances.items(), key=lambda item: item[1], reverse=True))

#     for itr in range(2, data.shape[1]):
#         class_name = list(data.columns)[itr].lower()
#         ## replacing oov classnames
#         class_name = class_name if class_name not in oov_class_map.keys() else oov_class_map[class_name]

#         gt_affordance[class_name] = rows[itr]

#         if rows[itr] > 0:
#             positive_classes.append(class_name)
#         else:
#             negative_classes.append(class_name)

#     prev_cor = correct
#     prev_wrong = wrong
    
#     if len(positive_classes) > 0:
#         for pos in positive_classes:
#             for neg in negative_classes:
#                 if predicted_affordances[pos] > predicted_affordances[neg]:
#                     correct += 1
#                 else:
#                     wrong += 1

#         print((correct-prev_cor)/(correct+wrong - prev_cor-prev_wrong))

#         ground_truth_classes.append(positive_classes)
#         predicted_classes.append(list(sorted_predicted_affordances.keys()))

#     # break

# accuracy = correct / (correct+wrong)

# print("Accuracy: %s"%accuracy )




image = Image.open("0.png")

# prepare image for the model
# inputs = processor(images=image, return_tensors="pt").to(device)
# pixel_values = inputs.pixel_values

# text_inputs = processor.tokenizer("Can Helicopter be used for Ride by Human? Answer Yes or No: \n\n", padding=True, return_tensors="pt")
# input_ids = text_inputs["input_ids"].to(device)

# model.eval()

incontext_examples = "Context: He tried to stop the car. Question: Can human Ride the car? Answer: Yes\n\nContext: The man fell off his horse as it fell to the ground. Question: Can human TypeOn the horse? Answer: No\n\n"

inputs = processor(image, text=f"{incontext_examples}Context: I saw a flying Helicopter. Question: Can human watch the Helicopter? Answer: ", return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs, max_new_tokens=10)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

# print(pixel_values)
# print(input_ids)
# generate caption
# model.generate(pixel_values=pixel_values, tomax_length=50)
# generated_ids = model.generate(pixel_values=pixel_values, text="Question: Can Helicopter be used for Ride by Human? Answer Yes or No:", max_length=50)
# generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
# print(generated_ids)
# print(generated_caption)
# print(processor.tokenizer.decode([1437]))