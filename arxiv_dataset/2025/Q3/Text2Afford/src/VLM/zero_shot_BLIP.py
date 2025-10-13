from transformers import AutoProcessor, BlipForConditionalGeneration, Blip2ForConditionalGeneration
import requests
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import datasets
import eval
import re, os
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"



# model = InstructBlipForConditionalGeneration.from_pretrained(checkpoint, load_in_8bit=True)
# processor = InstructBlipProcessor.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

model = Blip2ForConditionalGeneration.from_pretrained(

    "Salesforce/blip2-opt-2.7b", load_in_8bit=True

)

image_type = 'retrieval'
image_type = 'generation'

filename = "../data/ECCV_affordance_data.tsv"
filename = "../data/toloka_annotated_data.tsv"
filename = "../data/final_annotated_data.tsv"
# filename = "../data/Daivik_annotated.tsv"
filename = "../data/rare_object_annotated_data.tsv"
filename = "../data/Rare_Objects_Sayantan_Merged.tsv"
filename = "../data/rare_xnli.tsv"
filename = "../../data/final_normal_dataset.tsv"

data = pd.read_csv(filename, sep='\t')


label_names = list(data.columns[2:])
oov_class_map = {"play": "play using", "lookthrough": "look through", "siton": "sit on", "pourfrom": "pour from", "writewith": "write with", "typeon": "type using"}
labels_lower = [x.lower() for x in label_names]
classes = []
for lab in labels_lower:
    if lab not in oov_class_map.keys():
        classes.append(lab)
    else:
        classes.append(oov_class_map[lab])

incontext_examples = """Example 1: can human ride bicycle? 
Answer: Yes

Example 2: can human write with chalk? 
Answer: Yes

Example 3: can human grasp dog? 
Answer: No

Example 4: can human ride violin? 
Answer: No

Example 5: can human lift donkey? 
Answer: No"""

def predict_affordance_proba(model, processor, sentence, object_name, img_path):
    """Re-ranks the candidates answers for each question.

    Returns:
        ranked_ans: list of re-ranked candidate docids
        sorted_scores: list of relevancy scores of the answers
    -------------------
    Arguments:
        model - PyTorch model
        q_text - str - query
        cands -List of retrieved candidate docids
        max_seq_len - int
    """
    # Empty dict for the probability scores of affordances
    affordance_proba = {}
    img = Image.open(img_path)

    for affordance in classes:
        prompt = f"{incontext_examples}\n\nConsider the sentence - {sentence}. Now from this information, can human {affordance} the {object_name}? Accompanying this query is an image of the {object_name}. Note that the image may contain noise or variations in appearance. Given the textual description and the image, answer Yes or No whether the human can {affordance} the {object_name}.\nAnswer: "

        

        inputs = processor(images=img, text=prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(**inputs, max_length=256)

        # outputs = model.generate(
        #         **inputs,
        #         do_sample=False,
        #         num_beams=5,
        #         max_length=256,
        #         min_length=1,
        #         # top_p=0.9,
        #         repetition_penalty=1.5,
        #         length_penalty=1.0,
        #         temperature=1,
        # )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        print(generated_text)
        if 'yes' in generated_text.lower():
            affordance_proba[affordance] = 1
        else:
            affordance_proba[affordance] = 0

    
    return affordance_proba


## Initialize to calculate Accuracy
gt_affordance = {}
correct = 0
wrong = 0
avg_acc = []

## Initialize to calculate Mean Average Precision
ground_truth_classes = []
predicted_classes = []

all_predictions = []

for ids, rows in tqdm(data.iterrows()):
# predicted_affordances = predict_affordance_proba(model, sentence, object)
#     if ids >=2:
#         break
    if ids < 1868:
        continue
    positive_classes = []
    negative_classes = []
    sentence = rows[0]
    object_name = rows[1]

    image_path = f'../../data/test_images/{image_type}/{ids}'

    image_files = os.listdir(image_path)
    image_path_list = [os.path.join(image_path, image_file) for image_file in image_files]
    img_path = image_path_list[0]

    predicted_affordances = predict_affordance_proba(model, processor, sentence, object_name, img_path)
    sorted_predicted_affordances = dict(sorted(predicted_affordances.items(), key=lambda item: item[1], reverse=True))

    all_predictions.append(predicted_affordances)

    for itr in range(2, data.shape[1]):
        class_name = list(data.columns)[itr].lower()
        ## replacing oov classnames
        class_name = class_name if class_name not in oov_class_map.keys() else oov_class_map[class_name]

        gt_affordance[class_name] = rows[itr]

        if rows[itr] > 0:
            positive_classes.append(class_name)
        else:
            negative_classes.append(class_name)

    # print(object_name)
    # print(predicted_affordances)
    # print(gt_affordance)

    prev_cor = correct
    prev_wrong = wrong
    
    if len(positive_classes) > 0:
        for pos in positive_classes:
            for neg in negative_classes:
                if predicted_affordances[pos] > predicted_affordances[neg]:
                    correct += 1
                else:
                    wrong += 1

        avg_acc.append((correct-prev_cor)/(correct+wrong - prev_cor-prev_wrong))

    ground_truth_classes.append(positive_classes)
    predicted_classes.append(list(sorted_predicted_affordances.keys()))


accuracy = correct / (correct+wrong)

print("Accuracy: %s"%accuracy )
print("MAP: %s" %eval.mapk(ground_truth_classes, predicted_classes, 15))
print("Accuracy: %s"%(sum(avg_acc)/len(avg_acc)))

df = pd.DataFrame(all_predictions)

print(df.head())

df.to_csv("predictions_instructBlip.tsv", sep="\t")




############################################################## TEST ###############################################################

# model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b", load_in_8bit=True)
# processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")

# device = "cuda" if torch.cuda.is_available() else "cpu"
# # model.to(device)
# print(model.device)

# url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# image = Image.open("../training/0.png")
# # prompt = "What is unusual about this image?"
# "The image is given as a support for your understanding. Don't use any description of the image to answer the question."
# prompt = "I have a list of actions - 'Lift', 'Ride', 'Feed', 'TypeOn', 'Watch', 'Fix'. Can you tell me which of the actions can be performed on an Engine? PLEASE NOTE THAT THE PROVIDED IMAGE CAN BE NOISY AND UNRELATED TO THE QUESTION" 
# inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

# outputs = model.generate(
#         **inputs,
#         do_sample=False,
#         num_beams=5,
#         max_length=256,
#         min_length=1,
#         top_p=0.9,
#         repetition_penalty=1.5,
#         length_penalty=1.0,
#         temperature=1,
# )
# generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
# print(generated_text)