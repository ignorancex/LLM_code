import numpy as np
import torch

# for loading dataset
from data.celeba import CelebA, get_transform_celeba

# for various functions
from function.extract_caption import extract_caption
from function.extract_keyword import extract_keyword
from function.calculate_similarity import calc_similarity
from function.print_similarity import print_similarity_2

from tqdm import tqdm
import os
import pandas as pd


import argparse

# ignore SourceChangeWarning when loading model
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings('ignore', category=SourceChangeWarning)

MODEL_NAME = 'facexformer'
DATASET = 'celeba'

parser = argparse.ArgumentParser()
parser.add_argument('--model_output_dir', default=f'../../base_models/celeba_face_attribute_classification/{MODEL_NAME}/model output - face attribute classification - {DATASET}')
parser.add_argument('--split', default='val')
parser.add_argument('--extract_caption', default=True)
parser.add_argument('--save_result', default=True)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load dataset
preprocess = get_transform_celeba()
image_dir = './data/celebA/img_align_celeba/'
caption_dir = './data/celebA/caption/'
os.makedirs(caption_dir, exist_ok=True)
dataset = CelebA(data_dir='./data/celebA/', split=args.split, transform=preprocess)
# attributes = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald", "Bangs", "Big_Lips",
#               "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby",
#               "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male",
#               "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose",
#               "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
#               "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]
attributes = ['Bangs']

bias_dir = f'detected_biases'
os.makedirs(bias_dir, exist_ok=True)

# EXTRACT CAPTIONS AND "CORRECTIFY" DATASET
df_filename = f'{DATASET}-{args.split}-captions-{MODEL_NAME}-results.feather'

if os.path.exists(df_filename):
    df = pd.read_feather(df_filename)
    print('Captions already extracted. Skipping..')
else:
    OUTPUT_DIR = args.model_output_dir
    model_output = np.concatenate(
        [np.load(os.path.join(OUTPUT_DIR, f)) for f in sorted(os.listdir(OUTPUT_DIR)) if f.startswith('logits')],
        axis=0)[dataset.split_idx]
    preds = (model_output >= 0).astype(np.float32)
    records = []
    print('Start extracting captions..')
    for i, (image, target, img_filename) in enumerate(tqdm(dataset)):
        target = (target.cpu().numpy() + 1) / 2
        caption_path = caption_dir + img_filename + '.txt'
        if os.path.exists(caption_path):
            with open(caption_path, 'r') as f:
                caption = f.readline()
        else:
            image_path = os.path.join(image_dir, img_filename)
            caption = extract_caption(image_path)
            with open(caption_path, 'w') as f:
                f.write(caption)
        record = {'img_filename': img_filename, 'caption': caption}
        for j, attribute in enumerate(attributes):
            record[f'{attribute}-pred'] = preds[i, j].astype(np.float32)
            record[f'{attribute}-gt'] = target[j].astype(np.float32)
        records.append(record)
    print(f'Captions of {len(dataset)} images extracted')
    df = pd.DataFrame.from_records(records)
    df.to_feather(df_filename)

for attribute in attributes:
    diff_path_1 = os.path.join(bias_dir, f'{DATASET}-{dataset.split}_{MODEL_NAME}_{attribute}.csv')
    # if os.path.exists(diff_path_1):
    #     continue

    pred_col = f'{attribute}-pred'
    gt_col = f'{attribute}-gt'

    # extract keyword
    actual_positive = df[gt_col].values == 1.0
    actual_negative = df[gt_col].values == 0.0
    predicted_positive = df[pred_col].values == 1.0
    predicted_negative = df[pred_col].values == 0.0

    TP = actual_positive & predicted_positive  # df_correct_class_1
    FP = actual_negative & predicted_positive  # df_wrong_class_0
    FN = actual_positive & predicted_negative  # df_wrong_class_1
    TN = actual_negative & predicted_negative  # df_correct_class_0

    FN_captions = ' '.join(df[FN]['caption'].tolist())  # caption_wrong_class_1

    FN_keywords = extract_keyword(FN_captions)  # keywords_class_1

    # calculate similarity
    print('Start calculating scores..')
    FN_similarity = calc_similarity(image_dir, df[FN]['img_filename'], FN_keywords)  # similarity_wrong_class_1
    TP_similarity = calc_similarity(image_dir, df[TP]['img_filename'], FN_keywords)  # similarity_correct_class_1

    dist_P = FN_similarity - TP_similarity  # dist_class_1

    print(f'Result for class: {attribute}')
    diff_1 = print_similarity_2(FN_keywords, dist_P, df[actual_positive], pred_col, gt_col)

    # if args.save_result:
    #     diff_1.to_csv(diff_path_1)
