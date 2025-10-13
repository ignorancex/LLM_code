import json

import numpy as np
import torch
from torchvision.datasets import ImageNet

# for loading dataset
from imagenet_x import FACTORS
from imagenet_x.evaluate import ImageNetX, get_vanilla_transform

# for various functions
from function.extract_caption import extract_caption
from function.extract_keyword import extract_keyword
from function.calculate_similarity import calc_similarity_2
from function.print_similarity import print_similarity_2

from tqdm import tqdm
import os
import pandas as pd


import argparse

# ignore SourceChangeWarning when loading model
import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings('ignore', category=SourceChangeWarning)

MODEL_NAME = 'ResNet101_V2'
DATASET = 'inx'


parser = argparse.ArgumentParser()
parser.add_argument('--model_output_file', default=f'../../base_models/imagenet-x_classification/torchvision_models/results_{DATASET}_{MODEL_NAME}.npy')
parser.add_argument('--split', default='val')
parser.add_argument('--extract_caption', default=True)
parser.add_argument('--save_result', default=True)
args = parser.parse_args()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Fix deprecated np.bool -> bool to work with numpy >= 1.24 and change output
def fixed_getitem(self, index):
    img, target = ImageNet.__getitem__(self, index)
    img_path = self.samples[index][0]
    img_id = img_path.split("/")[-1]
    img_annotations = self.annotations_.loc[img_id]
    return img_path, img_id, target, img_annotations[FACTORS].values.astype(bool)


ImageNetX.__getitem__ = fixed_getitem

# load dataset
imagenet_val_path = './data/imagenet'
transforms = get_vanilla_transform()
dataset = ImageNetX(imagenet_val_path, transform=transforms, which_factor='multi', filter_prototypes=False)
caption_dir = './data/imagenet/caption/'
os.makedirs(caption_dir, exist_ok=True)
with open('./data/imagenet/imagenet_labels.json', 'r') as f:
    classes = json.load(f)

bias_dir = f'detected_biases'
os.makedirs(bias_dir, exist_ok=True)

# EXTRACT CAPTIONS AND "CORRECTIFY" DATASET
df_filename = f'{DATASET}-{args.split}-captions-{MODEL_NAME}-results.feather'

if os.path.exists(df_filename):
    df = pd.read_feather(df_filename)
    print('Captions already extracted. Skipping..')
else:
    MODEL_OUTPUT_FILE = args.model_output_file
    model_output = np.load(MODEL_OUTPUT_FILE)
    records = []
    print('Start extracting captions..')
    for i, (img_path, img_id, target, _) in enumerate(tqdm(dataset)):
        pred = int(model_output[i, 1])
        caption_path = caption_dir + img_id + '.txt'
        if os.path.exists(caption_path):
            with open(caption_path, 'r') as f:
                caption = f.readline()
        else:
            caption = extract_caption(img_path)
            with open(caption_path, 'w') as f:
                f.write(caption)
        record = {'img_path': img_path, 'caption': caption, 'target': target, 'pred': pred,
                  'target_class': classes[target], 'pred_class': classes[pred]}
        records.append(record)
    print(f'Captions of {len(dataset)} images extracted')
    df = pd.DataFrame.from_records(records)
    df.to_feather(df_filename)

for target_i, target_class in enumerate(classes):
    diff_path_1 = os.path.join(bias_dir, f'{DATASET}-{dataset.split}_{MODEL_NAME}_{target_class}.csv')
    if os.path.exists(diff_path_1):
        continue

    # extract keyword
    actual_positive = df['target'].values == target_i
    actual_negative = df['target'].values != target_i
    predicted_positive = df['pred'].values == target_i
    predicted_negative = df['pred'].values != target_i

    TP = actual_positive & predicted_positive  # df_correct_class_1
    FP = actual_negative & predicted_positive  # df_wrong_class_0
    FN = actual_positive & predicted_negative  # df_wrong_class_1
    TN = actual_negative & predicted_negative  # df_correct_class_0

    FN_captions = ' '.join(df[FN]['caption'].tolist())  # caption_wrong_class_1

    if len(FN_captions) == 0:
        continue

    FN_keywords = extract_keyword(FN_captions)  # keywords_class_1

    # calculate similarity
    print('Start calculating scores..')
    FN_similarity = calc_similarity_2(df[FN]['img_path'], FN_keywords)  # similarity_wrong_class_1
    TP_similarity = calc_similarity_2(df[TP]['img_path'], FN_keywords)  # similarity_correct_class_1

    dist_P = FN_similarity - TP_similarity  # dist_class_1

    print(f'Result for class: {target_class}')
    diff_1 = print_similarity_2(FN_keywords, dist_P, df[actual_positive], 'pred', 'target')

    if args.save_result:
        diff_1.to_csv(diff_path_1)
