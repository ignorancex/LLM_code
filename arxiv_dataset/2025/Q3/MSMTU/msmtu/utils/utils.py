import numpy as np
import json
import ujson as json
import os
import pandas as pd


def _convert(o: np.int):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def _read_json(json_path: str):
    with open(json_path, 'r') as file:
        data = json.load(file)

    return data


def _write_json(json_path: str, data: dict, indent: int = 4):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

import re


def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)

    return l


def get_delta_impute(num_dim: int = 6, seq_len: int = 12):

    deltas = [np.ones(num_dim) for month in range(seq_len)]

    return np.array(deltas)

def parse_rec(values, masks, label):
    deltas = get_delta_impute()

    rec = {
        'label': int(label),
        'values': np.asarray(values).tolist(),
        'masks': masks.astype('int32').tolist(),
        'evals': None,
        'eval_masks': None,
        'forwards': None,
        'deltas': deltas.tolist()
    }

    return rec


def reformat_json_path(impute_dir, json_path):
    # './json/LULC/09_ForestsOpDeNe/000665.json'
    path_list = json_path.split(sep='/')
    class_dir = os.path.join(impute_dir, path_list[-2])
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    json_impute_path = os.path.join(class_dir, path_list[-1])

    return json_impute_path


def get_original_metadata(json_path):
    with open(json_path, 'r') as f:
        pixel_data = json.load(f)

    return pixel_data['metadata']


def save_sample(sample, label: str, json_path: str, impute_dir: str):
    original_metadata = get_original_metadata(json_path)
    masks = np.ones_like(sample)
    rec = parse_rec(sample, masks, label)
    json_impute_path = reformat_json_path(impute_dir, json_path)

    with open(json_impute_path, 'w') as f:
        imputed_data = {
            'metadata': original_metadata,
            'ts_data': rec
        }
        json.dump(imputed_data, f, indent=4)


def save_predictions(predictions, csv_path, idx_to_class):
    """ Create a csv file with the predictions.
    - 'predictions' is a list of dictionaries like:
        {
            'predictions': y_h,
            'labels': labels,
            'probs': probs,
            'square_ids': square_ids
        }
    - The index of the csv is 'square_ids'
    - 'classes' is a list of the classes in the same order as the predictions and probs
    """
    classes = list(idx_to_class.values())
    true_classes = [cls + '_true' for cls in classes]
    pred_classes = [cls + '_pred' for cls in classes]
    columns = ['square_id'] + true_classes + pred_classes + \
              ['cls_max_true',  'cls_max_pred', 'cls_max_true_prob', 'cls_max_pred_prob']
    df = pd.DataFrame(columns=columns)

    for pred in predictions:
        data_dict = {
            'square_id': pred['square_ids'].int(),
            'cls_max_true': [idx_to_class[label.item()] for label in pred['labels']],
            'cls_max_pred': [idx_to_class[label.item()] for label in pred['predictions'].argmax(dim=1)],
            'cls_max_true_prob': pred['probs'].max(dim=1)[0],
            'cls_max_pred_prob': pred['predictions'].max(dim=1)[0],
        }
        for idx, cls in enumerate(classes):
            data_dict.update({
                cls + '_true': pred['probs'][:, idx],
            })
            data_dict.update({
                cls + '_pred': pred['predictions'][:, idx],
            })
        new_df = pd.DataFrame(data_dict, columns=columns)

        df = pd.concat([df, new_df])

    df.set_index('square_id', inplace=True)
    df.sort_index(inplace=True)

    df.to_csv(csv_path, index=True)
