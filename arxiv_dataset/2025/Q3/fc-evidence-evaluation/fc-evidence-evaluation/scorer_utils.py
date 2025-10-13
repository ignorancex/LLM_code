import json

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

import properties


def _confidence_confusion_matrix(actual_values, predicted_values, confidence_values):
    tp_ref = 0
    tp_supp = 0
    tp_nei = 0

    tn_ref_supp = 0
    tn_ref_nei = 0

    tn_supp_ref = 0
    tn_supp_nei = 0

    tn_nei_ref = 0
    tn_nei_supp = 0

    for actual_value, predicted_value, confidence_list in zip(actual_values, predicted_values, confidence_values):
        # let's first see if it's a true (t) or false prediction (f)
        confidence = confidence_list[predicted_value]
        if predicted_value == actual_value:  # t?
            if predicted_value == 0:
                tp_ref += 1 * confidence
            elif predicted_value == 1:
                tp_supp += 1 * confidence
            elif predicted_value == 2:
                tp_nei += 1 * confidence

        else:  # f?
            if actual_value == 0:  # refute
                if predicted_value == 1:
                    tn_ref_supp += 1 * confidence
                else:
                    tn_ref_nei += 1 * confidence
            elif actual_value == 1:  # support
                if predicted_value == 0:
                    tn_supp_ref += 1 * confidence
                else:
                    tn_supp_nei += 1 * confidence
            else:  # nei
                if predicted_value == 0:
                    tn_nei_ref += 1 * confidence
                else:
                    tn_nei_supp += 1 * confidence

    our_confusion_matrix = [
        [tp_ref, tn_ref_supp, tn_ref_nei],
        [tn_supp_ref, tp_supp, tn_supp_nei],
        [tn_nei_ref, tn_nei_supp, tp_nei]
    ]
    our_confusion_matrix = np.array(our_confusion_matrix)
    tp_sum = tp_ref + tp_supp + tp_nei
    f1_micro = tp_sum / (tp_sum + tn_ref_supp + tn_ref_nei + tn_supp_ref + tn_supp_nei + tn_nei_ref + tn_nei_supp)

    return our_confusion_matrix, f1_micro


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    confidence = nn.functional.softmax((torch.from_numpy(logits)).float(), dim=-1)

    predictions = np.argmax(logits, axis=-1)
    MCM, f1_micro = _confidence_confusion_matrix(labels, predictions, confidence)

    # f1_micro = f1_score(y_true=labels, y_pred=predictions, average='micro')
    f1_macro = f1_score(y_true=labels, y_pred=predictions, average='macro')
    return {"f1_micro": f1_micro, "f1_macro": f1_macro}


def map_label(response: str, is_two_classes: bool) -> int:
    label_str = json.loads(response)["label"].lower()
    # label_str = response.split(".")[-1].lower() if response.split(".")[-1] != "" else response.lower()
    try:
        if is_two_classes:
            label_pred = properties.LABEL_DICT[properties.Label(label_str)]
            if label_pred == 1:
                # only two-class dataset = support and not support, i.e. refute
                return 2
            return label_pred
        else:
            return properties.LABEL_DICT[properties.Label(label_str)]
    except Exception:
        print(f"unknown label_str: {label_str}")
        return -1
