"""This module implements different metrics used to evaluate the predictions
for the downstream tasks."""
from typing import Dict

import numpy as np
import pandas as pd
from absl import flags

FLAGS = flags.FLAGS


def postprocess_label(label: str) -> str:
    label = label.removesuffix("</s>")
    label = label.removeprefix("<s>")
    label = label.strip()
    return label


def sentiment_metric(prediction_file: str, num_labels: int) -> Dict[str, float]:
    """Compute the classification accuracy for sentiment classification. We
    report the following metrics:

    1 - accuracy: is the classification accuracy obtained by averaging the prediction scores across
                  the paraphrases of the input.
    2 - original_accuracy: is the classification accuracy obtained by the prediction score
                  of the original input text.
    3 - all_accuracy: is the classification accuracy obtained by averaging the prediction scores
                  from the paraphrases and the original input text.
    """

    df = pd.read_csv(prediction_file, delimiter=",")

    gold_labels = [postprocess_label(label) for label in df["gold_class"].tolist()]

    # This relies on the assumption that there is a prediction score for every label. (i.e. n label scores per input)
    predictions = [postprocess_label(label) for label in df["potential_class"].tolist()]

    assert len(predictions) % num_labels == 0
    prediction_labels = np.array(predictions).reshape((len(predictions) // num_labels, num_labels))

    return_metrics: Dict[str, float] = {}
    metrics = {
        "prediction_score": "accuracy",
        "original_prediction_score": "original_accuracy",
        "all_prediction_score": "all_accuracy",
    }

    for metric_column, metric in metrics.items():
        if metric_column in df.columns:
            scores = df[metric_column].tolist()
            prediction_scores = np.array(scores).reshape((len(predictions) // num_labels, num_labels))
            max_predictions = np.argmax(prediction_scores, axis=1)
            max_labels = []
            for index in range(len(predictions) // num_labels):
                labels_row = prediction_labels[index]
                max_labels.append(labels_row[max_predictions[index]])

            corrects = 0.0
            total = 0.0
            for index in range(len(predictions) // num_labels):
                total += 1.0
                if gold_labels[index * num_labels] == max_labels[index]:
                    corrects += 1.0

            accuracy = corrects / total
            return_metrics[metric] = accuracy

    return return_metrics
