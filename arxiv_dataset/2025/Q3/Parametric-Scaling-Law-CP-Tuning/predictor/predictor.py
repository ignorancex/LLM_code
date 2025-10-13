import warnings
import math
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from utils.utils import split_logits_labels,  build_score
from utils.metric import Metrics


class Predictor(nn.Module):
    def __init__(self, model, conformal, alpha, device, preprocessor = None):
        super(Predictor, self).__init__()
        self._model = model
        self.score_function = build_score(conformal)
        self.preprocessor = preprocessor
        self.alpha = alpha
        self.num_classes = 1000
        self._metric = Metrics()
        self.device = device

    def calibrate(self, calibloader):
        logits, labels = split_logits_labels(self._model, calibloader,self.device)
        if self.preprocessor:
            logits = self.preprocessor.scaling(logits, softmax=False)
        self.calculate_threshold(logits, labels)

    def calculate_threshold(self, logits, labels):
        alpha = self.alpha
        if alpha >= 1 or alpha <= 0:
            raise ValueError("Significance level 'alpha' must be in (0,1).")
        logits = logits.to(self.device)
        labels = labels.to(self.device)
        scores = self.score_function(logits, labels)
        self.q_hat = self._calculate_conformal_value(scores, alpha)

    def _calculate_conformal_value(self, scores, alpha):
        if len(scores) == 0:
            warnings.warn(
                "The number of scores is 0, which is a invalid scores. To avoid program crash, the threshold is "
                "set as torch.inf.")
            return torch.inf
        qunatile_value = math.ceil((scores.shape[0] + 1)*(1 - alpha)) / scores.shape[0]
        if qunatile_value > 1:
            warnings.warn(
                "The value of quantile exceeds 1. It should be a value in (0,1). To avoid program crash, the threshold "
                "is set as torch.inf.")
            return torch.inf

        return torch.quantile(scores, qunatile_value, interpolation="higher").to(self.device)

    def predict(self, x_batch):
        self._model.eval()
        if self._model is not None:
            tmp_logits = self._model(x_batch.to(self.device)).float()
        if self.preprocessor:
            tmp_logits = self.preprocessor.scaling(tmp_logits, softmax=False).detach()
        sets = self.predict_with_logits(tmp_logits)
        return sets

    def predict_with_logits(self, logits, q_hat=None):
        scores = self.score_function(logits).to(self.device)
        if q_hat is None:
            S = self._generate_prediction_set(scores, self.q_hat)
        else:
            S = self._generate_prediction_set(scores, q_hat)
        return S

    def evaluate(self, val_dataloader):
        prediction_sets = []
        probs_list = []
        labels_list = []
        with torch.no_grad():
            for examples in tqdm(val_dataloader):
                tmp_x, tmp_label = examples[0].to(self.device), examples[1].to(self.device)
                prediction_sets_batch = self.predict(tmp_x)
                prediction_sets.extend(prediction_sets_batch)
                tmp_probs = self._model(tmp_x)
                if self.preprocessor:
                    tmp_probs = self.preprocessor.scaling(tmp_probs, softmax=False).detach()
                probs_list.append(tmp_probs)
                labels_list.append(tmp_label)
        val_probs = torch.cat(probs_list)
        val_labels = torch.cat(labels_list)

        return self._metric('accuracy')(val_probs, val_labels, [1]), self._metric('coverage_rate')(prediction_sets, val_labels), self._metric('average_size')(prediction_sets, val_labels)

    def _generate_prediction_set(self, scores, q_hat):
        if len(scores.shape) == 1:
            return torch.argwhere(scores <= q_hat).reshape(-1).tolist()
        else:
            return [torch.argwhere(scores[i] <= q_hat).reshape(-1).tolist() for i in range(scores.shape[0])]
