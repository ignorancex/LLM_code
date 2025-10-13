"""
This script contains adapters for fast adaptation of
VLMs to downstream tasks/domains.

In particular, these adapters work over the vision and text
embeddings. Also, this code contains a Wrapper for zero-shot
classification

Implemented adapters:
Zero-shot, Linear Probe (LP)
"""

import copy
import torch
import numpy as np

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from cxrvlms.pretraining.data.transforms import augmentations_pretraining

# Device for training/inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
The first section contains only-vision adapters (i.e. linear probing)
"""


class AdapterWrapper(object):
    def __init__(self, model, targets, tta=False, fta=False):
        # Set model and number of targets
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.num_targets = len(targets)
        # Augmentation for training and for test-time augmentation
        self.tta = tta
        self.fta = fta
        self.number_augmentations = 20
        self.targets = targets

    def extract_vision_features(self, data_loader, transforms=None):
        self.model.eval()

        epoch_iterator = tqdm(
            data_loader, desc="Extracting features (X / X Steps)", dynamic_ncols=True
        )

        X, Y = [], []
        for step, batch in enumerate(epoch_iterator):
            images = batch["image"].to(device).to(torch.float32)

            with torch.no_grad():

                # Image augmentation
                if transforms is not None:
                    images = transforms(images)

                # Forward vision encoder
                x = self.model.vision_model(images)

            X.extend(x.cpu().detach().numpy())
            Y.extend(batch["label"].numpy())

        X = np.array(X)
        Y = np.array(Y)
        return X, Y

    def fit(self, loaders, transforms=None):
        data_loader = loaders["train"]

        if self.fta:
            transforms = augmentations_pretraining

        # Extract features and labels from generator
        if self.fta and transforms is not None:
            X, Y = [], []
            for i in range(self.number_augmentations):
                Xa, Ya = self.extract_vision_features(data_loader, transforms=transforms)
                X.append(Xa), Y.append(Ya)
            X = np.concatenate(X, 0)
            Y = np.concatenate(Y, 0)
        else:
            X, Y = self.extract_vision_features(data_loader, transforms=transforms)

        # Perform logistic regression
        self.train(X, Y)

    def train(self, X, Y):
        """
        Placeholder: function to be developed in a concrete adapter.
        """
        return

    def predict(self, loader, transforms=None):
        """
        Placeholder: function to be developed in a concrete adapter.
        """
        return


class LinearProbe(AdapterWrapper):
    def __init__(self, model, targets, tta=False, fta=False, c=0.316):
        super().__init__(model, targets, tta=tta, fta=fta)
        self.classifier = LogisticRegression(random_state=0, C=c, max_iter=1000, verbose=0,
                                             class_weight="balanced")

    def train(self, X, Y):

        # Train classifier
        self.classifier.fit(X, Y)

        # Set Linear Probe classifier into VLM model
        self.model.classifier = torch.nn.Linear(X.shape[-1], self.num_targets, bias=True)
        self.model.classifier.weight = torch.nn.Parameter(torch.tensor(self.classifier.coef_).to(torch.float32))
        self.model.classifier.bias = torch.nn.Parameter(torch.tensor(self.classifier.intercept_).to(torch.float32))
        self.model.classifier.to(device)

    def predict(self, loader, transforms=None):
        self.model.eval()

        # Set transforms on test-time augmentation
        if self.tta:
            transforms = augmentations_pretraining

        epoch_iterator = tqdm(
            loader, desc="Predicting (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )

        with torch.no_grad():
            refs, preds = [], []
            for step, batch in enumerate(epoch_iterator):
                images = batch["image"].to(device).to(torch.float32)
                Y = batch["label"].to(device).to(torch.long)

                # Forward
                if self.tta:
                    preds_tta = []
                    for i in range(self.number_augmentations):
                        x = self.model.vision_model(transforms(images))
                        score = self.model.classifier(x)
                        preds_tta.append(score.unsqueeze(-1))
                    score = torch.concat(preds_tta, -1).mean(-1)
                else:
                    x = self.model.vision_model(images)
                    score = self.model.classifier(x)
                # Activation for prediction
                if score.shape[-1] == 1:  # Binary case
                    score = torch.sigmoid(score)
                    score = torch.concat([1 - score, score], -1)
                else:  # Multi-class case
                    score = torch.softmax(score, -1)
                torch.cuda.empty_cache()

                refs.append(Y.cpu().detach().numpy())
                preds.append(score.cpu().detach().numpy())

        refs = np.concatenate(refs, 0)
        preds = np.concatenate(preds, 0)
        return refs, preds


"""
This section contains multimodal (vision-language) adapters.
"""


class LanguageAdapterWrapper(AdapterWrapper):
    def __init__(self, model, targets, prompt_type="names", tta=False, fta=False):
        super().__init__(model, targets, tta=tta, fta=fta)

        # Compute text prototypes
        if self.model.learning_criteria == "unimodal":
            self.text_embeds_dict, self.text_embeds, self.logit_scale = model.select_class_prototypes(targets)
        else:
            self.text_embeds_dict, self.text_embeds, self.logit_scale = model.compute_text_embeddings(
                targets, prompt_type=prompt_type)


class ZeroShot(LanguageAdapterWrapper):
    def __init__(self, model, targets, prompt_type="names", tta=False, fta=False):
        super().__init__(model, targets, prompt_type=prompt_type, tta=tta, fta=fta)

    def fit(self, loaders, transforms=None):
        """
        No training in zero-shot prediction
        """
        return

    def predict(self, loader, transforms=None):

        if self.tta:
            scores = []
            for i in range(self.number_augmentations):
                X, refs = self.extract_vision_features(loader, transforms=augmentations_pretraining)
                X = torch.tensor(X).to(device)
                with torch.no_grad():
                    score_i = torch.matmul(torch.tensor(X), self.text_embeds.t()) * self.logit_scale.exp()
                scores.append(score_i.unsqueeze(-1))
            score = torch.concat(scores, -1).mean(-1)
        else:
            X, refs = self.extract_vision_features(loader)
            X = torch.tensor(X).to(device)
            with torch.no_grad():
                score = torch.matmul(X, self.text_embeds.t().to(device)) * self.logit_scale.exp()

        # Softmax probs from scores
        if self.model.learning_criteria != "unimodal":
            preds = torch.softmax(score, dim=-1)
        else:
            preds = torch.sigmoid(score)
        preds = score
        preds = preds.detach().cpu().numpy()
        return refs, preds
