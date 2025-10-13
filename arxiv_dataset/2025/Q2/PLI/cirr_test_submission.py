import json
import multiprocessing
import os
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple, Dict

from torch import nn
from torchvision.transforms import transforms

import clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.cirr import CIRRDataset, _DEFAULT_CIRR_DATASET_ROOT
from lavis.models import load_model_and_preprocess
from options import get_experiment_config
from set_up import setup_experiment
from models import create_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def extract_index_features(dataset: CIRRDataset, visual):
    """
    Extract CIRR index features
    :param dataset: CIRR dataset in 'classic' mode
    :param visual: visual model
    :return: a tensor of features and a list of images
    """
    classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=multiprocessing.cpu_count(),
                                    pin_memory=True, collate_fn=collate_fn)
    index_features, index_names = [], []
    for name, img in tqdm(classic_val_loader):
        img = img.to(device, non_blocking=True)
        with torch.no_grad():
            batch_features = visual(img) # (batch_size, feature_dim)
            index_features.append(batch_features)
            index_names.extend(name)
    index_features = torch.cat(index_features, dim=0).to(device, non_blocking=True) # (num_images, feature_dim)
    return index_features, index_names

def generate_cirr_test_submissions(models, file_name, image_transform, text_transform, **kwargs):
    """
   Generate and save CIRR test submission files to be submitted to evaluation server
    :param models: models to be used for inference, a dict of models
    :param file_name: name of the submission file
    :param image_transform: image transform to be used for inference
    :param text_transform: text transform to be used for inference
   """
    img_weight = kwargs.get('img_weight', 0.5)
    # evaluation
    for key in models.keys():
        models[key].eval()

    # Define the dataset and extract index features
    classic_test_dataset = CIRRDataset(root_path=_DEFAULT_CIRR_DATASET_ROOT, split='test1', img_transform=image_transform, text_transform=text_transform, mode='classic')
    index_features, index_names = extract_index_features(classic_test_dataset, models['image_encoder'])
    relative_test_dataset = CIRRDataset(root_path=_DEFAULT_CIRR_DATASET_ROOT, split='test1', img_transform=image_transform, text_transform=text_transform, mode='relative')

    # Generate test prediction dicts for CIRR
    pairid_to_predictions, pairid_to_group_predictions = generate_cirr_test_dicts(relative_test_dataset, models,
                                                                                  index_features, index_names, img_weight)

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_predictions)
    group_submission.update(pairid_to_group_predictions)

    # Define submission path
    submissions_folder_path = os.path.join(_DEFAULT_CIRR_DATASET_ROOT, "submission", 'CIRR')
    Path(submissions_folder_path).mkdir(exist_ok=True, parents=True)

    print(f"Saving CIRR test predictions in {submissions_folder_path}/recall_submission_{file_name}.json")
    with open(f"{submissions_folder_path}/recall_submission_{file_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(f"{submissions_folder_path}/recall_subset_submission_{file_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)


def generate_cirr_test_dicts(relative_test_dataset: CIRRDataset, models: dict, index_features: torch.tensor,
                             index_names: List[str], img_weight) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Compute test prediction dicts for CIRR dataset
    :param relative_test_dataset: CIRR test dataset in relative mode
    :param models: models to be used for inference, a dict of models
    :param index_features: test index features
    :param index_names: test index names
    :return: Top50 global and Top3 subset prediction for each query (reference_name, caption)
    """

    # Generate predictions
    predicted_features, reference_names, group_members, pairs_id = \
        generate_cirr_test_predictions(models, relative_test_dataset, index_names, index_features, img_weight)

    print(f"Compute CIRR prediction dicts")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T # (num_queries, num_images)
    sorted_indices = torch.argsort(distances, dim=-1).cpu() # (num_queries, num_images)
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names),
                                                                                             -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(len(sorted_index_names), -1)
    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_predictions = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                             zip(pairs_id, sorted_index_names)}
    pairid_to_group_predictions = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                   zip(pairs_id, sorted_group_names)}

    return pairid_to_predictions, pairid_to_group_predictions


def generate_cirr_test_predictions(models: dict, relative_test_dataset: CIRRDataset,
                                   index_names: List[str], index_features: torch.tensor, img_weight) -> \
        Tuple[torch.tensor, List[str], List[List[str]], List[str]]:
    """
    Compute CIRR predictions on the test set
    :param models: models to be used for inference, a dict of models
    :param relative_test_dataset: CIRR test dataset in relative mode
    :param index_features: test index features
    :param index_names: test index names

    :return: predicted_features, reference_names, group_members and pairs_id
    """
    print(f"Compute CIRR test predictions")

    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32,
                                      num_workers=multiprocessing.cpu_count(), pin_memory=True)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize pairs_id, predicted_features, group_members and reference_names
    pairs_id = []
    predicted_features = []
    group_members = []
    reference_names = []

    for batch_pairs_id, batch_reference_names, captions, batch_group_members in tqdm(
            relative_test_loader):  # Load data
        batch_group_members = np.array(batch_group_members).T.tolist()  # Convert to tensor
        captions = captions.to(device)

        # Compute the predicted features
        with torch.no_grad():
            text_features = models['text_encoder'](captions).float()

            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features

            # encoder not normalize
            if models['compositor'].code() == 'Combiner':
                models['compositor'] = models['compositor'].to(device)
                batch_predicted_features = F.normalize(models['compositor'].combine_features(reference_image_features * img_weight, text_features), dim=-1)
            else:
                composed_embeds = img_weight * (F.normalize)(reference_image_features, dim=-1) + F.normalize(
                    text_features, dim=-1)
                batch_predicted_features = F.normalize(models['compositor'](composed_embeds), dim=-1) # (batch_size, feature_dim)

        predicted_features.append(batch_predicted_features) # [tensor, tensor, ...]
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)
        pairs_id.extend(batch_pairs_id)

    return torch.cat(predicted_features).to(device), reference_names, group_members, pairs_id


def load_model(model_path, models: dict):
    """
    Load a test model
    :param model_path: path to the model
    :param models: dict of models
    :return: the model with checkpoint loaded
    """
    print(f"Load model")
    # 原先以字典的方式保存为 .pth
    log_data = torch.load(model_path)['model_state_dict']
    for model_name in models.keys():
        # models[model_name].load_state_dict(log_data[model_name])
        if model_name in log_data.keys():
            if isinstance(models[model_name], nn.DataParallel):
                models[model_name].module.load_state_dict(log_data[model_name])
            else:
                models[model_name].load_state_dict(log_data[model_name])
    return models


from data import transform_factory
def main():
    configs = get_experiment_config()
    export_root, configs = setup_experiment(configs)
    models = create_models(configs) # pretrain_dataset != 'None' can get model dict

    image_transforms, text_transforms = transform_factory(configs)
    image_transforms, text_transforms = image_transforms['val'], text_transforms['val']

    if configs["checkpoint_path"] != "None":
        models = load_model(configs["checkpoint_path"], models)
    print(models.keys())
    file_name = "".join(configs["checkpoint_path"].split("/")[-2:])
    generate_cirr_test_submissions(models, file_name, image_transforms, text_transforms, **configs)


if __name__ == '__main__':
    main()
