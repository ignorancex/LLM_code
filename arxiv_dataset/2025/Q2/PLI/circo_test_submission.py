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

from data.circo import CIRCODataset, _DEFAULT_CIRCO_DATASET_ROOT
from data import transform_factory
from lavis.models import load_model_and_preprocess
from options import get_experiment_config
from set_up import setup_experiment
from models import create_models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()

def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def circo_generate_test_predictions(models, relative_test_dataset: CIRCODataset,
                                    index_features: torch.tensor, index_names: List[str], img_weight) \
        -> [torch.Tensor, List[List[str]]]:
    """
    Generate the test prediction features for the CIRCO dataset
    Simply copy from cirr_test_submission.py
    """
    print("Generate test prediction features")
    # Create the test dataloader
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=10,
                                      pin_memory=False, collate_fn=collate_fn, shuffle=False)
    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))
    predicted_features_list = []
    query_ids_list = []
    ref_ids = []

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        reference_img, batch_reference_names, captions, shared_concept, query_ids = batch
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
            if models['compositor'].code() == 'Combiner':
                models['compositor'] = models['compositor'].to(device)
                batch_predicted_features = F.normalize(
                    models['compositor'].combine_features(reference_image_features, text_features), dim=-1)
            else:
                composed_embeds = img_weight * F.normalize(reference_image_features, dim=-1) + F.normalize(
                    text_features, dim=-1)
                batch_predicted_features = F.normalize(models['compositor'](composed_embeds),
                                                       dim=-1)  # (batch_size, feature_dim)

        predicted_features_list.append(batch_predicted_features)
        query_ids_list.extend(query_ids)
        ref_ids.extend(batch_reference_names)

    predicted_features = torch.vstack(predicted_features_list).to(device)
    return predicted_features, query_ids_list, ref_ids


def circo_generate_test_dict(relative_test_dataset: CIRCODataset, clip_model: CLIP, index_features: torch.Tensor,
                             index_names: List[str], img_weight) \
        -> Dict[str, List[str]]:
    """
    Generate the test submission dicts for the CIRCO dataset given the pseudo tokens
    """

    # Get the predicted features
    predicted_features, query_ids, ref_ids = circo_generate_test_predictions(clip_model, relative_test_dataset,
                                                                    index_features, index_names, img_weight)

    # Normalize the features
    index_features = index_features.float().to(device)
    index_features = F.normalize(index_features, dim=-1)

    # Compute the similarity
    similarity = predicted_features @ index_features.T # (num_queries, num_images)
    # if ref_ids == index_names, mask the same image
    for i in range(len(ref_ids)):
        similarity[i, index_names.index(ref_ids[i])] = -np.inf

    sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Generate prediction dicts
    queryid_to_retrieved_images = {query_id: query_sorted_names[:50].tolist() for
                                   (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)}

    return queryid_to_retrieved_images
@torch.no_grad()
def circo_generate_test_submission_file(models, submission_name: str, image_transform, text_transform, img_weight) -> None:
    """
    Generate the test submission file for the CIRCO dataset given the pseudo tokens
    """
    # evaluation
    for key in models.keys():
        models[key].eval()
    # Compute the index features
    classic_test_dataset = CIRCODataset(root_path=_DEFAULT_CIRCO_DATASET_ROOT, split='test', mode='classic', img_transform=image_transform, text_transform=text_transform)
    # Simple reuse the function from cirr_test_submission.py
    def extract_index_features(dataset, visual):
        """
        Extract CIRR index features
        :param dataset: CIRR dataset in 'classic' mode
        :param visual: visual model
        :return: a tensor of features and a list of images
        """
        classic_val_loader = DataLoader(dataset=dataset, batch_size=32, num_workers=multiprocessing.cpu_count(),
                                        pin_memory=True, collate_fn=collate_fn)
        index_features, index_names = [], []
        for img, name in tqdm(classic_val_loader):
            img = img.to(device, non_blocking=True)
            with torch.no_grad():
                batch_features = visual(img)  # (batch_size, feature_dim)
                index_features.append(batch_features)
                index_names.extend(name)
        index_features = torch.cat(index_features, dim=0).to(device, non_blocking=True)  # (num_images, feature_dim)
        return index_features, index_names
    index_features, index_names = extract_index_features(classic_test_dataset, models['image_encoder'])

    relative_test_dataset = CIRCODataset(root_path=_DEFAULT_CIRCO_DATASET_ROOT, split='test', mode='relative', img_transform=image_transform, text_transform=text_transform)

    # Get the predictions dict
    queryid_to_retrieved_images = circo_generate_test_dict(relative_test_dataset, models, index_features,
                                                           index_names, img_weight)

    submissions_folder_path = Path(os.path.join('data', "submission", 'CIRCO'))
    submissions_folder_path.mkdir(exist_ok=True, parents=True)
    print("Save submission file to: ", submissions_folder_path / f"{submission_name}.json")

    with open(submissions_folder_path / f"{submission_name}.json", 'w+') as file:
        json.dump(queryid_to_retrieved_images, file, sort_keys=True)
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
def main():
    configs = get_experiment_config()
    export_root, configs = setup_experiment(configs)
    models = create_models(configs)  # pretrain_dataset != 'None' can get model dict

    image_transforms, text_transforms = transform_factory(configs)
    image_transforms, text_transforms = image_transforms['val'], text_transforms['val']

    if configs["checkpoint_path"] != "None":
        models = load_model(configs["checkpoint_path"], models)

    img_weight = configs['img_weight'] if 'img_weight' in configs else 0.5
    file_name = "".join(configs["checkpoint_path"].split("/")[-2:])
    circo_generate_test_submission_file(models, file_name, image_transforms, text_transforms, img_weight)


if __name__ == '__main__':
    main()
