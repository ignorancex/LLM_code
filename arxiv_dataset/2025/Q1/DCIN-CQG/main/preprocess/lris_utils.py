from pathlib import Path

import timm
import torch
from torchvision import transforms
import torch.nn.functional as F

import pickle as pkl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from PIL import Image, ImageOps
from typing import List

def imread_pil(img_path: str, exif_transpose=True):
    img = Image.open(img_path).convert('RGB')
    if exif_transpose:
        img = ImageOps.exif_transpose(img)

    return img

def load_model_and_db(model_name: str, db_root: str='extracted_features_2k_data', device:str='cuda'):
    """ Load pre-trained model and the corresponding vector database """
    # load timm model
    avai_models = [
        'resnet50',
        'vit_large_patch14_clip_336.openai_ft_in12k_in1k',
        'convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384',
        'swinv2_large_window12to24_192to384.ms_in22k_ft_in1k',
    ]
    assert model_name in avai_models, f'{model_name} not in {avai_models}'

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.to(device)

    # load vector DB
    with open(Path(f"{db_root}/{model_name}_feature_dict_v2.pkl"), "rb") as f:
        feature_dict = pkl.load(f)

    feature_vectors = []
    ref_img_paths = list(feature_dict.keys())
    for img_path in feature_dict:
        feature_vectors.append(feature_dict[img_path])

    feature_vectors = np.array(feature_vectors)

    return model, feature_vectors, ref_img_paths

def search_ref_image(query_img_path: str, model: torch.nn.Module, feature_vectors: np.ndarray, 
                     ref_img_paths: List, img_size: int, device: str='cuda'):
    """
    Search the vector database and return the reference path based on query image
    Args:
        query_img_path (str): path to the query image
        model (torch.nn.Module): timm pre-trained model
        feature_vectors (np.ndarray): reference feature vectors
        ref_img_paths (str): reference image paths
        img_size (int): image input size
        device (str): GPU device for inference
    Return:
        ref_img_path (str): path to the found ref image
    """
    # define data transform
    data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # load query image
    query_img = imread_pil(query_img_path)
    query_img = data_transform(query_img)
    query_img = query_img.unsqueeze(0).to(device)

    # extract query feature
    model.eval()
    with torch.backends.cudnn.flags(deterministic=True, benchmark=True):
        with torch.no_grad():
            query_feat = model(query_img)
    # normalize the feature to unit norm
    query_feat = query_feat.detach().cpu()
    query_feat = F.normalize(query_feat, dim=1)
    query_feat = query_feat.numpy()

    # compute cosine similarity
    cosine_similarities = cosine_similarity(feature_vectors, query_feat)

    # find the index of the most similar vector
    most_similar_index = np.argmax(cosine_similarities)

    return ref_img_paths[most_similar_index]