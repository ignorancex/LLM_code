"""
Preprocess and load datasets for training.
"""
import os
import functools
import io
import json
import math
import re
import random
import pickle
import numpy as np
import torch
import torchvision
from PIL import Image
from scipy.optimize import linear_sum_assignment
from loguru import logger
from src.metrics.ocr_free_utils import extract_text_with_color, flatten_tree

from torch_geometric.loader import DataLoader

from train.data_utils import *

def preprocess_image(sample, image_processor):
    """
    Convert images to tensors for training.
    Augmentations: random horizontal flip.
    Normalization handled by wds.
    """
    image = [image_processor(s).unsqueeze(0) for s in sample]
    image = torch.cat(image, dim=0)
    image = torchvision.transforms.RandomHorizontalFlip(p=0.5)(image)
    return image


def preprocess_websight_text(sample, tokenizer):
    """
    Preprocess text for WebSight HF.
    Captions are truncated to 1024 tokens by default.
    """
    tokenizer.padding_side = "right"
    sample = [
        (f"<graph><image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample
    ]
    text = tokenizer(
        sample,
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    return text["input_ids"][0], text["attention_mask"][0]

def prepare_output_from_html(pth_to_html, idea):
    text_list = flatten_tree(extract_text_with_color(pth_to_html))
    with open(pth_to_html, 'r') as file:
        html_content = file.read()

    output_template = f"""Generate HTML code for the following webpage idea: {idea}
    Generated HTML code:
    {html_content}"""
    return output_template


def get_websight_dataset(batch_size, image_processor, clip_model, tokenizer, vision_encoder_path, num_samples=50):
    """
    Retrieves a dataset of websight data using the HuggingFace M4/WebSight dataset.
    
    Returns:
        DataLoader: A DataLoader object containing the websight dataset.
    """
    
    # create two preprocess functions that take in the passed in image_processor and tokenizer
    preprocess_image_fn = functools.partial(
        preprocess_image, image_processor=image_processor
    )
    preprocess_text_fn = functools.partial(preprocess_websight_text, tokenizer=tokenizer)
    
    websight_list = []
    
    if os.path.exists(f"./websight_hf/data_{num_samples}.pickle"):
        if torch.distributed.get_rank() == 0:
            logger.info("Loading data...")
        with open(f"./websight_hf/data_{num_samples}.pickle", 'rb') as handle:
            websight_hf = pickle.load(handle)
        websight_hf = websight_hf[:num_samples]
        if torch.distributed.get_rank() == 0:
            logger.info("Completely loaded data") 
    else:
        websight_hf = get_data_from_hf(
            num_samples=num_samples, 
            preprocess_image_fn=preprocess_image_fn, 
            vision_encoder_path=vision_encoder_path,
            clip_model=clip_model
        )
        with open(f"./websight_hf/data_{num_samples}.pickle", 'wb') as handle:
            pickle.dump(websight_hf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    if torch.distributed.get_rank() == 0:
        logger.info("Preparing DataLoader...")

    with open('./websight_hf/webpage_idea.json') as json_file:
        webpage_idea = json.load(json_file)

    for idx, dt in tqdm(enumerate(websight_hf)):
        processed_image = preprocess_image_fn([dt['image']]).squeeze(0)
        pth_to_html = f"./websight_hf/html/{idx}.html"
        idea = webpage_idea[str(idx)]
        text_input = prepare_output_from_html(pth_to_html, idea)
        text_input_ids, attention_mask = preprocess_text_fn([text_input])
        
        processed_dt = dict()
        processed_dt['image'] = processed_image
        processed_dt['text'] = torch.tensor(text_input_ids, dtype=torch.long)
        processed_dt['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
        processed_dt['graph'] = dt['graph']
        websight_list.append(processed_dt)
    if torch.distributed.get_rank() == 0:
        logger.info("Completely prepared DataLoader.")
    
    ds = WebsightDataset(websight_list)
    train_sampler = DistributedSampler(ds, shuffle=True)
    train_loader = DataLoader(
        ds, 
        batch_size=batch_size,
        num_workers=0,
        sampler=train_sampler, 
        pin_memory=True,
        drop_last=True
    )
    return train_loader, len(train_loader)