"""
Util functions for initializing webdataset objects
"""

import time
import json
import os
import re
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import open_clip

from tqdm import tqdm
from numpy import asarray
from loguru import logger
from PIL import Image
from einops import rearrange
from torch_geometric.data import HeteroData, Dataset

from paddleocr import PaddleOCR

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Constants
MIN_OVERLAP_PERCENTAGE = 80
IMAGE_SIZE_THRESHOLD = 3000 * 3000

# Helper functions
def replace_img_src(content):
    pattern = r'(<img[^>]*src=")https://[^"]*/(\d+)x(\d+)(/[^"]*")'
    replacement = r'\1rick.jpg" width="\2" height="\3"'
    return re.sub(pattern, replacement, content)

def masking_as_black_boxes(image, bboxes, color=(0, 0, 0)):
    image_copy = image.copy()
    for x, y, u, v in bboxes:
        x_min = min(int(x[0]), int(y[0]), int(u[0]), int(v[0]))
        x_max = max(int(x[0]), int(y[0]), int(u[0]), int(v[0]))
        y_min = min(int(x[1]), int(y[1]), int(u[1]), int(v[1]))
        y_max = max(int(x[1]), int(y[1]), int(u[1]), int(v[1]))
        cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), color, thickness=-1)
    return image_copy

def get_overlap_area(bbox1, bbox2):
    """
    Given two bounding boxes, return the area of their intersection
    """
    i_bbox = (max(bbox1[0], bbox2[0]), max(bbox1[1], bbox2[1]), min(bbox1[2], bbox2[2]), min(bbox1[3], bbox2[3]))
    x_min, y_min, x_max, y_max = i_bbox
    if x_max < x_min or y_max < y_min:
        return 0
    return (x_max - x_min + 1) * (y_max - y_min + 1)

# OCR related functions
def process_ocr_result(ocr_result, PIL_image):
    if ocr_result is None:
        print(f"Image failed OCR (it could because there is no text in the image)")
        return {
            "ocr_info": [],
            "size": PIL_image.size
        }
    
    bboxes = [line[0] for line in ocr_result]
    txts = [line[1][0] for line in ocr_result]
    
    ocr_info = []
    for i in range(len(ocr_result)):
        x, y, u, v = bboxes[i]
        x_min = min(int(x[0]), int(y[0]), int(u[0]), int(v[0]))
        x_max = max(int(x[0]), int(y[0]), int(u[0]), int(v[0]))
        y_min = min(int(x[1]), int(y[1]), int(u[1]), int(v[1]))
        y_max = max(int(x[1]), int(y[1]), int(u[1]), int(v[1]))
        ocr_info.append(((x_min, y_min, x_max, y_max), txts[i]))
    
    return {
        "ocr_info": ocr_info,
        "size": PIL_image.size
    }

def get_masked_image_from_ocr(num_samples, ocr_model, ocr_save_dir, save_dir, pth_to_ocr_results, websight_ds):
    ocr_results = {}
    for img_idx, websight_dt in enumerate(websight_ds):
        if int(img_idx) >= num_samples:
            break
        
        PIL_image = websight_dt['image']
        numpy_image = asarray(PIL_image)
        ocr_result = ocr_model.ocr(numpy_image, cls=True)[0]

        ocr_results[img_idx] = process_ocr_result(ocr_result, PIL_image)
        
        # Export images in which the texts are masked
        height, width = numpy_image.shape[:2]
        colored_image = masking_as_black_boxes(image=numpy_image, bboxes=[info[0] for info in ocr_results[img_idx]["ocr_info"]])
        cv2.imwrite(f"{ocr_save_dir}{img_idx}_masked.png", colored_image)       
    
    if not os.path.exists(pth_to_ocr_results):
        with open(pth_to_ocr_results, "w") as outfile: 
            json.dump(ocr_results, outfile, indent=4)

# Segmentation related functions
def export_image(anns, pth_to_seg_img):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    img = ((img) * 255).astype(np.uint8)
    im_show = Image.fromarray(img)
    im_show.save(pth_to_seg_img)

def get_segmentation_from_masked_image(
    num_samples,
    sam_model_ckpt,
    sam_model_type,
    seg_save_dir,
    ocr_save_dir,
    save_dir,
    device,
    pth_to_seg_results
):
    sam = sam_model_registry[sam_model_type](checkpoint=sam_model_ckpt)
    sam.to(device=device)
    sam.eval()
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    img_paths = []
    for filename in os.listdir(ocr_save_dir):
        if filename[:-4].endswith("masked"):
            img_idx = filename[:-11]
            img_paths.append((ocr_save_dir + filename, img_idx))

    segmented_area = {}
    for img_path, img_idx in tqdm(img_paths):
        if int(img_idx) >= num_samples:
            continue
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        masks = mask_generator.generate(image)

        segmented_area[img_idx] = []
        for mask in masks:
            x_min = mask['bbox'][0]
            y_min = mask['bbox'][1]
            x_max = x_min + mask['bbox'][2]
            y_max = y_min + mask['bbox'][3]
            segmented_area[img_idx].append((x_min, y_min, x_max, y_max))
        
        pth_to_seg_img = f"{seg_save_dir}{img_idx}_segmented.png"
        if not os.path.exists(pth_to_seg_img):
            export_image(masks, pth_to_seg_img)
    
    if not os.path.exists(pth_to_seg_results):
        with open(pth_to_seg_results, "w") as outfile: 
            json.dump(segmented_area, outfile, indent=4)

# Graph preparation functions
def prepare_graph_input(
    num_samples,
    pth_to_ocr_results,
    pth_to_seg_results,
    preprocess_image_fn,
    vision_encoder_path,
    clip_model,
    save_dir,
    device,
    websight_ds
):
    ocr_results = json.load(open(pth_to_ocr_results))
    seg_results = json.load(open(pth_to_seg_results))
    tokenizer = open_clip.get_tokenizer(vision_encoder_path)
    clip_model.eval()
    clip_model.to(device)
    img_idx_to_graph = {}
    for img_idx in tqdm(seg_results.keys()):
        if int(img_idx) >= num_samples:
            continue

        graph = HeteroData()
        img = websight_ds[int(img_idx)]['image']
        
        components_info = []
        texts_info = []
        
        # Add full image to list of image components
        seg_results[img_idx].append((0, 0, ocr_results[img_idx]["size"][1], ocr_results[img_idx]["size"][0]))
        # Store components' info
        for bbox in seg_results[img_idx]:
            x_min, y_min, x_max, y_max = bbox
            component = img.crop((x_min, y_min, x_max+1, y_max+1))
            component_x = preprocess_image_fn([component]).squeeze(0).to(device)
            component_x = rearrange(component_x, "c h w -> 1 c h w")
            component_x = clip_model.encode_image(component_x)
            
            height, width = component.convert("RGB").size
            if width == 1:
                continue
            components_info.append((bbox, component_x))
        # Store texts' info
        for text_info in ocr_results[img_idx]["ocr_info"]:
            text = text_info[1]
            bbox = text_info[0]
            text_token_ids = tokenizer([text]).to(device)
            text_embed = clip_model.encode_text(text_token_ids)
            texts_info.append((text, bbox, text_embed.to("cpu")))
        
        # Prepare edges
        edge_text_to_comp = []
        edge_comp_to_text = []
        edge_comp_to_comp = []
        
        for component_id, component in enumerate(components_info):
            for text_id, text_info in enumerate(ocr_results[img_idx]["ocr_info"]):
                component_bbox = component[0]
                text = text_info[1]
                text_bbox = text_info[0]
                text_area = (text_bbox[2] - text_bbox[0] + 1) * (text_bbox[3] - text_bbox[1] + 1)
                overlap_area = get_overlap_area(text_bbox, component_bbox)
                if overlap_area / text_area * 100 >= MIN_OVERLAP_PERCENTAGE:
                    edge_text_to_comp.append((text_id, component_id))
                    edge_comp_to_text.append((component_id, text_id))
            
        for node_u, img_u in enumerate(components_info):
            for node_v, img_v in enumerate(components_info):
                if node_u != node_v:
                    u_bbox = img_u[0]
                    v_bbox = img_v[0]
                    area_u = (u_bbox[2] - u_bbox[0] + 1) * (u_bbox[3] - u_bbox[1] + 1)
                    overlap_area = get_overlap_area(u_bbox, v_bbox)
                    if overlap_area / area_u * 100 >= MIN_OVERLAP_PERCENTAGE:
                        edge_comp_to_comp.append((node_u, node_v))
                        edge_comp_to_comp.append((node_v, node_u))
                
        tmp = []
        for text_feature in texts_info:
            tmp.append(text_feature[2])
        if len(tmp) > 0:
            graph["text"].x = torch.stack(tmp, dim=0)

        tmp = []
        for comp_feature in components_info:
            tmp.append(comp_feature[1])
        if len(tmp) > 0:
            graph["image"].x = torch.stack(tmp, dim=0)
        
        if len(edge_text_to_comp) > 0:
            graph["text", "to", "image"].edge_index = torch.Tensor(edge_text_to_comp).T.to(torch.int64)
        if len(edge_comp_to_text) > 0:
            graph["image", "to", "text"].edge_index = torch.Tensor(edge_comp_to_text).T.to(torch.int64)
        if len(edge_comp_to_comp) > 0:
            graph["image", "to", "image"].edge_index = torch.Tensor(edge_comp_to_comp).T.to(torch.int64)
        graph.to(device)
        graph = graph.to_homogeneous()
        graph.x = graph.x.to(torch.float32)
        graph.x = rearrange(graph.x, "a 1 b -> a b")
        graph.edge_index  = graph.edge_index.to(torch.int64)
        graph.to("cpu")
        img_idx_to_graph[img_idx] = graph
        
    return img_idx_to_graph

# Main data loading function
def get_data_from_hf(
    num_samples, 
    preprocess_image_fn,
    vision_encoder_path,
    clip_model,
    save_dir = "./websight_hf/"
):
    from datasets import load_dataset
    
    # Load and filter dataset
    ds = load_dataset("HuggingFaceM4/WebSight", "v0.2")
    websight_ds = [dt for dt in ds['train'].take(30000) 
                   if dt['image'].size[0] * dt['image'].size[1] <= IMAGE_SIZE_THRESHOLD][:num_samples]
    del ds

    # Create directories
    directories = {
        'html': save_dir + "html/",
        'raw': save_dir + "raw/",
        'ocr': save_dir + "ocr2/",
        'seg': save_dir + "segmentation2/"
    }
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)

    # Process HTML and save screenshots
    for idx, dt in enumerate(websight_ds):
        process_html_and_screenshot(dt, idx, directories)

    # Prepare paths and convert images
    pth_to_ocr_results = save_dir + f"ocr_results_{num_samples}.json"
    pth_to_seg_results = save_dir + f"segmentation_results_{num_samples}.json"
    convert_images(websight_ds, directories['raw'])

    # Perform OCR, segmentation, and graph preparation
    perform_ocr(websight_ds, save_dir, directories, pth_to_ocr_results)
    perform_segmentation(websight_ds, save_dir, directories, pth_to_seg_results)
    prepare_graphs(websight_ds, pth_to_ocr_results, pth_to_seg_results, preprocess_image_fn, vision_encoder_path, clip_model, save_dir)

    return websight_ds

# Helper functions for get_data_from_hf
def process_html_and_screenshot(dt, idx, directories):
    formated_html = replace_img_src(dt['text'])
    pth_to_html = f"{directories['html']}{idx}.html"
    pth_to_raw = f"{directories['raw']}{idx}.png"
    with open(pth_to_html, "w") as file:
        file.write(formated_html)
    os.system(f"python eval/screenshot_single.py --html {pth_to_html} --png {pth_to_raw}")

def convert_images(websight_ds, raw_dir):
    pth_to_raw = {}
    for filename in os.listdir(raw_dir):
        if filename.endswith("png"):
            idx = filename[:-4]
            pth_to_raw[int(idx)] = raw_dir + filename
    start = time.time()
    for idx in range(len(websight_ds)):
        websight_ds[idx]['image'] = Image.open(pth_to_raw[idx]).convert("RGB")
    end = time.time()
    print(f"Time to convert: {end-start} seconds")

def perform_ocr(websight_ds, save_dir, directories, pth_to_ocr_results):
    ocr_model = PaddleOCR(use_angle_cls=True)
    logger.info("Prepare masked images...")
    get_masked_image_from_ocr(
        num_samples=len(websight_ds),
        ocr_model=ocr_model,
        ocr_save_dir=directories['ocr'],
        save_dir=save_dir,
        pth_to_ocr_results=pth_to_ocr_results,
        websight_ds=websight_ds
    )

def perform_segmentation(websight_ds, save_dir, directories, pth_to_seg_results):
    sam_model_ckpt = "sam_vit_b_01ec64.pth"
    sam_model_type = "vit_b"
    device="cuda"
    logger.info("Prepare segmented images...")
    get_segmentation_from_masked_image(
        num_samples=len(websight_ds),
        sam_model_ckpt=sam_model_ckpt,
        sam_model_type=sam_model_type,
        seg_save_dir=directories['seg'],
        ocr_save_dir=directories['ocr'],
        save_dir=save_dir,
        device=device,
        pth_to_seg_results=pth_to_seg_results
    )

def prepare_graphs(websight_ds, pth_to_ocr_results, pth_to_seg_results, preprocess_image_fn, vision_encoder_path, clip_model, save_dir):
    img_idx_to_graph = prepare_graph_input(
        num_samples=len(websight_ds),
        pth_to_ocr_results=pth_to_ocr_results,
        pth_to_seg_results=pth_to_seg_results,
        preprocess_image_fn=preprocess_image_fn,
        vision_encoder_path=vision_encoder_path,
        clip_model=clip_model,
        save_dir=save_dir,
        device="cuda",
        websight_ds=websight_ds                                     
    ) 
    for i in range(len(websight_ds)):
        websight_ds[i]["graph"] = img_idx_to_graph[str(i)]

class WebsightDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]