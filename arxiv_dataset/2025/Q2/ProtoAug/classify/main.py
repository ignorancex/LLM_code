import os
import sys
import json
import time
import math
import random
import datetime
import traceback
from pathlib import Path
from os.path import join as ospj
import wandb
import pickle
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import gc
import faiss
import numpy as np
from PIL import Image

from utils import (
    fix_random_seeds,
    cosine_scheduler,
    MetricLogger,
)

from config import get_args
from data import get_data_loader, get_synth_train_data_loader
from models.clip import CLIP
from models.resnet50 import ResNet50
from util_data import SUBSET_NAMES
import os
import cv2
from joblib import Parallel, delayed
import random
from PIL import Image, ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(0)
######################################################################
# FAISS CLUSTERING CODE
######################################################################
import os
def chunkify(lst, chunk_size):
    """Yield successive chunk_size-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]
def load_and_flatten_images_in_batches_pillow(
    image_paths, 
    img_size=(224, 224), 
    n_jobs=4, 
    batch_size=5000
):
    """
    Similar batch-loading for PIL.
    """
    from joblib import Parallel, delayed

    def process_image_pillow(path):
        try:
            with Image.open(path).convert("RGB") as img:
                img = img.resize(img_size, Image.BICUBIC)
                arr = np.array(img, dtype=np.float32) / 255.0
                return arr.flatten()
        except:
            return None

    all_features = []

    for chunk in chunkify(image_paths, batch_size):
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_image_pillow)(p) for p in chunk
        )
        # filter out Nones
        results = [r for r in results if r is not None]
        if not results:
            continue
        chunk_array = np.vstack(results).astype(np.float32)
        all_features.append(chunk_array)

    if len(all_features) == 0:
        return np.zeros((0, img_size[0]*img_size[1]*3), dtype=np.float32)
    else:
        return np.concatenate(all_features, axis=0)
def get_image_paths(base_dir, num_images_per_subfolder=16):
    """
    Get the first N image file paths from each subfolder in base_dir.

    Parameters:
        base_dir (str): The base directory to search for images.
        num_images_per_subfolder (int): Number of images to select from each subfolder.

    Returns:
        list: A list of image file paths.
    """
    image_paths = []
    if num_images_per_subfolder>0:
        for root, dirs, files in os.walk(base_dir):
            # Filter image files in the current folder
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            # Sort files to ensure consistent order (e.g., alphabetical)
            image_files.sort()
            
            # Select the first N images (parameterized)
            selected_images = image_files[:num_images_per_subfolder]
            
            # Add full paths of selected images to the list
            image_paths.extend(os.path.join(root, f) for f in selected_images)
    else:
        image_paths = []
    return image_paths

def load_and_flatten_images(image_paths, img_size=(224, 224),n_jobs = 1):
    """
    Load images, resize to img_size, convert to NumPy arrays and flatten.
    Return a NumPy array of shape (N, D), where N = number of images 
    and D = img_size[0]*img_size[1]*3 (assuming RGB).
    """
    def process_image_pillow(path):
        img = Image.open(path).convert("RGB")
        img = img.resize(img_size, Image.BICUBIC)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.flatten()
        return arr
    data = Parallel(n_jobs=n_jobs)(
        delayed(process_image_pillow)(path) for path in image_paths
    )
    return np.vstack(data).astype(np.float32)

def load_and_flatten_images_cv2(image_paths, img_size=(224, 224), n_jobs=4):
    """
    Load images in parallel, resize to img_size, normalize, and flatten.
    Returns a NumPy array of shape (N, D).
    """
    def process_image(path):
    
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
        # Convert BGR to RGB if needed
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img.flatten()
    
    # Use joblib to parallelize loading/resizing
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_image)(path) for path in image_paths
    )
    return np.vstack(results).astype(np.float32)
def cluster_images_with_faiss(real_dir, synthetic_dir, num_shot, num_image_per_subfolder, num_centroids=10, niter=20, verbose=True):
    # Get image lists
    real_image_paths = get_image_paths(real_dir, num_images_per_subfolder=num_shot)
    synthetic_image_paths = get_image_paths(synthetic_dir,num_images_per_subfolder=num_image_per_subfolder)
    #print(0)
    # Load images as flattened arrays
    if 'sun397' in real_dir:
        real_data = load_and_flatten_images(real_image_paths)
        synthetic_data = load_and_flatten_images(synthetic_image_paths)
    elif 'imagenet' in real_dir:
        real_data = load_and_flatten_images_in_batches_pillow(real_image_paths)
        synthetic_data = load_and_flatten_images_in_batches_pillow(synthetic_image_paths)
    else:
        if num_shot >0:
            real_data = load_and_flatten_images_cv2(real_image_paths)
            synthetic_data = load_and_flatten_images_cv2(synthetic_image_paths)
        else: 
            real_data = []
            synthetic_data = load_and_flatten_images_cv2(synthetic_image_paths)


    print(1)
    # Combine data
    if num_shot > 0:
        all_data = np.vstack([synthetic_data, real_data])  # shape (N, D)
    else:
        all_data = np.vstack([synthetic_data])  # shape (N, D)

    N, D = all_data.shape
    #res = faiss.StandardGpuResources()

    # Use Faiss Kmeans with GPU support
    kmeans = faiss.Kmeans(d=D, k=num_centroids, niter=niter, verbose=verbose,gpu=1)
    kmeans.train(all_data)

    # Assign each image to a cluster
    _, assignments = kmeans.index.search(all_data, 1)
    assignments = assignments.reshape(-1)  # (N,)
    print(2)
    # Build the TS dictionary
    TS = {i: [[], []] for i in range(num_centroids)}
    num_gen = len(synthetic_image_paths)

    for i, cluster_idx in enumerate(assignments):
        if i < num_gen:
            # Real image
            TS[cluster_idx][1].append(i)
        else:
            # Synthetic image
            #synth_index = i - num_real
            TS[cluster_idx][0].append(i)

    return TS, real_image_paths, synthetic_image_paths

def chunkify(lst, chunk_size):
    """Yield successive chunk_size-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]

def load_images_no_resize_cv2(image_paths, batch_size=500, n_jobs=4):
    """
    Chunk-load images (3x224x224) without resizing.
    Yields arrays of shape (chunk_size, 3*224*224).
    """
    import cv2
    from joblib import Parallel, delayed
    import numpy as np

    def process_image(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        # Check if already 224x224. If not, skip or handle it.
        if img.shape[:2] != (224, 224):
            return None
        # Convert BGR -> RGB, float32, [0..1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img.flatten()

    for chunk in chunkify(image_paths, batch_size):
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_image)(p) for p in chunk
        )
        # Filter out None
        results = [r for r in results if r is not None]
        if len(results) == 0:
            continue
        yield np.vstack(results).astype(np.float32)

def load_images_resizing_cv2(
    image_paths, 
    out_size=(224, 224), 
    batch_size=500, 
    n_jobs=4
):
    """
    Chunk-load images (any size, e.g. 512x512) and resize to out_size before flattening.
    Yields arrays of shape (chunk_size, 3*out_size[0]*out_size[1]).
    """
    import cv2
    from joblib import Parallel, delayed
    import numpy as np

    def process_image(path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        # Resize to out_size
        img = cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)
        # Convert BGR -> RGB, float32, [0..1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img.flatten()

    for chunk in chunkify(image_paths, batch_size):
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_image)(p) for p in chunk
        )
        # Filter out None
        results = [r for r in results if r is not None]
        if len(results) == 0:
            continue
        yield np.vstack(results).astype(np.float32)
def cluster_images_with_faiss_imagenet_only(
    real_dir, 
    synthetic_dir, 
    num_image_per_subfolder,
    num_centroids=10, 
    niter=20, 
    verbose=True
):
    """
    1) Train K-Means on real images only (assume they are 3x224x224).
    2) Assign clusters to real images (for indexing).
    3) Assign clusters to synthetic images (resize from 3x512x512 -> 3x224x224).
    4) Return TS, real_paths, synth_paths
    """
    # Get image paths
    real_image_paths = get_image_paths(real_dir, num_images_per_subfolder=num_image_per_subfolder)
    synthetic_image_paths = get_image_paths(synthetic_dir, num_images_per_subfolder=num_image_per_subfolder)

    # Load all real images in chunks => one big array for KMeans
    all_real_arrays = []
    for real_chunk in load_images_resizing_cv2(real_image_paths, batch_size=500, n_jobs=4):
        #print(real_chunk)
        all_real_arrays.append(real_chunk)
    if len(all_real_arrays) == 0:
        # no real data found, return empty
        TS = {i: [[], []] for i in range(num_centroids)}
        return TS, real_image_paths, synthetic_image_paths

    real_data = np.concatenate(all_real_arrays, axis=0)  # shape (N_real, 3*224*224)
    N_real, D = real_data.shape

    # Train kmeans on real data only
    kmeans = faiss.Kmeans(d=D, k=num_centroids, niter=niter, verbose=verbose, gpu=1)
    kmeans.train(real_data)

    # Assign clusters to real images (again chunk loading, to handle huge real sets)
    real_assignments = []
    real_count = 0
    for real_chunk in load_images_resizing_cv2(real_image_paths, batch_size=500, n_jobs=4):
        _, idx = kmeans.index.search(real_chunk, 1)
        real_assignments.extend(idx.flatten())
        real_count += real_chunk.shape[0]
    assert real_count == N_real, "Mismatch in real images assignment count."

    # Assign clusters to synthetic images (resize from 512->224)
    synth_assignments = []
    for synth_chunk in load_images_resizing_cv2(synthetic_image_paths, out_size=(224,224), batch_size=500, n_jobs=4):
        _, idx = kmeans.index.search(synth_chunk, 1)
        synth_assignments.extend(idx.flatten())
    N_synth = len(synth_assignments)

    # Build TS in usual format: TS[cluster_id] = [synth_ids, real_ids]

    TS = {i: [[], []] for i in range(num_centroids)}

    # Synthetic first
    for i, c_idx in enumerate(synth_assignments):
        TS[c_idx][0].append(i)  # i in [0..N_synth-1]

    # Real next
    for j, c_idx in enumerate(real_assignments):
        # offset the real index by N_synth
        TS[c_idx][1].append(j + N_synth)

    return TS, real_image_paths, synthetic_image_paths
def cluster_images_with_faiss_imagenet(
    real_dir, 
    synthetic_dir,
    num_image_per_subfolder=16,
    num_centroids=10, 
    niter=20, 
    verbose=True,
    batch_size=10000,
    memmap_dir="./faiss_memmap",
    dataset_name=None,
):
    """
    - If 'imagenet' in dataset_name -> train KMeans on real 224x224 only.
      Then assign clusters to real & synthetic in chunks, storing assignments
      in memory-mapped arrays. Build TS at the end.
    - No need to return image paths. 
    - We store results in memory-mapped arrays to encourage usage of swap
      if physical RAM is insufficient.
    """

    # Make sure memmap_dir exists
    os.makedirs(memmap_dir, exist_ok=True)

    # Gather image paths
    real_image_paths = get_image_paths(real_dir, 16)
    synthetic_image_paths = get_image_paths(synthetic_dir, num_image_per_subfolder)
    N_real = len(real_image_paths)
    N_synth = len(synthetic_image_paths)
    print(f"[INFO] Found {N_real} real images, {N_synth} synthetic images.")
 
    centroids_path = os.path.join(memmap_dir, "cluster_centroids.npy")
    if not os.path.isfile(centroids_path):
        print("[INFO] Loading all real data to train KMeans (this may use swap if large).")
        all_real_chunks = []
        for chunk in load_images_resizing_cv2(real_image_paths, batch_size=batch_size, n_jobs=4):
            all_real_chunks.append(chunk)
        if len(all_real_chunks) == 0:
            print("[WARN] No real data found; returning empty TS.")
            return {}  # or create an empty TS dict

        real_data = np.concatenate(all_real_chunks, axis=0)
        del all_real_chunks  # free references so Python can reclaim memory
        D = real_data.shape[1]
        print("[INFO] Training KMeans with FAISS...")
        kmeans = faiss.Kmeans(d=D, k=num_centroids, niter=niter, verbose=verbose, gpu=1)
        kmeans.train(real_data)
        del real_data
        print(f"[INFO] Saving centroids to {centroids_path}")
        np.save(centroids_path, kmeans.centroids)
        index = kmeans.index
        
    else:
        centroids = np.load(centroids_path)
        num_centroids = centroids.shape[0]
        D = centroids.shape[1]

        # 2) Build Faiss index
        index = faiss.IndexFlatL2(D)  # L2 distance
        index.add(centroids)   
    # We can also free real_data if we won't need it except for re-assigning
    N_synth = len(synthetic_image_paths)
    # If we haven't created it yet, do so. If it already exists, open in 'r+' mode.
    real_assign_path = os.path.join(memmap_dir, "real_assignments.mmp")
    synth_assign_path = os.path.join(memmap_dir, "synth_assignments.mmp")
    if not os.path.isfile(real_assign_path):
        print(f"[INFO] Creating new memmap for {N_synth} synthetic assignments with -1.")
        real_assign_mm = np.memmap(real_assign_path, dtype='int32', mode='w+', shape=(N_synth,))
        real_assign_mm[:] = 0
        real_assign_mm.flush()
    else:
        # open existing
        print(f"[INFO] Opening existing memmap for synthetic assignments.")
        real_assign_mm = np.memmap(real_assign_path, dtype='int32', mode='r+')
        if len(real_assign_mm) != N_real:
            raise ValueError(f"Memmap shape {len(real_assign_mm)} != number of synthetic images {N_synth}")
    # If doesn't exist, create it filled with -1
    if not os.path.isfile(synth_assign_path):
        print(f"[INFO] Creating new memmap for {N_synth} synthetic assignments with -1.")
        synth_assign_mm = np.memmap(synth_assign_path, dtype='int32', mode='w+', shape=(N_synth,))
        synth_assign_mm[:] = 0
        synth_assign_mm.flush()
    else:
        # open existing
        print(f"[INFO] Opening existing memmap for synthetic assignments.")
        synth_assign_mm = np.memmap(synth_assign_path, dtype='int32', mode='r+')
        if len(synth_assign_mm) != N_synth:
            raise ValueError(f"Memmap shape {len(synth_assign_mm)} != number of synthetic images {N_synth}")
    start_idx = 0

    for chunk_data in load_images_resizing_cv2(real_image_paths, (224,224), batch_size, n_jobs=4):
        chunk_size = chunk_data.shape[0]
        _, idx = index.search(chunk_data, 1)
        real_assign_mm[start_idx : start_idx+ chunk_size] = idx.reshape(-1)
        start_idx += chunk_size
        real_assign_mm.flush()
        del chunk_data
        del idx
        gc.collect()
    # 4) Assign [start_offset..end_offset], chunk by chunk
    start_offset = 0
    end_offset = 252000
    end_offset = min(end_offset, N_synth)  # clamp to the total available
    subset_paths = synthetic_image_paths[start_offset:end_offset]
    current_idx = start_offset
    # Memory-map arrays to store assignments

    print("[INFO] Assigning clusters to synthetic images...")
    for chunk_data in load_images_resizing_cv2(subset_paths, (224,224), batch_size, n_jobs=4):
        chunk_size = chunk_data.shape[0]
        _, idx = index.search(chunk_data, 1)
        synth_assign_mm[current_idx : current_idx+ chunk_size] = idx.reshape(-1)
        current_idx += chunk_size
        synth_assign_mm.flush()
        del chunk_data
        del idx
        gc.collect()
        print(f"[INFO] Assigned chunk. {current_idx}/{end_offset} done. Freed memory ...")

    # Build TS = { cluster_id: [list_of_synth_indices, list_of_real_indices], ... }
    # Because you no longer want to return the image paths, we build TS with integer IDs only.
    print("[INFO] Building TS dictionary in memory. If extremely large, store on disk.")
    TS = {i: [[], []] for i in range(num_centroids)}

    # Synthetic (indices [0..N_synth-1])
    # We'll chunk-read from the memmap in slices to avoid reading everything at once if it's huge
    chunk_size = 10000  # pick any
    start = 0
    while start < N_synth:
        end = min(start + chunk_size, N_synth)
        chunk_idx = synth_assign_mm[start:end]  # shape: (end-start,)
        # Append each image index to TS
        for i, c_idx in enumerate(chunk_idx, start=start):
            TS[c_idx][0].append(i)
        start = end

    # 6b) Real (indices offset by N_synth => [N_synth..N_synth+N_real-1])
    start = 0
    while start < N_real:
        end = min(start + chunk_size, N_real)
        chunk_idx = real_assign_mm[start:end]
        for j, c_idx in enumerate(chunk_idx, start=start):
            TS[c_idx][1].append(j + N_synth)
        start = end

    return TS
######################################################################
# TRAINING AND EVAL CODE
######################################################################

def load_data_loader(args):
    train_loader, test_loader = get_data_loader(
        real_train_data_dir=args.real_train_data_dir,
        real_test_data_dir=args.real_test_data_dir,
        metadata_dir=args.metadata_dir,
        dataset=args.dataset, 
        bs=args.batch_size,
        eval_bs=args.batch_size_eval,
        n_img_per_cls=args.n_img_per_cls,
        is_synth_train=args.is_synth_train,
        n_shot=args.n_shot,
        real_train_fewshot_data_dir=args.real_train_fewshot_data_dir,
        is_pooled_fewshot=args.is_pooled_fewshot,
        model_type=args.model_type,
    )
    return train_loader, test_loader

def load_synth_train_data_loader(args):
    synth_train_loader = get_synth_train_data_loader(
        synth_train_data_dir=args.synth_train_data_dir,
        bs=args.batch_size,
        n_img_per_cls=args.n_img_per_cls,
        dataset=args.dataset,
        n_shot=args.n_shot,
        real_train_fewshot_data_dir=args.real_train_fewshot_data_dir,
        is_pooled_fewshot=args.is_pooled_fewshot,
        model_type=args.model_type,
    )
    return synth_train_loader

@torch.no_grad()
def get_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]

@torch.no_grad()
def eval(model, criterion, data_loader, epoch, fp16_scaler, args):
    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    is_last = epoch + 1 == args.epochs
    if is_last:
        targets = []
        outputs = []

    model.eval()

    for it, (image, label) in enumerate(
        metric_logger.log_every(data_loader, 100, header)
    ):
        image = image.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # compute output
        with torch.amp.autocast("cuda"):
            output = model(image, phase="eval")
            loss = criterion(output, label)

        acc1, acc5 = get_accuracy(output, label, topk=(1, 5))

        # record logs
        metric_logger.update(loss=loss.item())
        metric_logger.update(top1=acc1.item())
        metric_logger.update(top5=acc5.item())

        if is_last:
            targets.append(label)
            outputs.append(output)

    metric_logger.synchronize_between_processes()
    print("Averaged test stats:", metric_logger)

    stat_dict = {"test/{}".format(k): meter.global_avg for k, meter in metric_logger.meters.items()}

    if is_last:
        targets = torch.cat(targets)
        outputs = torch.cat(outputs)

        # calculate per class accuracy
        acc_per_class = [
            get_accuracy(outputs[targets == cls_idx], targets[targets == cls_idx], topk=(1,))[0].item() 
            for cls_idx in range(args.n_classes)
        ]
        for cls_idx, acc in enumerate(acc_per_class):
            print("{} [{}]: {}".format(SUBSET_NAMES[args.dataset][cls_idx], cls_idx, str(acc)))
            stat_dict[SUBSET_NAMES[args.dataset][cls_idx] + '_cls-acc'] = acc

    return stat_dict


def save_model(args, model, optimizer, epoch, fp16_scaler, file_name):
    state_dict = model.state_dict()
    save_dict = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch + 1,
        "args": args,
    }
    if fp16_scaler is not None:
        save_dict["fp16_scaler"] = fp16_scaler.state_dict()
    torch.save(save_dict, os.path.join(args.output_dir, file_name))

def train_one_epoch(
    model, criterion, data_loader, optimizer, scheduler, epoch, fp16_scaler, cutmix_or_mixup, args,
    val_loader, best_stats, best_top1, global_idx_to_region=None
):
    args.n_classes = len(SUBSET_NAMES[args.dataset])

    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    model.train()
    total_regu_time = 0

    for it, batch in enumerate(
        metric_logger.log_every(data_loader, 100, header)
    ):
        if args.is_synth_train and args.is_pooled_fewshot:
            image, label, is_real, global_idx = batch
        elif args.is_synth_train and args.is_pooled_fewshot == False:
            image, label, is_real, global_idx = batch
        else:
            image, label = batch

            if not args.is_synth_train or not args.is_pooled_fewshot:
                is_real = torch.ones_like(label)  # placeholder

        label_origin = label
        label_origin = label_origin.cuda(non_blocking=True)

        # CutMix and MixUp augmentation
        if args.is_mix_aug:
            p = random.random()
            if p < 0.2:
                if args.is_synth_train and args.is_pooled_fewshot:
                    new_image = torch.zeros_like(image)
                    new_label = torch.stack([torch.zeros_like(label)] * args.n_classes, dim=1).mul(1.0)

                    image_real, label_real = image[is_real==1], label[is_real==1]
                    image_synth, label_synth = image[is_real==0], label[is_real==0]

                    image_real, label_real = cutmix_or_mixup(image_real, label_real)
                    image_synth, label_synth = cutmix_or_mixup(image_synth, label_synth)

                    new_image[is_real==1] = image_real
                    new_image[is_real==0] = image_synth
                    new_label[is_real==1] = label_real
                    new_label[is_real==0] = label_synth

                    image = new_image
                    label = new_label
                else:
                    image, label = cutmix_or_mixup(image, label)

        it_global = len(data_loader) * epoch + it  # global training iteration

        image = image.squeeze(1).to(torch.float16).cuda(non_blocking=True)
        label = label.cuda(non_blocking=True)

        # update weight decay and learning rate according to their schedule
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = args.lr_schedule[it_global]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = args.wd

        # forward pass
        with torch.amp.autocast("cuda"):
            logit = model(image)

            if args.is_synth_train and args.is_pooled_fewshot:
                if (is_real==1).any():
                    loss_real = criterion(logit[is_real == 1], label[is_real == 1])
                else:
                    loss_real = torch.tensor(0.0).cuda()

                if (is_real==0).any():
                    loss_gen = criterion(logit[is_real == 0], label[is_real == 0])
                else:
                    loss_gen = torch.tensor(0.0).cuda()

                loss = args.lambda_1 * loss_real + (1 - args.lambda_1) * loss_gen
            elif args.is_synth_train and (not args.is_pooled_fewshot):
                # Synthetic-only: treat all samples as synthetic.
                loss_gen = criterion(logit, label)
                loss = loss_gen
                loss_real = torch.tensor(0.0).cuda()
            else:
                loss = criterion(logit, label)
                loss_real = loss
                loss_gen = torch.tensor(0.0).cuda()

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        # === ADDITIONAL REGULARIZATION USING TS ===
        regu_start_time = time.time()
        if args.is_synth_train and args.num_iters > 0:


            g = args.n_img_per_cls * args.n_classes # a trick to keep this one fix. Only difference in the paper with a constant
            total_discrepancy_loss = 0.0
            total_robustness_loss = 0.0
            if args.is_pooled_fewshot: 
                # region_real_in_batch and region_syn_in_batch store *batch indices* grouped by region
                region_real_in_batch = [[] for _ in range(int(args.num_centroids))]
                region_syn_in_batch  = [[] for _ in range(int(args.num_centroids))]

                for b_idx, g_idx in enumerate(global_idx):
                    region_idx, is_real = global_idx_to_region[g_idx.item()]
                    if is_real:
                        region_real_in_batch[region_idx].append(b_idx)
                    else:
                        region_syn_in_batch[region_idx].append(b_idx)
                total_discrepancy_loss = 0.0
                total_robustness_loss  = 0.0

                for region_idx in range(int(args.num_centroids)):
                    real_indices = region_real_in_batch[region_idx]  # batch indices for real
                    syn_indices  = region_syn_in_batch[region_idx]   # batch indices for synthetic
                    num_real_in_region = len(real_indices)
                    num_syn_in_region  = len(syn_indices)

                    # (a) Discrepancy: real vs. synthetic
                    if num_real_in_region > 0 and num_syn_in_region > 0:
                        pred_real = logit[real_indices]  # shape: (N_r, dim)
                        pred_syn  = logit[syn_indices]   # shape: (N_s, dim)

                        pairwise_mse = F.mse_loss(
                            pred_real.unsqueeze(1),
                            pred_syn.unsqueeze(0),
                            reduction='none'
                        )  # shape: (N_r, N_s, dim)
                        pairwise_mse = pairwise_mse.mean(dim=-1)  # shape: (N_r, N_s)
                        discrepancy_loss = pairwise_mse.sum() * (1.0 / (g * num_real_in_region))
                        total_discrepancy_loss += discrepancy_loss

                    # (b) Robustness: among synthetic in this region (pairwise among syn_indices)
                    if num_syn_in_region > 1 and num_real_in_region > 0:
                        pred_syn_region = logit[syn_indices]  # shape: (N_s, dim)
                        pairwise_mse_syn = F.mse_loss(
                            pred_syn_region.unsqueeze(1),
                            pred_syn_region.unsqueeze(0),
                            reduction='none'
                        )  # shape: (N_s, N_s, dim)
                        pairwise_mse_syn = pairwise_mse_syn.mean(dim=-1)  # shape: (N_s, N_s)
                        # We'll sum only the upper triangle to avoid double-counting i-j and j-i
                        i_upper = torch.triu_indices(num_syn_in_region, num_syn_in_region, offset=1)
                        mse_upper = pairwise_mse_syn[i_upper[0], i_upper[1]]  # shape: (num_upper,)
                        # Weighted by (1/g)*(1/num_syn_in_region)
                        robustness_loss = mse_upper.sum() * (1.0 / g) * (1.0 / num_syn_in_region)
                        total_robustness_loss += robustness_loss

                total_loss = args.ce_real * loss_real + args.ce_syn * loss_gen \
                            + args.lam_dis * total_discrepancy_loss \
                            + args.lam_rob * total_robustness_loss
            else:
                # Synthetic-only: all samples are synthetic, so only compute robustness loss.
                region_syn_in_batch = [[] for _ in range(int(args.num_centroids))]
                for b_idx, g_idx in enumerate(global_idx):
                    region_idx, flag = global_idx_to_region[g_idx.item()]
                    region_syn_in_batch[region_idx].append(b_idx)
                for region_idx in range(int(args.num_centroids)):
                    num_syn = len(region_syn_in_batch[region_idx])
                    if num_syn > 1:
                        pred_syn_region = logit[region_syn_in_batch[region_idx]]
                        pairwise_mse_syn = F.mse_loss(pred_syn_region.unsqueeze(1), pred_syn_region.unsqueeze(0), reduction='none').mean(dim=-1)
                        i_upper = torch.triu_indices(num_syn, num_syn, offset=1)
                        mse_upper = pairwise_mse_syn[i_upper[0], i_upper[1]]
                        robustness_loss = mse_upper.sum() * (1.0 / (g * num_syn))
                        total_robustness_loss += robustness_loss
                total_loss = args.ce_syn * loss_gen + args.lam_rob * total_robustness_loss
        else:
            total_loss = loss
        regu_end_time=time.time()
        total_regu_time += regu_end_time - regu_start_time        
        optimizer.zero_grad()
        if fp16_scaler is None:
            total_loss.backward()
            optimizer.step()
        else:
            fp16_scaler.scale(total_loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        with torch.no_grad():
            acc1, acc5 = get_accuracy(logit.detach(), label_origin, topk=(1, 5))
            metric_logger.update(top1=acc1.item())
            metric_logger.update(loss=total_loss.item())
            if args.is_synth_train and args.is_pooled_fewshot and args.num_iters > 0:
                metric_logger.update(total_discrepancy_loss=total_discrepancy_loss)
                metric_logger.update(total_robustness_loss=total_robustness_loss)
            elif args.is_synth_train and (not args.is_pooled_fewshot) and args.num_iters > 0:
                metric_logger.update(total_robustness_loss=total_robustness_loss)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        if scheduler is not None:
            scheduler.step()

    metric_logger.synchronize_between_processes()
    print("Averaged train stats:", metric_logger)
    print("Regutime", total_regu_time)

    return {"train/{}".format(k): meter.global_avg for k, meter in metric_logger.meters.items()}, best_stats, best_top1


def main(args):
    args.n_classes = len(SUBSET_NAMES[args.dataset])
    os.makedirs(args.output_dir, exist_ok=True)

    fix_random_seeds(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True

    ###########################################################################
    # 1) Check for cached cluster file
    ###########################################################################
    if args.is_synth_train == True and args.num_iters > 0:
        print("In cluster")
        cluster_cache_path = os.path.join("", args.dataset, "lora", args.fewshot_seed, str(args.n_img_per_cls),str(args.n_shot) + "shot", str(args.num_centroids), "cluster_results.pkl")
        os.makedirs(os.path.dirname(cluster_cache_path), exist_ok=True)

        # If the file exists, load clustering results. Otherwise, run clustering.
        if os.path.isfile(cluster_cache_path):
            print(f"[INFO] Loading FAISS clustering results from {cluster_cache_path}")
            with open(cluster_cache_path, "rb") as f:
                cached_data = pickle.load(f)
            if args.dataset =='imagenet':
                TS = cached_data
            else:
                TS = cached_data["TS"]
                real_image_paths = cached_data["real_image_paths"]
                synthetic_image_paths = cached_data["synthetic_image_paths"]
        else:
            print("[INFO] Clustering results not found. Running FAISS clustering...")
            if not hasattr(args, 'num_centroids'):
                args.num_centroids = 20
            if args.dataset == 'imagenet':
                TS = cluster_images_with_faiss_imagenet(
                args.real_train_data_dir,
                args.synth_train_data_dir,
                num_centroids=int(args.num_centroids),
                num_image_per_subfolder=int(args.n_img_per_cls),
                niter=args.num_iters,
                verbose=True,
                )

                print(f"[INFO] Done. TS saved to {cluster_cache_path}")
                with open("./faiss_memmap/cluster_results.pkl", "wb") as f:
                    pickle.dump(TS, f)
            else:
                TS, real_image_paths, synthetic_image_paths = cluster_images_with_faiss(
                    args.real_train_data_dir,
                    args.synth_train_data_dir,
                    num_centroids=int(args.num_centroids),
                    num_shot = int(args.n_shot),
                    num_image_per_subfolder=int(args.n_img_per_cls),
                    niter=args.num_iters,
                    verbose=True,
                )

                # Save clustering results so we can load them in future runs
                print(f"[INFO] Saving FAISS clustering results to {cluster_cache_path}")
                with open(cluster_cache_path, "wb") as f:
                    pickle.dump({
                        "TS": TS,
                        "real_image_paths": real_image_paths,
                        "synthetic_image_paths": synthetic_image_paths
                    }, f)
        ###########################################################################


        # Create a dict to map each global index -> (region_index, is_real)
        global_idx_to_region = {}  # key: global_id, value: (region_index, bool_is_real)

        for region_index, (real_list, syn_list) in TS.items():
            # Convert real_list, syn_list to sets or just iterate as is
            for r_id in real_list:
                global_idx_to_region[r_id] = (region_index, True)  # (region_idx, is_real=True)
            for s_id in syn_list:
                global_idx_to_region[s_id] = (region_index, False) # (region_idx, is_real=False)
    # Data loader
    train_loader, val_loader = load_data_loader(args)
    if args.is_synth_train:
        train_loader = load_synth_train_data_loader(args)

    # Model and optimizer
    if args.model_type == "clip":
        model = CLIP(
            dataset=args.dataset,
            is_lora_image=args.is_lora_image,
            is_lora_text=args.is_lora_text,
            clip_download_dir=args.clip_download_dir,
            clip_version=args.clip_version,
        )
        params_groups = model.learnable_params()
    elif args.model_type == "resnet50": 
        model = ResNet50(n_classes=args.n_classes)
        params_groups = model.parameters()
    #model = torch.nn.DataParallel(model)

    model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    # CutMix and MixUp augmentation
    if args.is_mix_aug:
        cutmix = v2.CutMix(num_classes=args.n_classes)
        mixup = v2.MixUp(num_classes=args.n_classes)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    else:
        cutmix_or_mixup = None

    scheduler = None
    optimizer = torch.optim.AdamW(
        params_groups, lr=args.lr, weight_decay=args.wd,
    )
    args.lr_schedule = cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
        start_warmup_value=args.min_lr,
    )

    fp16_scaler = None
    if args.use_fp16:
        # mixed precision training
        fp16_scaler = torch.amp.GradScaler("cuda")
    # Logging setup
    if args.log == 'wandb':
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        _ = os.system('wandb login {}'.format(args.wandb_key))
        os.environ['WANDB_API_KEY'] = args.wandb_key
        wandb.init(
            dir = '', project= "datadream",entity = "",
            reinit= True, settings = wandb.Settings(_disable_stats = True, _disable_meta = True),  
            group=args.wandb_group, 
            name=args.wandb_group,
            config=vars(args)
        )
        args.wandb_url = wandb.run.get_url()
    elif args.log == 'tensorboard':
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = os.path.join(args.output_dir, "tb-{}".format(args.local_rank))
        Path(tb_dir).mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(tb_dir, flush_secs=30)
    print("=> Training starts ...")
    start_time = time.time()

    best_stats = {}
    best_top1 = 0.

    for epoch in range(0, args.epochs):
        train_stats, best_stats, best_top1 = train_one_epoch(
            model, criterion, train_loader, optimizer, scheduler, epoch, fp16_scaler, cutmix_or_mixup, args,
            val_loader, best_stats, best_top1, global_idx_to_region
        )

        test_stats = eval(
            model, criterion, val_loader, epoch, fp16_scaler, args)

        if test_stats["test/top1"] > best_top1:
            best_top1 = test_stats["test/top1"]
            best_stats = test_stats
            save_model(args, model, optimizer, epoch, fp16_scaler, "best_checkpoint.pth")

        if epoch + 1 == args.epochs:
            test_stats['test/best_top1'] = best_stats["test/top1"]
            test_stats['test/best_loss'] = best_stats["test/loss"]

        if args.log == 'wandb':
            train_stats.update({"epoch": epoch})
            wandb.log(train_stats)
            wandb.log(test_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as e:
        print(traceback.format_exc())
