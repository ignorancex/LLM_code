import imp
import os
import sys
sys.path.append("./")
import cv2

from utils.util_funcs import draw_dilation_monusac

import numpy as np

def pannuke_move(split="train"):
    file_names_hover = os.listdir(f"/mnt/group-ai-medical-SHARD/private/wenhuazhang/data/nuclei-level-multi-class/hovernet/pannuker/{split}/256x256_120x120")

    images = []
    annos = []
    for file_name in file_names_hover:
        file_path = f"/mnt/group-ai-medical-SHARD/private/wenhuazhang/data/nuclei-level-multi-class/hovernet/pannuker/{split}/256x256_120x120/{file_name}"
        anno = np.load(file_path)

        images.append(anno[:, :, :3])
        annos.append(anno[:, :, 3:])
    
    images = np.array(images, dtype=np.uint8)
    annos = np.array(annos)
    os.makedirs("data_pannuke", exist_ok=True)
    with open(f"data_pannuke/images_{split}.npy", "wb") as f:
        np.save(f, images)
    
    with open(f"data_pannuke/labels_{split}.npy", "wb") as f:
        np.save(f, annos)
    
def monusac_move(split="train"):
    file_names_hover = os.listdir(f"/mnt/group-ai-medical-SHARD/private/wenhuazhang/data/nuclei-level-multi-class/hovernet/monusac/{split}/256x256_120x120")

    images = []
    annos = []
    for file_name in file_names_hover:
        file_path = f"/mnt/group-ai-medical-SHARD/private/wenhuazhang/data/nuclei-level-multi-class/hovernet/monusac/{split}/256x256_120x120/{file_name}"
        anno = np.load(file_path)

        images.append(anno[:, :, :3])
        annos.append(anno[:, :, 3:])
    
    images = np.array(images, dtype=np.uint8)
    annos = np.array(annos)
    os.makedirs("data_monusac", exist_ok=True)
    with open(f"data_monusac/images_{split}.npy", "wb") as f:
        np.save(f, images)
    
    with open(f"data_monusac/labels_{split}.npy", "wb") as f:
        np.save(f, annos)

def conic_move():
    labels_path = "data/labels.npy"
    annos = np.load(labels_path)
    print("annos shape: ", annos.shape)

    print("anno max min 0: ", np.max(annos[0, :, :, 0]), np.min(annos[0, :, :, 0]))
    print("anno max min 1: ", np.max(annos[0, :, :, 1]), np.min(annos[0, :, :, 1]))
    

if __name__ == "__main__":
    # pannuke_move("train")
    # pannuke_move("test")
    monusac_move("train")
    monusac_move("test")
    # conic_move()