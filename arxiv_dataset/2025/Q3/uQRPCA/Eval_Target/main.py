# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import sys

# === Path Settings ===
path_gt = 'C:/Users/86138/Desktop/target_gt/overpass/'   # Ground truth binary masks
path_fg = 'C:/Users/86138/Desktop/overpass'              # Predicted target masks (to be evaluated)

# === File Renaming Function ===
def rename_files(path, prefix):
    """
    Rename all files in the given path using a consistent prefix and numeric index.
    Format: prefix_0000.jpg, prefix_0001.jpg, ...
    """
    files = [f for f in listdir(path) if isfile(join(path, f))]
    files.sort()
    for i, file in enumerate(files):
        new_name = f"{prefix}_{i:04d}.jpg"  # JPG and PNG supported
        os.rename(join(path, file), join(path, new_name))
    return [f for f in listdir(path) if isfile(join(path, f))]

# Rename ground truth files (comment out if already done)
files_gt = rename_files(path_gt, 'gt')
# files_fg = rename_files(path_fg, 'fg')  # Optional renaming for target files

# Get list of files in both GT and predicted folders
files_gt = [f for f in listdir(path_gt) if isfile(join(path_gt, f))]
files_fg = [f for f in listdir(path_fg) if isfile(join(path_fg, f))]

# Sort file names for consistent processing
files_gt.sort()
files_fg.sort()

# === Initialize Evaluation Metrics ===
TP = .0   # True Positive pixels
FP = .0   # False Positive pixels
TN = .0   # True Negative pixels
FN = .0   # False Negative pixels
Recall = .0
Precision = .0
Fscore = .0

# === Define Colors for Visualization ===
green = [0, 255, 0]       # FN (missed foreground)
blue  = [255, 0, 0]
red   = [0, 0, 255]       # FP (false detection)
white = [255, 255, 255]   # TP (correct detection)
black = [0, 0, 0]         # TN (correct background)

def extract_number(filename):
    """Extract numeric index from filename."""
    return int(''.join(filter(str.isdigit, filename)))

print('Processing')
k = 1
for file_gt in files_gt:
    # Extract index and find corresponding predicted mask (offset by +1 if needed)
    num_gt = extract_number(file_gt)
    matched_fg_files = [f for f in files_fg if extract_number(f) == num_gt + 1]

    if not matched_fg_files:
        print(f"No matching FG file found for {file_gt}")
        continue

    file_fg = matched_fg_files[0]

    print(k, file_gt, file_fg)
    img_gt = cv2.imread(join(path_gt, file_gt), cv2.IMREAD_GRAYSCALE)
    img_fg = cv2.imread(join(path_fg, file_fg), cv2.IMREAD_GRAYSCALE)

    # Resize predicted mask to match ground truth size
    rows, cols = img_gt.shape
    img_fg = cv2.resize(img_fg, (cols, rows))
    img_res = np.zeros((rows, cols, 3), np.uint8)

    # Pixel-wise classification
    for i in xrange(rows):
        for j in xrange(cols):
            pixel_gt = img_gt[i, j]
            pixel_fg = img_fg[i, j]
            if pixel_gt == 255 and pixel_fg == 255:
                TP += 1
                img_res[i, j] = white
            elif pixel_gt == 0 and pixel_fg == 255:
                FP += 1
                img_res[i, j] = red
            elif pixel_gt == 0 and pixel_fg == 0:
                TN += 1
                img_res[i, j] = black
            elif pixel_gt == 255 and pixel_fg == 0:
                FN += 1
                img_res[i, j] = green

    # Visualization (optional)
    cv2.imshow('GT', img_gt)
    cv2.imshow('FG', img_fg)
    cv2.imshow('SC', img_res)
    # cv2.imwrite(join(path_sc, file_gt), img_res)  # Save visual result if needed
    cv2.waitKey(1)
    k += 1
    # break  # Uncomment to run only the first image

cv2.destroyAllWindows()

# === Final Score Computation ===
Recall    = TP / (TP + FN) if TP + FN != 0 else 0
Precision = TP / (TP + FP) if TP + FP != 0 else 0
Fscore    = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) != 0 else 0

# === Print Final Evaluation Results ===
print('Score:')
print('TP: ', TP)
print('FP: ', FP)
print('TN: ', TN)
print('FN: ', FN)
print('Recall: ', Recall)
print('Precision: ', Precision)
print('Fscore: ', Fscore)
