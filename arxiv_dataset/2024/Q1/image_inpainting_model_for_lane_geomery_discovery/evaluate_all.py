import cv2
from PIL import Image
import glob
import os
import numpy as np
from sklearn.metrics import confusion_matrix


def get_ccq_metrics(x, y):
    """Returns the following metrics of a given prediction and label.
    The metrics are computed as follows:
    - Correctness  : TP / (TP + FP)
    - Completeness : TP / (TP + FN)
    - Quality      : TP / (TP + FP + FN)
    -

    Args:
        x (np.array): 2D binary numpy prediction array.
        y (np.array): 2D binary numpy ground truth array.

    Returns:
        corr (float): Correctness metric
        comp (float): Completeness metric
        qual (float): Quality metric
    """
    #from sklearn.metrics import confusion_matrix

    # Input arrays must have the same label value
    assert np.all(np.unique(x) == np.unique(y))
    tn, fp, fn, tp = confusion_matrix(y.ravel(), x.ravel()).ravel()

    corr = tp / (tp + fp)
    comp = tp / (tp + fn)
    qual = tp / (tp + fp + fn)

    return corr, comp, qual


# for all images in folder, calculate the metric
all_correctness = []
all_completeness = []
all_quality = []
image_path = '/home/shong/mass_data/val_out_oct_16/' #'/Users/soojunghong/PycharmProjects/inpainting_metric/test_images/'

for f in glob.glob(image_path + "*.png"):
        fname = os.path.basename(f)
        print('current file :', fname)
        img = cv2.imread(image_path+fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        #(thresh, blackAndWhiteImage) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        (thresh, blackAndWhiteImage) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img = blackAndWhiteImage

        # TODO : correct the pixel
        label_img = img[:, :256]
        mask_img = img[:, 256:512]  # 0, 255 (also works with 0, 1)
        pred_img = img[:, 512:]  # 0, 255 (thresholded prediction)

        corr, comp, qual = get_ccq_metrics(pred_img, label_img)  # TODO : correct the image name
        print('correctness/completeness/quality : ', fname, corr, comp, qual)
        all_correctness.append(corr)
        all_completeness.append(comp)
        all_quality.append(qual)


avg_correctness = np.mean(all_correctness)
avg_completeness = np.mean(all_completeness)
avg_quality = np.mean(all_quality)
print('>>>>> avg correctness/completeness/quality : ', avg_correctness, avg_completeness, avg_quality)
