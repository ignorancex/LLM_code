import os
import cv2
import json
import numpy as np
import random
import albumentations as A
from glob import glob
from copy import deepcopy
from PIL import Image, ImageDraw
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


def extract_polygons(annotations, img_id):
    list_poly = []
    num_points = []
    list_annos = []
    tmp_annotations = deepcopy(annotations)
    for ann in tmp_annotations:
        image_id = ann['image_id']
        if image_id == img_id:
            sub_num_points = []
            list_points = ann['segmentation']
            for points in list_points:
                points = [(int(points[2*i]), int(points[2*i+1]))
                          for i in range(len(points)//2)]

                sub_num_points.append(len(points))
                list_poly.extend(points)
            num_points.append(sub_num_points)
            list_annos.append(deepcopy(ann))
    return list_poly, num_points, list_annos


def keypoint2polygons(keypoints, num_points):
    start_idx = 0
    list_polys = []
    for sub_points in num_points:
        sub_poly = []
        for points in sub_points:
            sub_poly.append(keypoints[start_idx:start_idx+points])
            start_idx += points

        list_polys.append(sub_poly)

    return list_polys


def visualize_polygons(img, list_polys, colors: list):
    for i, polys in enumerate(list_polys):
        for poly in polys:
            poly_np = np.array(poly)
            img = cv2.polylines(img, np.int32(
                [poly_np]), True, colors[i % len(colors)], 2)

    return img


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    plt.show()


def get_iou(bb1, bb2):

    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


def get_confusion_matrix(gt, pred_list):
    pred_label = []
    gt_label = []

    for pred_ann in pred_list:
        pred_id = pred_ann['image_id']
        for gt_ann in gt['annotations']:
            gt_id = gt_ann['image_id']
            if pred_id == gt_id:

                pred_box = list(map(int, pred_ann['bbox']))
                gt_box = list(map(int, gt_ann['bbox']))
                x1, y1, w, h = gt_box
                x2 = x1+w
                y2 = y1+h

                iou = get_iou(pred_box, [x1, y1, x2, y2])
                if iou > 0.5:
                    pred_label.append(pred_ann['category_id'])
                    gt_label.append(gt_ann['category_id'])

    return confusion_matrix(gt_label, pred_label)


def get_miss_classify(cm: np.array, list_names: list) -> list:
    miscls = []
    for i in range(cm.shape[0]):
        idx = -1
        best = -1
        for j in range(cm.shape[1]):
            if i != j and cm[i][j] > best:
                best = cm[i][j]
                idx = j
        if cm[i][idx] > 10:
            miscls.append(set([list_names[i], list_names[idx]]))
    miscls = np.unique(np.array(miscls))
    return miscls


def get_miss_classify_plus(cm: np.array, list_names: list) -> list:
    miscls = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i][j] > 10:
                miscls.append(set([list_names[i], list_names[j]]))

    miscls = np.unique(np.array(miscls))
    return miscls


def get_class_samples(list_images: str, class_name: str, num_samples: int) -> list:
    sub_images = list(filter(lambda s: s['file_name'].split(
        '/')[-2] == class_name, list_images))
    if num_samples > len(sub_images):
        num_samples = len(sub_images)
    return random.sample(sub_images, num_samples)


def polygon2segmentation(list_polygon: list) -> list:
    list_segment = []
    for poly in list_polygon:
        segment = []
        [segment.extend(list(point)) for point in poly]
        list_segment.append(segment)
    return list_segment


def update_properties(ann: dict, img_shape: tuple, polygon: list, image_id: int) -> dict:
    mask_img = Image.new('L', img_shape, 0)
    ann['segmentation'] = polygon2segmentation(polygon)
    tmp_poly_list = []
    [tmp_poly_list.append((int(point[0]), int(point[1])))
     for point in polygon[0]]

    ImageDraw.Draw(mask_img).polygon(tmp_poly_list, outline=1, fill=1)
    ann['area'] = int(np.sum(np.array(mask_img)))
    x1, x2 = np.min(np.array(tmp_poly_list)[:, 0]), np.max(
        np.array(tmp_poly_list)[:, 0])
    y1, y2 = np.min(np.array(tmp_poly_list)[:, 1]), np.max(
        np.array(tmp_poly_list)[:, 1])
    ann['bbox'] = list(map(int, [x1, y1, x2-x1, y2-y1]))
    ann['image_id'] = image_id
    return ann


def mixup_image(img1: dict, img2: dict, new_image_id: int, save_root: str, train_anno: dict):
    list_anno = []
    image_anno = dict()

    poly1, num1, ann1 = extract_polygons(train_anno['annotations'], img1['id'])
    poly2, num2, ann2 = extract_polygons(train_anno['annotations'], img2['id'])
    w1, h1 = img1['width'], img1['height']
    w2, h2 = img2['width'], img2['height']
    W, H = int((w1+w2)/2), int((h1+h2)/2)

    image_anno['width'] = W
    image_anno['height'] = H
    image_anno['id'] = new_image_id
    image_anno['file_name'] = os.path.join(save_root, img1['file_name'].split(
        '/')[-1].split('.')[0] + '&' + img2['file_name'].split('/')[-1].split('.')[0] + '.jpg')

    transform = A.Compose([
        A.Resize(height=H, width=W)
    ], keypoint_params=A.KeypointParams(format='xy'))

    image1 = cv2.imread(img1['file_name'])
    image2 = cv2.imread(img2['file_name'])

    transformed1 = transform(image=image1, keypoints=poly1)
    transformed_image1 = transformed1['image']
    transformed_keypoints1 = transformed1['keypoints']

    transformed2 = transform(image=image2, keypoints=poly2)
    transformed_image2 = transformed2['image']
    transformed_keypoints2 = transformed2['keypoints']

    transformed_img = transformed_image1*0.5 + transformed_image2*0.5

    transformed_poly1 = keypoint2polygons(transformed_keypoints1, num1)
    transformed_poly2 = keypoint2polygons(transformed_keypoints2, num2)

    for i in range(len(ann1)):
        ann1[i] = update_properties(
            ann1[i], (W, H), transformed_poly1[i], new_image_id)
        list_anno.append(ann1[i])
    for i in range(len(ann2)):
        ann2[i] = update_properties(
            ann2[i], (W, H), transformed_poly2[i], new_image_id)
        list_anno.append(ann2[i])

    return transformed_img, transformed_poly1 + transformed_poly2, image_anno, list_anno
