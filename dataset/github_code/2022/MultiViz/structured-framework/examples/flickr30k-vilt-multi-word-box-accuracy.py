import sys
import os

import json

sys.path.insert(1, os.getcwd())
from datasets.flickr30k import Flickr30kDataset
from models.flickr30k_vilt import Flickr30KVilt
from transformers import ViltProcessor
import torch.nn.functional as F
from visualizations.visualizegradient import *
import random
import copy

import matplotlib.pyplot as plt
import matplotlib.patches as patches

random.seed(42)
# get the dataset
data = Flickr30kDataset("valid")
# set target sentence idx
target_idx = 0

# get the model
analysismodel = Flickr30KVilt(target_idx=target_idx, device="cuda")

# unimodal image gradient
"""
for instance_idx in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
    instance = data.getdata(instance_idx)

    # get the model predictions
    preds = analysismodel.forward(instance)

    # compute and print grad saliency with and without multiply orig:
    saliency = get_saliency_map(instance, analysismodel, 0)
    grads = saliency[0]

    t = normalize255(torch.sum(torch.abs(grads), dim=0), fac=255)
    heatmap2d(
        t,
        f"visuals/flickr30k-vilt-{instance_idx}-{target_idx}-saliency.png",
        instance[0],
    )
"""
from misc.flickr30k_vilt_target_ids import *

id_to_tids = {
    50: instance_text_target_ids_50,
    100: instance_text_target_ids_100,
    150: instance_text_target_ids_150,
    200: instance_text_target_ids_200,
    500: instance_text_target_ids_500,
    250: instance_text_target_ids_250,
    300: instance_text_target_ids_300,
    400: instance_text_target_ids_400,
    600: instance_text_target_ids_600,
    700: instance_text_target_ids_700,
    800: instance_text_target_ids_800,
    900: instance_text_target_ids_900,
    1000: instance_text_target_ids_1000,
    350: instance_text_target_ids_350,
    450: instance_text_target_ids_450,
    550: instance_text_target_ids_550,
    650: instance_text_target_ids_650,
    750: instance_text_target_ids_750,
    850: instance_text_target_ids_850,
    950: instance_text_target_ids_950,
}


class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def find_top_k_bounding_boxes(id_to_boxes, pixel_grads, num_gt_boxes):
    top_k_box_ids = []

    box_id_to_avg_grad = {}
    for box_id, boxes in id_to_boxes.items():
        box_id_means = []
        for box_iter in boxes:
            x1, y1, x2, y2 = box_iter
            box_id_means.append(np.mean(pixel_grads[y1:y2, x1:x2]))
        box_id_to_avg_grad[box_id] = np.mean(box_id_means)

    # Sort dictionary box_id_to_avg_grad by value
    sorted_box_id_to_avg_grad = sorted(
        box_id_to_avg_grad.items(), key=lambda kv: kv[1], reverse=True
    )
    for i in range(num_gt_boxes):
        top_k_box_ids.append(sorted_box_id_to_avg_grad[i][0])
    return top_k_box_ids


for instance_idx, tid_dict in id_to_tids.items():
    key_to_logits = {}
    key_to_logits[str(instance_idx)] = {}

    # Get the Instance
    instance = data.getdata(instance_idx)
    (
        first_sentence,
        id_to_boxes,
        id_to_phrase,
    ) = data.get_entities_data_first_sentence(instance_idx)
    # print(id_to_boxes, id_to_phrase)

    # Get Original Logits
    original_probs, _ = analysismodel.forward(instance)
    original_logits = original_probs.detach().cpu().numpy()[0]
    

    for key, value in tid_dict.items():
        key_to_logits[str(instance_idx)][key] = {}
        key_to_logits[str(instance_idx)][key]["original_logits"] = original_logits

        # Calculate the Double Grad
        # print(instance_idx)
        processor = ViltProcessor.from_pretrained(
            "dandelin/vilt-b32-finetuned-flickr30k"
        )

        grads, di, tids = analysismodel.getdoublegrad(
            instance, instance[-1], value["ids"]
        )

        # print(
        #     dict(
        #         enumerate(
        #             processor.tokenizer.convert_ids_to_tokens(
        #                 tids[0].detach().cpu().numpy()
        #             )
        #         )
        #     )
        # )

        grads = grads[0]

        # Save the Double Grad Image
        normalized_grads_dg = normalize255(torch.sum(torch.abs(grads), dim=0), fac=255)

        heatmap2d(
            normalized_grads_dg,
            f"visuals/flickr30k-vilt-{key}-doublegrad.png",
            instance[0],
        )

        # Get the New Text
        new_tids = tids[0].detach().cpu().numpy().tolist()
        new_tids = new_tids[: value["ids"][0]] + new_tids[value["ids"][-1] + 1 :]
        sep_index = new_tids.index(processor.tokenizer.sep_token_id)

        new_text = processor.tokenizer.decode(new_tids[1:sep_index])

        # Save new text in a file
        with open(f"visuals/flickr30k-vilt-{key}-new_text.txt", "w") as f:
            f.write(new_text)

        # Load and resize original image
        normalized_grads = (
            normalize255(torch.sum(torch.abs(grads), dim=0), fac=255)
            .detach()
            .cpu()
            .numpy()
        )
        img = cv2.resize(
            np.asarray(Image.open(instance[0])),
            (normalized_grads.shape[1], normalized_grads.shape[0]),
        )

        gt_img = copy.deepcopy(img)
        random_box_img = copy.deepcopy(img)
        new_box_img = copy.deepcopy(img)

        # Ground Truth Box Drop
        # drop ground truth based on Flickr30k Entities
        # Find double grad text object

        boxes_to_drop = []
        for idx, phrase in id_to_phrase.items():
            # Check if there is an intersection between value["text"] and phrase
            phr = " " + phrase.lower() + " "
            val = " " + value["text"].lower() + " "
            if val in phr or phr in val:
                boxes_to_drop.append(idx)

        # drop boxes in image
        num_gt_boxes = 0
        gt_box_ids = []
        mask = np.zeros(gt_img.shape[:-1])
        for box_id in boxes_to_drop:
            if box_id in id_to_boxes:
                for box_iter in id_to_boxes[box_id]:
                    x1, y1, x2, y2 = box_iter
                    gt_img[y1:y2, x1:x2] = 0
                    mask[y1:y2, x1:x2] = 1
                gt_box_ids.append(box_id)
                num_gt_boxes += 1
            else:
                print("Couldn't find box with box_id: ", box_id)
        gt_img_path = f"visuals/flickr30k-vilt-{key}-gt_img.jpg"

        plt.imsave(gt_img_path, gt_img)

        new_instance = (gt_img_path, [new_text])

        new_probs, _ = analysismodel.forward(new_instance)
        new_logits = new_probs.detach().cpu().numpy()[0]
        key_to_logits[str(instance_idx)][key]["ground_truth_logits"] = new_logits

        # Find matching boxes in img
        dg_box_ids = find_top_k_bounding_boxes(
            id_to_boxes, normalized_grads, num_gt_boxes
        )
        for box_id in dg_box_ids:
            if box_id in id_to_boxes:
                for box_iter in id_to_boxes[box_id]:
                    x1, y1, x2, y2 = box_iter
                    new_box_img[y1:y2, x1:x2] = 0

        new_box_img_path = f"visuals/flickr30k-vilt-{key}-new_box_img.jpg"
        plt.imsave(new_box_img_path, new_box_img)

        new_box_img_unmasked = copy.deepcopy(img)

        # Create figure and axes
        fig, ax = plt.subplots()

        # Display the image
        ax.imshow(new_box_img_unmasked)

        for box_id in dg_box_ids:
            if box_id in id_to_boxes:
                for box_iter in id_to_boxes[box_id]:
                    x1, y1, x2, y2 = box_iter
                    rect = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=1,
                        edgecolor="r",
                        facecolor="none",
                    )
                    ax.add_patch(rect)

        # Save the figure
        plt.axis('off')
        plt.savefig(f"visuals/flickr30k-vilt-{key}-new_box_img_with_boxes.jpg")
        plt.close()

        print("Key: ", key)
        print(gt_box_ids, dg_box_ids)
        new_instance = (new_box_img_path, [new_text])

        new_probs, _ = analysismodel.forward(new_instance)
        new_logits = new_probs.detach().cpu().numpy()[0]
        key_to_logits[str(instance_idx)][key]["doublegrad_box_logits"] = new_logits

        # Randomly select num_gt_boxes from the ids and drop them

        random_box_ids = random.sample(id_to_boxes.keys(), num_gt_boxes)

        for box_id in random_box_ids:
            if box_id in id_to_boxes:
                for box_iter in id_to_boxes[box_id]:
                    x1, y1, x2, y2 = box_iter
                    random_box_img[y1:y2, x1:x2] = 0

        random_box_img_path = f"visuals/flickr30k-vilt-{key}-random_box_img.jpg"
        plt.imsave(random_box_img_path, random_box_img)

        new_instance = (random_box_img_path, [new_text])

        new_probs, _ = analysismodel.forward(new_instance)
        new_logits = new_probs.detach().cpu().numpy()[0]
        key_to_logits[str(instance_idx)][key]["random_box_logits"] = new_logits

        num_dg_matching_boxes = len([x for x in gt_box_ids if x in dg_box_ids])
        num_random_matching_boxes = len([x for x in gt_box_ids if x in random_box_ids])
        key_to_logits[str(instance_idx)][key][
            "num_dg_matching_boxes"
        ] = num_dg_matching_boxes
        key_to_logits[str(instance_idx)][key][
            "num_random_matching_boxes"
        ] = num_random_matching_boxes
        key_to_logits[str(instance_idx)][key]["num_gt_boxes"] = num_gt_boxes

    # Write key_to_logits to JSON file
    with open(f"key_to_logits-box-acc-{instance_idx}.json", "w") as f:
        json.dump(key_to_logits, f, cls=NumpyFloatValuesEncoder)
