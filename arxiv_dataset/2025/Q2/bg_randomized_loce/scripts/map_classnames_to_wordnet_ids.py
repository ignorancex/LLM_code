# Get voc to imagenet mapping
from collections import defaultdict
from pprint import pprint

from robustness.tools.imagenet_helpers import ImageNetHierarchy

from bg_randomized_loce.data_structures.mscoco import MSCOCOSegmentationDataset

if __name__ == "__main__":

    ## MSCOCO
    classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
     'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
     'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
     'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
     'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
     'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
     'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
     'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    ## PASCAL VOC
    classes = MSCOCOSegmentationDataset.ALL_CAT_NAMES_BY_ID.values()

    imagenet_hierarchy = ImageNetHierarchy('/workspace/data/data-remote/imagenet/ILSVRC/Data/CLS-LOC', '/workspace/data/data-remote/imagenet')

    ## Get the wordnet superclass associated with each label
    class_to_wordnet_superclass_ids: dict[str, set] = defaultdict(set)
    for cls in classes:

        # manually set search terms where needed
        classes_to_find = [cls]
        if cls == "cow": classes_to_find = ["cow", "bovine", "buffalo", "bison"]
        if cls == "giraffe": classes_to_find = ["giraffe", "giraffa"]
        if cls == "mouse": classes_to_find = ["computer mouse"]
        if cls == "keyboard": classes_to_find = ["keyboard", "keyboard instrument", "computer keyboard"]
        if cls == "bed": classes_to_find = ["bed", "baby bed"]
        if cls == "sheep": classes_to_find = ["sheep", "wild sheep"]
        if cls == "boat": classes_to_find = ["boat", "ship"]
        if cls == "book": classes_to_find = []  # not represented :-/
        if cls == "pottedplant": classes_to_find = ["potted plant", "plant", "flowerpot"]
        if cls == "tvmonitor": classes_to_find = ["television", "monitor"]
        if cls == "diningtable": classes_to_find = ["dining table"]
        if cls == "aeroplane": classes_to_find = ["airplane", "aeroplane"]

        for class_name in classes_to_find:
            for cnt, (wnid, ndesc_in, ndesc_total) in enumerate(imagenet_hierarchy.wnid_sorted):
                desc =  imagenet_hierarchy.wnid_to_name[wnid]


                # Skip manually found exceptions
                if (class_name == "cow") and ("sea cow" in desc): continue
                if (class_name == "truck") and ("garden truck" in desc): continue
                if (class_name == "sheep") and any((s in desc for s in ["sheep dog", "sheepdog", "wild sheep", "mountain sheep"])): continue
                if (class_name == "mouse") and ("computer mouse" in desc): continue
                if (class_name == "keyboard") and ("keyboard instrument" in desc): continue
                if (class_name == "bed") and ("baby bed" in desc or "bedroom furniture" in desc): continue
                if (class_name == "book") and ("bookcase" in desc): continue
                if (class_name == "train") and ("restraint" in desc): continue
                if (class_name == "spoon") and ("spoonbill" in desc): continue
                if (class_name == "person") and ("personal" in desc): continue
                if (class_name == "fork") and ("forklift" in desc): continue
                if (class_name == "cup") and ("porcupine" in desc): continue
                if (class_name == "cat") and ("domesticated animal" in desc or ("catarrhine" in desc)): continue
                if (class_name == "car") and ("carnivore" in desc): continue
                if (class_name == "bus") and ("business" in desc): continue
                if (class_name == "monitor") and ("lizard" in desc): continue
                if (class_name == "plant") and ("industrial" in desc or "structure" in desc or "organ" in desc): continue

                names = [name.lower() for name in desc.split(', ')]# for word in name.split(' ')]
                if any((class_name.lower() in name for name in names)):
                    class_to_wordnet_superclass_ids[cls].add((wnid, desc))
                    break

    print("No superclass found:", [v for v in classes if v not in class_to_wordnet_superclass_ids.keys()])
    pprint(class_to_wordnet_superclass_ids)

    ## Now find all subclasses for the superclasses
    class_to_wnids_with_desc: dict[str, list] = defaultdict(list)
    for cls in classes:

        # No superclass found?
        if cls not in class_to_wordnet_superclass_ids.keys():
            class_to_wnids_with_desc[cls] = []
            continue

        # Go through all children classes
        for ancestor_wnid, _ in class_to_wordnet_superclass_ids[cls]:
            # if it is already a leaf, add it right away
            if len(imagenet_hierarchy.tree[ancestor_wnid].descendants_all) == 0:
                class_to_wnids_with_desc[cls].append((ancestor_wnid, imagenet_hierarchy.wnid_to_name[ancestor_wnid]))
            else:
                for _, wnid in enumerate(imagenet_hierarchy.tree[ancestor_wnid].descendants_all):
                    if wnid in imagenet_hierarchy.in_wnids:
                        class_to_wnids_with_desc[cls].append((wnid, imagenet_hierarchy.wnid_to_name[wnid]))

    pprint(class_to_wnids_with_desc)

    class_to_wnids: dict[str, list] = {cls: [wnid_desc[0] for wnid_desc in wnids_descs]
                                      for cls, wnids_descs in class_to_wnids_with_desc.items()}
    pprint(class_to_wnids)