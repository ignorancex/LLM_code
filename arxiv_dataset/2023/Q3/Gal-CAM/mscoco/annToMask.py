import os
import imageio
import numpy as np
from torch import multiprocessing
from pycocotools.coco import COCO
from torch.utils.data import Subset

category_map = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}

def work(process_id, infer_dataset, coco, mask_path):
    databin = infer_dataset[process_id]
    print(len(databin))
    for imgId in databin:
        curImg = coco.imgs[imgId]
        imageSize = (curImg['height'], curImg['width'])
        labelMap = np.zeros(imageSize)

        # Get annotations of the current image (may be empty)
        annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)
        imgAnnots = coco.loadAnns(annIds)

        # Combine all annotations of this image in labelMap
        # labelMasks = mask.decode([a['segmentation'] for a in imgAnnots])
        for i in range(len(imgAnnots)):
            labelMask = coco.annToMask(imgAnnots[i]) == 1
            newLabel = imgAnnots[i]['category_id']
            labelMap[labelMask] = category_map[str(newLabel)]

        imageio.imsave(os.path.join(mask_path, str(imgId) + '.png'), labelMap.astype(np.uint8))
        imageio.imsave(os.path.join(mask_path, str(curImg['file_name'])), labelMap.astype(np.uint8))

if __name__ == '__main__':
    #os.environ['MKL_THREADING_LAYER'] = 'GNU'
    annFile = '../dataset/annotations/instances_train2017.json'
    mask_path = '../dataset/mask/train2017'
    os.makedirs(mask_path, exist_ok=True)
    coco = COCO(annFile)
    num_workers = 2
    ids = list(coco.imgs.keys())
    print(len(ids))
    num_per_worker = (len(ids)//num_workers) + 1
    dataset = [ ids[i*num_per_worker:(i+1)*num_per_worker] for i in range(num_workers)]
    multiprocessing.spawn(work, nprocs=num_workers, args=(dataset,coco,mask_path), join=True)

    annFile = '../dataset/annotations/instances_val2017.json'
    mask_path = '../dataset/mask/val2017'
    os.makedirs(mask_path, exist_ok=True)
    coco = COCO(annFile)
    ids = list(coco.imgs.keys())
    print(len(ids))
    num_per_worker = (len(ids)//num_workers) + 1
    dataset = [ ids[i*num_per_worker:(i+1)*num_per_worker] for i in range(num_workers)]
    multiprocessing.spawn(work, nprocs=num_workers, args=(dataset,coco,mask_path), join=True)
