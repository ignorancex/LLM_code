import numpy as np
import os
import os.path as osp
import mscoco.dataloader
import chainercv
#from chainercv.datasets import VOCInstanceSegmentationDataset
from chainercv.datasets import COCOInstanceSegmentationDataset

def run(args):
    data_keyword = args.chainer_eval_set
    dataset_ins = COCOInstanceSegmentationDataset(split=args.chainer_eval_set, data_dir=args.mscoco_root)
    dataset = mscoco.dataloader.COCOSegmentationDataset(image_dir = osp.join(args.mscoco_root, data_keyword+'2017/'), 
        anno_path= osp.join(args.mscoco_root,'annotations/instances_'+data_keyword+'2017.json'),
        masks_path=osp.join(args.mscoco_root,'mask/'+data_keyword+'2017'),crop_size=512)
    pred_class = []
    pred_mask = []
    pred_score = []
    gt_masks = []
    gt_labels = []
    out_dir = None
    n_img = 0
    for i, pack in enumerate(dataset):
        #print(id)
        if (out_dir is not None) and (not os.path.exists(os.path.join(out_dir, id + '.npy'))):
            continue
        filename = pack['name'].split('.')[0]
        gt_masks.append(dataset_ins .get_example_by_keys(i, (1,))[0])
        gt_labels.append(dataset_ins .get_example_by_keys(i, (2,))[0])
        ins_out = np.load(os.path.join(args.ins_seg_out_dir, filename + '.npy'), allow_pickle=True).item()
        pred_class.append(ins_out['class'])
        pred_mask.append(ins_out['mask'])
        pred_score.append(ins_out['score'])
        n_img += 1
    print(n_img)
    print('0.25iou:', chainercv.evaluations.eval_instance_segmentation_voc(pred_mask, pred_class, pred_score, gt_masks, gt_labels,
                                                                 iou_thresh=0.25))
    print('0.5iou:', chainercv.evaluations.eval_instance_segmentation_voc(pred_mask, pred_class, pred_score, gt_masks, gt_labels,
                                                                 iou_thresh=0.5))
    print('0.75iou:', chainercv.evaluations.eval_instance_segmentation_voc(pred_mask, pred_class, pred_score, gt_masks, gt_labels,
                                                                 iou_thresh=0.75))
                                                                 
    """print('P,R - 0.25iou:', chainercv.evaluations.calc_instance_segmentation_voc_prec_rec(pred_mask, pred_class, pred_score, gt_masks, gt_labels,
                                                                 iou_thresh=0.25))                                                               
    print('P,R - 0.5iou:', chainercv.evaluations.calc_instance_segmentation_voc_prec_rec(pred_mask, pred_class, pred_score, gt_masks, gt_labels,
                                                                 iou_thresh=0.5))                                                            
    print('P,R - 0.75iou:', chainercv.evaluations.calc_instance_segmentation_voc_prec_rec(pred_mask, pred_class, pred_score, gt_masks, gt_labels,
                                                                 iou_thresh=0.75))"""
