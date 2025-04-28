import numpy as np
import os
import os.path as osp
import mscoco.dataloader
import chainercv
#from chainercv.datasets import VOCInstanceSegmentationDataset
from chainercv.datasets import COCOInstanceSegmentationDataset

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util


def run_keys(args):
	# path to annotations file and result file
	data_keyword = args.chainer_eval_set
	ann_file = osp.join(args.mscoco_root,'annotations/instances_'+data_keyword+'2017.json') 
	#res_file = 'results/keypoints_val2017_results.json'
	
	# load annotations and create COCO object
	coco_gt = COCO(ann_file)
	
	coco_results = []
	
	ignore_list = [219, 278, 535]
	for i, pack in enumerate(coco_gt.dataset['images']):
		if i in ignore_list:
			continue
		filename = pack['file_name'].split('.')[0]
		ins_out = np.load(os.path.join(args.ins_seg_out_dir, filename + '.npy'), allow_pickle=True).item()
		
		keypoints_2 = ins_out['keypoints'].tolist()[0][::-1]; keypoints_2.append(2)
		coco_results.append(
						{
							"image_id": pack['id'],
							"category_id": int(ins_out['class'].tolist()[0]+1),
							'keypoints': keypoints_2,
							"score": ins_out['score'].tolist()[0],
							"id": i,
							"bbox": [0,0,0,0],
						})
						

	# load results and create COCO object
	coco_dt = coco_gt.loadRes(coco_results)

	# create COCO evaluation object
	coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')

	# specify which keypoint types to evaluate`4 1	
	coco_eval.params.kpt_oks_sigmas = np.array([0.5]) #1.07
	#[0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72,0.62, 0.62, 0.79, 0.79, 0.72, 0.72, 1.07, 1.07, 0.87])

	# run evaluation
	coco_eval.evaluate()
	coco_eval.accumulate()
	coco_eval.summarize()
	
	print ("For each class:")
	precisions =coco_eval.eval['precision']
	class_names = [x['name'] for x in coco_gt.dataset['categories']]

	#assert len(class_names) == precisions.shape[2]         
	results_per_category = []
	for idx, name in enumerate(class_names):
		precision = precisions[0, :, idx, 0, -1] # see coco_eval.params.iouThrs , 0 is for IoU threshold 0.5
		precision = precision[precision > -1]
		ap = np.mean(precision) if precision.size else float("nan")
		results_per_category.append(ap)
		print ("Class", idx, "AP =", ap)
	print("Mean mIoU = ", np.mean(results_per_category))


def plotPR_keys(args):	
	# path to annotations file and result file
	data_keyword = args.chainer_eval_set
	ann_file = osp.join(args.mscoco_root,'annotations/instances_'+data_keyword+'2017.json') 
	#res_file = 'results/keypoints_val2017_results.json'
	
	# load annotations and create COCO object
	coco_gt = COCO(ann_file)
	
	coco_results = []
	
	ignore_list = [219, 278, 535]
	for i, pack in enumerate(coco_gt.dataset['images']):
		if i in ignore_list:
			continue
		filename = pack['file_name'].split('.')[0]
		ins_out = np.load(os.path.join(args.ins_seg_out_dir, filename + '.npy'), allow_pickle=True).item()
		
		keypoints_2 = ins_out['keypoints'].tolist()[0][::-1]; keypoints_2.append(2)
		coco_results.append(
						{
							"image_id": pack['id'],
							"category_id": int(ins_out['class'].tolist()[0]+1),
							'keypoints': keypoints_2,
							"score": ins_out['score'].tolist()[0],
							"id": i,
							"bbox": [0,0,0,0],
						})
						

	# load results and create COCO object
	coco_dt = coco_gt.loadRes(coco_results)

	# create COCO evaluation object
	coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints')

	# specify which keypoint types to evaluate`4 1	
	coco_eval.params.kpt_oks_sigmas = np.array([0.5]) #1.07
	#[0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72,0.62, 0.62, 0.79, 0.79, 0.72, 0.72, 1.07, 1.07, 0.87])

	# Get the list of category IDs in the ground truth annotations
	catIds = [2, 1, 3, 4]#coco_gt.getCatIds()
	class_names = ['FR-I', 'FR-II', 'FR-X', 'R']
	class_i = 0
	# Set up plot
	plt.figure(figsize=(5, 5))
	
	# Iterate over each category ID and plot the precision-recall curve
	for catId in catIds:
		# Set the category ID for evaluation
		coco_eval.params.catIds = [catId]

		# Compute precision and recall for the category
		coco_eval.evaluate()
		coco_eval.accumulate()
		coco_eval.summarize()
		
		# Extract precision values for the category
		precision = coco_eval.eval['precision'][0, :, 0, 0, 0]
		precision = precision[precision > -1]
		
		# Plot the precision-recall curve for the category
		# Choose color and linestyle for the current class
		color = plt.cm.tab10(catId)
		linestyle = [':', '--', '-'][catId % 3]
		plt.plot(coco_eval.params.recThrs, precision, color=color, linestyle=linestyle, linewidth=4, label='{}'.format(class_names[class_i]))
		class_i+=1

	# Customize plot
	plt.xlabel('Recall', fontsize=16)
	plt.ylabel('Precision', fontsize=16)
	if data_keyword == 'val':
		data_keyword = 'Test'
	if data_keyword == 'train':
		data_keyword = 'Train'
	plt.title(f'Keypoints ({data_keyword})', fontsize=16)
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.legend(fontsize=14)
	# Increase the axes size and label size
	plt.tick_params(axis='both', which='both', labelsize=14)
	plt.savefig(f"./plots/PRcurve_keys_{data_keyword}.pdf", bbox_inches='tight')
	plt.show()


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
        ins_out = np.load(os.path.join(args.ins_seg_out_dir, filename + '.npy'), allow_pickle=True).item()
        if ins_out is None:
            print ('None for', i)
            continue
        gt_masks.append(dataset_ins .get_example_by_keys(i, (1,))[0])
        gt_labels.append(dataset_ins .get_example_by_keys(i, (2,))[0])
        pred_class.append(ins_out['class'])
        pred_mask.append(ins_out['mask'])
        pred_score.append(ins_out['score'])
        n_img += 1
    print(n_img)
    print('0.4iou:', chainercv.evaluations.eval_instance_segmentation_voc(pred_mask, pred_class, pred_score, gt_masks, gt_labels,
                                                                 iou_thresh=0.4))
    print('0.5iou:', chainercv.evaluations.eval_instance_segmentation_voc(pred_mask, pred_class, pred_score, gt_masks, gt_labels,
                                                                 iou_thresh=0.5))
    print('0.6iou:', chainercv.evaluations.eval_instance_segmentation_voc(pred_mask, pred_class, pred_score, gt_masks, gt_labels,
                                                                 iou_thresh=0.6))

	#print(chainercv.evaluations.eval_instance_segmentation_coco(pred_mask, pred_class, pred_score, gt_masks, gt_labels))
    """print('P,R - 0.25iou:', chainercv.evaluations.calc_instance_segmentation_voc_prec_rec(pred_mask, pred_class, pred_score, gt_masks, gt_labels,
                                                                 iou_thresh=0.25))                                                               
    print('P,R - 0.5iou:', chainercv.evaluations.calc_instance_segmentation_voc_prec_rec(pred_mask, pred_class, pred_score, gt_masks, gt_labels,
                                                                 iou_thresh=0.5))                                                            
    print('P,R - 0.75iou:', chainercv.evaluations.calc_instance_segmentation_voc_prec_rec(pred_mask, pred_class, pred_score, gt_masks, gt_labels,
                                                                 iou_thresh=0.75))"""
                                                                 

def plotPR(args):	
	# path to annotations file and result file
	data_keyword = args.chainer_eval_set
	ann_file = osp.join(args.mscoco_root,'annotations/instances_'+data_keyword+'2017.json') 
	#res_file = 'results/keypoints_val2017_results.json'
	
	# load annotations and create COCO object
	coco_gt = COCO(ann_file)
	
	coco_results = []
	
	ignore_list = [219, 278, 535]
	for i, pack in enumerate(coco_gt.dataset['images']):
		if i in ignore_list:
			continue
		filename = pack['file_name'].split('.')[0]
		ins_out = np.load(os.path.join(args.ins_seg_out_dir, filename + '.npy'), allow_pickle=True).item()
		
		keypoints_2 = ins_out['keypoints'].tolist()[0][::-1]; keypoints_2.append(2)
		mask = ins_out['mask']
		
		rle = mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
		rle["counts"] = rle["counts"].decode("utf-8")
		coco_results.append(
						{
							"image_id": pack['id'],
							"category_id": int(ins_out['class'].tolist()[0]+1),
							'keypoints': keypoints_2,
							"score": ins_out['score'].tolist()[0],
							"id": i,
							"bbox": [0,0,0,0],
							"segmentation": rle,
						})
						

	# load results and create COCO object
	coco_dt = coco_gt.loadRes(coco_results)

	# create COCO evaluation object
	coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')

	# specify which keypoint types to evaluate`4 1	
	coco_eval.params.kpt_oks_sigmas = np.array([0.5]) #1.07
	#[0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72,0.62, 0.62, 0.79, 0.79, 0.72, 0.72, 1.07, 1.07, 0.87])

	# Get the list of category IDs in the ground truth annotations
	catIds = [2, 1, 3, 4]#coco_gt.getCatIds()
	class_names = ['FR-I', 'FR-II', 'FR-X', 'R']
	class_i = 0
	# Set up plot
	plt.figure(figsize=(5, 5))
	
	# Iterate over each category ID and plot the precision-recall curve
	for catId in catIds:
		# Set the category ID for evaluation
		coco_eval.params.catIds = [catId]

		# Compute precision and recall for the category
		coco_eval.evaluate()
		coco_eval.accumulate()
		coco_eval.summarize()
		
		# Extract precision values for the category
		precision = coco_eval.eval['precision'][0, :, 0, 0, -1]
		precision = precision[precision > -1]
		
		# Plot the precision-recall curve for the category
		# Choose color and linestyle for the current class
		color = plt.cm.tab10(catId)
		linestyle = [':', '--', '-'][catId % 3]
		plt.plot(coco_eval.params.recThrs, precision, color=color, linestyle=linestyle, linewidth=4, label='{}'.format(class_names[class_i]))
		class_i+=1

	# Customize plot
	plt.xlabel('Recall', fontsize=16)
	plt.ylabel('Precision', fontsize=16)
	if data_keyword == 'val':
		data_keyword = 'Test'
	if data_keyword == 'train':
		data_keyword = 'Train'
	plt.title(f'Segmentation ({data_keyword})', fontsize=16)
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.legend(fontsize=14)
	# Increase the axes size and label size
	plt.tick_params(axis='both', which='both', labelsize=14)
	plt.savefig(f"./plots/PRcurve_segm_{data_keyword}.pdf", bbox_inches='tight')
	plt.show()
	
