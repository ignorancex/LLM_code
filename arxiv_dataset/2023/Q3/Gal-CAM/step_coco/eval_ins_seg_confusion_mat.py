import numpy as np
import os
import os.path as osp
#import mscoco.dataloader
#import chainercv
#from chainercv.datasets import VOCInstanceSegmentationDataset
#from chainercv.datasets import COCOInstanceSegmentationDataset

#from pycocotools.coco import COCO
#from pycocotools.cocoeval import COCOeval
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import faster_coco_eval.core.mask as mask_util
import faster_coco_eval
from faster_coco_eval import COCO
from faster_coco_eval.extra import PreviewResults
	
def confusion_mat(args, threshold_iou = 0.5, iouType = 'segm'):	
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
		category_id = int(ins_out['class'].tolist()[0]+1)
		score = ins_out['score'].tolist()[0]
		
		rle = mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
		rle["counts"] = rle["counts"].decode("utf-8")
		coco_results.append(
						{
							"image_id": pack['id'],
							"category_id": category_id,
							'keypoints': keypoints_2,
							"score": score,
							"id": i,
							"bbox": [0,0,0,0],
							"segmentation": rle,
						})
						

	# load results and create COCO object
	coco_dt = coco_gt.loadRes(coco_results)

	results = PreviewResults(cocoGt=coco_gt, cocoDt=coco_dt, iou_tresh=threshold_iou, iouType=iouType)
	confusion_matrix = results.compute_confusion_matrix()
	
	# Get the list of category IDs in the ground truth annotations
	catIds = [2, 1, 3, 4] #coco_gt.getCatIds()
	class_names = ['FR-I', 'FR-II', 'FR-X', 'R']
	num_imgs_p_class = []
	for catId in range(1,len(catIds)+1):
		num_imgs_p_class.append(len(coco_gt.getImgIds(catIds=catId)))
	
	# Normalize the confusion matrix
	confusion_matrix_norm = confusion_matrix/np.array(num_imgs_p_class)[:, np.newaxis]
	print(confusion_matrix_norm)
	# Change names to class_names
	FP = confusion_matrix_norm[:,4]; FP[[0, 1]] = FP[[1, 0]]
	FN = confusion_matrix_norm[:,5]; FN[[0, 1]] = FN[[1, 0]]
	class_names_json = ['FR-II', 'FR-I', 'FR-X', 'R']
	mapping = [class_names_json.index(label) for label in class_names]
	confusion_matrix_norm = confusion_matrix_norm[mapping][:, mapping]
	confusion_matrix_norm = np.hstack((confusion_matrix_norm, np.array(FP).reshape(-1, 1), np.array(FN).reshape(-1, 1)))
	print(confusion_matrix_norm)
	
	class_names_new = ['FR-I', 'FR-II', 'FR-X', 'R', 'FP', 'FN']
	
	# Create a figure and axis
	fig, ax = plt.subplots(figsize=(7, 7))

	# Create a color-coded text representation of the confusion matrix
	im = ax.imshow(confusion_matrix_norm, cmap='Blues', vmin = 0, vmax = 1)

	# Set axis labels and title
	ax.set_xlabel('Predicted Class', fontsize=16)
	ax.set_ylabel('True Class', fontsize=16)	
	if data_keyword == 'val':
		data_keyword = 'Testing set'
	if data_keyword == 'train':
		data_keyword = 'Training set'
	ax.set_title(f'{data_keyword} (IoU>0.5)', fontsize=16)

	# Set x and y axis tick labels and their font size
	ax.set_xticks(np.arange(len(class_names_new)))
	ax.set_yticks(np.arange(len(class_names)))
	ax.set_xticklabels(class_names_new, fontsize=14)
	ax.set_yticklabels(class_names, fontsize=14)

	# Rotate x-axis tick labels for better readability
	plt.xticks(rotation=45)

	# Loop over data dimensions and create text annotations
	for i in range(len(class_names)):
		for j in range(len(class_names_new)):
			text = ax.text(j, i, format(confusion_matrix_norm[i, j], '.2f'), ha='center', va='center', color='black')

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	# Add a colorbar
	cbar = fig.colorbar(im, cax=cax)
	# Increase font size of colorbar tick labels
	cbar.ax.tick_params(labelsize=14)
	
	plt.savefig(f"./plots/Confusion_segm_{data_keyword}.pdf", bbox_inches='tight')

	# Show the plot
	plt.show()
	
	
	
	
	"""#############################
	# create COCO evaluation object
	coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')

	# specify which keypoint types to evaluate`4 1	
	coco_eval.params.kpt_oks_sigmas = np.array([0.5]) #1.07
	#[0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72,0.62, 0.62, 0.79, 0.79, 0.72, 0.72, 1.07, 1.07, 0.87])

	# Get the list of category IDs in the ground truth annotations
	catIds = [2, 1, 3, 4] #coco_gt.getCatIds()
	class_names = ['FR-I', 'FR-II', 'FR-X', 'R']
	class_i = 0
	
	
	# Set up plot
	#plt.figure(figsize=(5, 5))
	
	# Get ground truth annotations for the image
	gt_ann_ids = coco_gt.getAnnIds()
	gt_anns = coco_gt.loadAnns(gt_ann_ids)

	# Get predicted annotations for the image
	pred_ann_ids = coco_dt.getAnnIds()
	pred_anns = coco_dt.loadAnns(pred_ann_ids)
	
	#print (len(gt_anns), len(pred_anns))
	
	# Initialize confusion matrix
	n_classes = len(catIds)
	confusion_matrix_n = np.zeros((n_classes, n_classes), dtype=np.float)
	confusion_matrix_d = np.zeros((n_classes, n_classes), dtype=np.float)
	
	# Iterate over ground truth and predicted annotations
	ious = []
	for gt_ann in gt_anns:
		if gt_ann['id'] in ignore_list:
			continue
		#gt_mask = gt_ann['segmentation']
		curImg = coco_gt.imgs[gt_ann['id']]
		imageSize = (curImg['height'], curImg['width'])
		gt_mask = np.zeros(imageSize, dtype=np.int32)
		gt_mask = coco_gt.annToMask(gt_ann)# == 1
		gt_category_id = gt_ann['category_id']

		for pred_ann in pred_anns:
			if gt_ann['id'] == pred_ann['id']:
				pred_mask = np.array(mask_util.decode(pred_ann['segmentation']))
				pred_category_id = pred_ann['category_id']
				pred_score = pred_ann['score']

				intersection = np.logical_and(gt_mask, pred_mask)
				union = np.logical_or(gt_mask, pred_mask)

				iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0
				if iou >= iou_threshold and pred_score >= 0.5:
					confusion_matrix_n[pred_category_id-1, gt_category_id-1] += 1 # iou
				confusion_matrix_d[pred_category_id-1, gt_category_id-1] += 1
	
	#n_sources = []			
	#for catId in catIds:
	#	# Get predicted annotations for the category
	#	pred_ann_ids = coco_dt.getAnnIds(catIds=catId)
	#	n_sources.append(len(pred_ann_ids))
	#confusion_matrix = confusion_matrix/n_sources
	
	# Normalize the confusion matrix
	confusion_matrix_norm = confusion_matrix_n/confusion_matrix_d
	#confusion_matrix_norm = confusion_matrix_n / confusion_matrix_n.sum(axis=1, keepdims=True)
	print(confusion_matrix_norm.T)
	
	# Change names to class_names
	class_names_json = ['FR-II', 'FR-I', 'FR-X', 'R']
	mapping = [class_names_json.index(label) for label in class_names]
	confusion_matrix_norm = confusion_matrix_norm[mapping][:, mapping]
	
	# Create a figure and axis
	fig, ax = plt.subplots(figsize=(5, 5))

	# Create a color-coded text representation of the confusion matrix
	im = ax.imshow(confusion_matrix_norm, cmap='Blues', vmin = 0, vmax = 1)

	# Set axis labels and title
	ax.set_xlabel('Predicted Class', fontsize=16)
	ax.set_ylabel('True Class', fontsize=16)	
	if data_keyword == 'val':
		data_keyword = 'Test'
	if data_keyword == 'train':
		data_keyword = 'Train'
	ax.set_title(f'Segmentation ({data_keyword})', fontsize=16)

	# Set x and y axis tick labels and their font size
	ax.set_xticks(np.arange(len(class_names)))
	ax.set_yticks(np.arange(len(class_names)))
	ax.set_xticklabels(class_names, fontsize=14)
	ax.set_yticklabels(class_names, fontsize=14)

	# Rotate x-axis tick labels for better readability
	plt.xticks(rotation=45)

	# Loop over data dimensions and create text annotations
	for i in range(len(class_names)):
		for j in range(len(class_names)):
			text = ax.text(j, i, format(confusion_matrix_norm[i, j], '.2f'), ha='center', va='center', color='black')

	# Add a colorbar
	cbar = fig.colorbar(im)

	# Increase font size of colorbar tick labels
	cbar.ax.tick_params(labelsize=14)
	
	plt.savefig(f"./plots/Confusion_segm_{data_keyword}.pdf", bbox_inches='tight')

	# Show the plot
	plt.show()


	
	# Initialize confusion matrix
	n_classes = len(catIds)
	confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.float)

	# Iterate over each category ID and plot the precision-recall curve
	for catId in catIds:
		# Get ground truth annotations for the category
		gt_ann_ids = coco_gt.getAnnIds(catIds=catId)
		gt_anns = coco_gt.loadAnns(gt_ann_ids)
	
		# Get predicted annotations for the category
		pred_ann_ids = coco_dt.getAnnIds(catIds=catId)
		pred_anns = coco_dt.loadAnns(pred_ann_ids)
		
		print (len(gt_anns), len(pred_anns))
	
		# Create masks for ground truth and predicted annotations
		#gt_masks = [mask_utils.decode(gt_ann['segmentation']) for gt_ann in gt_anns]
		#pred_masks = [mask_utils.decode(pred_ann['segmentation']) for pred_ann in pred_anns]
		#gt_masks = [gt_ann['segmentation'] for gt_ann in gt_anns]
		pred_masks = np.array([mask_util.decode(pred_ann['segmentation']) for pred_ann in pred_anns])
		gt_masks = np.zeros(pred_masks.shape, dtype=np.int32)
		for i in range(len(gt_anns)):
			gt_masks[i] = coco_gt.annToMask(gt_anns[i]) #== 1
	
		# Compute the intersection over union (IoU) between each ground truth and predicted mask
		iou_matrix = np.zeros((len(gt_masks), len(pred_masks)))
		for i, gt_mask in enumerate(gt_masks):
			for j, pred_mask in enumerate(pred_masks):
				if gt_ann['id'] == pred_ann['id']:
					intersection = np.logical_and(gt_mask, pred_mask).sum()
					union = np.logical_or(gt_mask, pred_mask).sum()
					iou = intersection / union if union > 0 else 0.0
					iou_matrix[i, j] = iou
	
		# Normalize the IoU values
		#iou_matrix /= np.sum(iou_matrix, axis=1, keepdims=True)
	
		# Update confusion matrix
		for i in range(len(gt_masks)):
			best_iou = np.max(iou_matrix[i])
			best_match = np.argmax(iou_matrix[i])
		
			# Determine true positive or false positive based on IoU threshold
			if best_iou > iou_threshold:
				confusion_matrix[catId-1, catId-1] += 1  # True positive
			else:
				confusion_matrix[catId-1, best_match] += 1  # False positive



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
	#plt.savefig(f"./plots/PRcurve_segm_{data_keyword}.pdf", bbox_inches='tight')
	plt.show()"""
