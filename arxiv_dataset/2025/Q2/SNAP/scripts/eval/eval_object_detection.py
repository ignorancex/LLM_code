import os
import json
import copy
from os.path import join
import pandas as pd
from tqdm import tqdm
from cocoLRP import COCO
from cocoevalLRP import COCOeval, Params
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

sensor_bias_gt = f'annotations/SNAP_COCO.json'
data_path = os.environ['SNAP_DATA_PATH']

attrs = ['EV_offset', 'Lux', 'Shutter_speed', 'F-Number', 'ISO']
metrics = ['AP', 'oLRP', 'oLRP Loc', 'oLRP FP', 'oLRP FN']

stats_labels = ['AP', 
				'AP 50', 
				'AP 75', 
				'AP small', 
				'AP medium', 
				'AP large', 
				'AR maxDets 1', 
				'AR maxDets 10', 
				'AR maxDets 100',
				'AR small',
				'AR medium', 
				'AR large',
				'oLRP',
				'oLRP Loc',
				'oLRP FP',
				'oLRP FN',
				'oLRP small',
				'oLRP medium',
				'oLRP large']

data_df = pd.read_excel('annotations/SNAP.xlsx')
data_df.drop(data_df[data_df['EV_offset'] == 'auto'].index, inplace=True)

data_df.drop(data_df[data_df['EV_offset'] == -6].index, inplace=True)
data_df.drop(data_df[data_df['EV_offset'] == 9].index, inplace=True)

shutter_speed = data_df['Shutter_speed'].unique()
shutter_speed_float = [eval(str(x)) for x in shutter_speed]
sort_idx = np.argsort(shutter_speed_float)
shutter_speed_sorted = list(shutter_speed[sort_idx])

stats_df = []

tot_ap_all = []

# compute metrics by image
stats_by_image = {metric: None for metric in metrics}

tot_stats_by_attr = {attr: [] for attr in attrs}

raw_res_dir = 'raw_results/object_detection/'
eval_res_dir = 'eval_results/object_detection/'

# load ground truth
coco_gt = COCO(sensor_bias_gt)

raw_res_paths = sorted([x for x in os.listdir(raw_res_dir)])
# iterate over raw model results
for raw_res_path in raw_res_paths:
	print(raw_res_path)
	model_name = os.path.basename(raw_res_path).replace('.json', '')

	
	# stats by attribute for each model
	stats_by_attr = {attr: None for attr in attrs}

	save_path = join(eval_res_dir, raw_res_path.replace('json', 'xlsx'))

	# load predictions for this model
	coco_dt = coco_gt.loadRes(join(raw_res_dir, raw_res_path))

	# evaluate on the whole data
	coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
	coco_eval.evaluate()
	coco_eval.accumulate()
	coco_eval.summarize()

	eval_res = {x: y for x, y in zip(stats_labels, coco_eval.stats)}

	# save results
	tot_ap = {'Model': model_name, 'AP': eval_res['AP'], 'oLRP': eval_res['oLRP'], 'oLRP Loc': eval_res['oLRP Loc'], 'oLRP FP': eval_res['oLRP FP']}
	tot_ap_all.append(tot_ap)

	if os.path.exists(save_path):
		eval_res_df = pd.read_excel(save_path, sheet_name='By image')
	else:
		records = []
		num_images = len(data_df)
		for img_idx, img_path in tqdm(zip(data_df.index.to_list(), data_df['Path'].to_list()), total=num_images):
			record = {'Path': img_path}

			params = copy.deepcopy(coco_eval.params)
			params.imgIds = [img_idx]
			coco_eval.accumulate(p=params)

			# print the results
			coco_eval.summarize()
			record.update({k:v for k, v in zip(stats_labels, coco_eval.stats)})
			#print(record)
			records.append(record)

		eval_res_df = pd.DataFrame.from_dict(records)

	for metric in metrics:
		if stats_by_image[metric] is None:
			stats_by_image[metric] = pd.DataFrame(data_df['Path'].to_list(), columns=['Path'])

		stats_by_image[metric][model_name] = eval_res_df[metric].to_list()

	eval_res_df = pd.merge(data_df, eval_res_df, how='inner', on='Path')

	# evaluate by attribute
	for attr in attrs:
		attr_df = []
		# iterate over unique values of the attribute (e.g. EV_offset is in range [-8, 8])
		print(attr, data_df[attr].unique())

		if attr == 'Shutter_speed':
			attr_vals = shutter_speed_sorted
		else:
			attr_vals = sorted(data_df[attr].unique())

		for val in attr_vals:
			# get corresponding image ids for each value
			imgIds = data_df[data_df[attr] == val].index.to_list()

			params = copy.deepcopy(coco_eval.params)
			params.imgIds = imgIds
			coco_eval.accumulate(p=params)

			# print the results
			coco_eval.summarize()
			record = {'Num_images': len(imgIds), attr: val}
			record.update({k:v for k, v in zip(stats_labels, coco_eval.stats)})
			attr_df.append(record)

		# evaluation results for all values of the attribute
		attr_df = pd.DataFrame.from_dict(attr_df)
		rename_dict = {m: f'{model_name}_{m}' for m in attr_df.columns if m not in ['Num_images', attr]} # adds model name to metric label
		attr_df = attr_df.rename(columns=rename_dict).transpose()
		
		stats_by_attr[attr] = attr_df

		tot_stats_by_attr[attr].append(attr_df)

	with pd.ExcelWriter(save_path) as writer:
		pd.DataFrame.from_dict([tot_ap]).to_excel(writer, sheet_name='Total AP')
		
		eval_res_df.to_excel(writer, sheet_name='By image')
		#for metric in metrics:
		#	stats_by_image[metric][['Path', model_name]].to_excel(writer, sheet_name=f'By_image_{metric}')

		for attr, attr_df in stats_by_attr.items():
			df = attr_df.copy()
			df.index = df.index.str.replace(model_name + '_', '')
			df.to_excel(writer, sheet_name=attr)

# write summary statistics for all data
tot_ap_all = pd.DataFrame.from_dict(tot_ap_all).sort_values(by='AP', ascending=False).round(2)
sort_order = tot_ap_all.index

# extract rows with metric of interest from stats for all models
def get_metric_df(df_list, metric, new_sort_order):
	all_df = pd.concat(df_list, axis=0)
	metric_df = all_df[all_df.index.str.endswith(metric)].astype(float).round(2)
	metric_df.index = metric_df.index.str.replace(f'_{metric}','')
	metric_df = metric_df.reset_index().reindex(labels=new_sort_order)
	metric_df = pd.concat([all_df.reset_index().head(n=2), metric_df], axis=0)
	return metric_df

# write to excel
writer = pd.ExcelWriter(join(eval_res_dir, 'all_models.xlsx'), engine="xlsxwriter")
tot_ap_all.to_excel(writer, index=False, sheet_name='all_data')
workbook = writer.book
#worksheet = writer.sheets['all_data']
(max_row, max_col) = tot_ap_all.shape
writer.sheets['all_data'].conditional_format(1, 1, max_row, max_col, {'type': '3_color_scale'})

# write per-image top1 accuracy and top1 confidence for all models
for metric in metrics:
	stats_by_image[metric].to_excel(writer, index=False, sheet_name=f'By_image_{metric}')

# write dataframes for each metric and attribute to a separate sheet
for metric in metrics:
	for attr, attr_df in tot_stats_by_attr.items():
		sheet_name = f'{attr}_{metric}'
		attr_metric_df = get_metric_df(attr_df, metric, sort_order)
		attr_metric_df.to_excel(writer, index=False, sheet_name=sheet_name)
		(max_row, max_col) = attr_metric_df.shape
		# conditional format on the whole table
		writer.sheets[sheet_name].conditional_format(3, 1, max_row, max_col, {'type': '3_color_scale'})
writer.close()