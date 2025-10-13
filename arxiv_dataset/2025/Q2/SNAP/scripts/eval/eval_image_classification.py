import os
import csv
import json
from os.path import join
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle as pkl

raw_res_dir = 'raw_results/image_classification/'
eval_res_dir = 'eval_results/image_classification/'

data_df = pd.read_excel(f'annotations/SNAP.xlsx', engine='openpyxl')
auto_df = data_df[data_df['EV_offset'] == 'auto']
data_df.drop(data_df[data_df['EV_offset'] == 'auto'].index, inplace=True)

data_df.drop(data_df[data_df['EV_offset'] == -6].index, inplace=True)
data_df.drop(data_df[data_df['EV_offset'] == 9].index, inplace=True)

class_label_dict = {'spoon': [910],
				    'bottle': [898],
				    'phone': [487],
				    'umbrella': [879],
				    'tv': [664],
				    'banana': [954],
				    'orange': [950],
				    'basketball': [430],
				    'ball': [430],
				    'cup': [968, 504], # 968 - cup, 504 - coffee mug
				    'laptop': [620, 681], # 620 - laptop, 681 - notebook
				    'remote': [761],
				    'book': [921],
				    'mouse': [673],
				    'keyboard': [508],
				    'backpack': [414],
				    'water bottle': [898],
				    'tie': [906],
				    'comic book': [917], 
				    'remote': [761],
				    'cup': [968]
				    }

tot_acc_all = []
attrs = ['Class', 'EV_offset']
metrics = ['top1_acc', 'top5_acc']

tot_stats_by_attr = {attr: [] for attr in attrs}

# compute metrics by image
stats_by_image = {metric: None for metric in ['top1_acc']}

res_files = sorted([x for x in os.listdir(raw_res_dir)])

# iterate over the raw results for each image classification model
for raw_res_path in res_files:

	model_name = os.path.basename(raw_res_path).replace(f'.xlsx', '')

	print(f'Processing {raw_res_path}...')
	raw_res_df = pd.read_excel(join(raw_res_dir, raw_res_path), engine='openpyxl')

	# compute average metrics by attribute
	stats_by_attr = {attr: None for attr in attrs}

	eval_res_dict = []

	# compute top1, top5 error, and top1 semantic distance for each image
	for idx, row in tqdm(raw_res_df.iterrows()):
		img_class = ' '.join(row['Path'].split('/')[0].split('_')[1:])
		gt_labels = class_label_dict[img_class]
		pred_class = [x[0] for x in eval(row['Pred'])]

		top1 = pred_class[0] in gt_labels
		top5 = len(list(set(pred_class) & set(gt_labels))) > 0

		eval_res_dict.append({'Class': img_class, 'Path': row['Path'], 'top1_acc': int(top1), 'top5_acc': int(top5)})

	# aggregate statistics by attribute
	def summary_stats(eval_df, data_df, by_attr='Class'):

		acc_df = eval_df.groupby(by=[by_attr], as_index=False).sum(numeric_only=True)[[by_attr, 'top1_acc', 'top5_acc']].copy(deep=True)
		acc_df['Num_images'] = eval_df.groupby(by=[by_attr], as_index=False).count()['top1_acc']
		acc_df['top1_acc'] = acc_df['top1_acc']/acc_df['Num_images']
		acc_df['top5_acc'] = acc_df['top5_acc']/acc_df['Num_images']

		return acc_df

	eval_res_df = pd.DataFrame.from_dict(eval_res_dict)

	for metric in ['top1_acc']:	
		if stats_by_image[metric] is None:
			stats_by_image[metric] = pd.DataFrame(eval_res_df['Path'].to_list(), columns=['Path'])

		stats_by_image[metric][model_name] = eval_res_df[metric].to_list()

	tot_acc = eval_res_df[['top1_acc', 'top5_acc']].sum()/len(eval_res_df)
	eval_res_df = pd.merge(data_df, eval_res_df, how='inner', on='Path')
	print(len(eval_res_df))

	for attr in tot_stats_by_attr.keys():
		stats_by_attr[attr] = summary_stats(eval_res_df, data_df, by_attr=attr)

	with pd.ExcelWriter(join(eval_res_dir, raw_res_path)) as writer:
		eval_res_df.to_excel(writer, sheet_name='By image')
		tot_acc.to_excel(writer, sheet_name='Total acc')
		for attr, attr_df in stats_by_attr.items():
			attr_df.to_excel(writer, sheet_name=attr)	

	tot_acc_all.append({'Model': model_name, 'top1_acc': tot_acc[0], 'top5_acc': tot_acc[1]})

	for attr, attr_df in stats_by_attr.items():
		select_cols = [attr, 'Num_images'] + metrics
		rename_cols = [attr, 'Num_images'] + [f'{model_name}_{m}' for m in metrics]
		rename_dict = {s:r for s,r in zip(select_cols, rename_cols)}

		by_attr_t = attr_df[select_cols].rename(columns=rename_dict).transpose()
		tot_stats_by_attr[attr].append(by_attr_t)

# write summary statistics for all data
tot_acc_all = pd.DataFrame.from_dict(tot_acc_all).sort_values(by='top1_acc', ascending=False).round(2)
sort_order = tot_acc_all.index

def get_metric_df(df_list, metric, new_sort_order):
	#all_df = pd.concat([df.iloc[2:, :] for df, i in enumerate(df_list) if i > 0 else df], axis=0)
	# from each metric df, remove first 2 rows (counts and metric)
	all_df = pd.concat([df.iloc[2:, :] if i > 0 else df for i, df in enumerate(df_list)], axis=0)
	metric_df = all_df[all_df.index.str.contains(metric)].astype(float).round(2)
	metric_df.index = metric_df.index.str.replace(f'_{metric}','')
	metric_df = metric_df.reset_index().reindex(labels=new_sort_order)
	metric_df = pd.concat([all_df.reset_index().head(n=2), metric_df], axis=0)
	return metric_df

# write to excel
writer = pd.ExcelWriter(join(eval_res_dir, f'all_models.xlsx'), engine="xlsxwriter")
tot_acc_all.to_excel(writer, index=False, sheet_name='all_data')
workbook = writer.book
#worksheet = writer.sheets['all_data']
(max_row, max_col) = tot_acc_all.shape

# write top1 and top5 acc for all models for the entire dataset
writer.sheets['all_data'].conditional_format(1, 1, max_row, max_col, {'type': '3_color_scale'})

# write per-image top1 accuracy and top1 confidence for all models
for metric in ['top1_acc']:
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