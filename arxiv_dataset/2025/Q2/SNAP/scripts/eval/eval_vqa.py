import os
import re
import json
from os.path import join, isfile
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle as pkl

raw_res_dir = 'raw_results/vqa/'
eval_res_dir = 'eval_results/vqa/'

os.makedirs(eval_res_dir, exist_ok=True)

qrange = range(1,5)

data_df = pd.read_excel(f'annotations/SNAP.xlsx', engine='openpyxl')

auto_df = data_df[data_df['EV_offset'] == 'auto']
data_df.drop(data_df[data_df['EV_offset'] == 'auto'].index, inplace=True)

data_df.drop(data_df[data_df['EV_offset'] == -6].index, inplace=True)
data_df.drop(data_df[data_df['EV_offset'] == 9].index, inplace=True)

with open(f'annotations/SNAP_VQA.json') as fid:
	question_dict = json.load(fid)

data_df = data_df.loc[data_df['Path'].isin(question_dict)]

shutter_speed = data_df['Shutter_speed'].unique()
shutter_speed_float = [eval(str(x)) for x in shutter_speed]
sort_idx = np.argsort(shutter_speed_float)
shutter_speed = list(shutter_speed[sort_idx])

tot_acc_all = []
attrs = ['EV_offset'] # Category
metrics = ['acc', 'len', 'acc1']

# max answer lengths
max_ans_len = {'A1': 14, 'A2': 5, 'A3': 14, 'A4': 5}

# open-ended: what class?
def eval_q1(obj_id, gt_ans, ans_opts, pred_ans, pred_ans_len):	

	acc = int(any([obj_id in x.strip() for x in pred_ans.lower().split(',')]))
	
	err_fact = int(not acc)

	# faithfullness
	# check that only one class is returned
	# check that answer length is not too long
	err_faith = int((len(pred_ans.split(',')) > 1) or (pred_ans_len > max_ans_len['A1']))

	# hard accuracy
	acc_h = int(not (err_fact or err_faith))

	return acc, acc_h, err_fact, err_faith

# open-ended: how many
def eval_q2(obj_id, gt_ans, ans_opts, pred_ans, pred_ans_len):
	gt_ans_list = [x.strip() for x in gt_ans.split(',')]

	# check if any of the gt answers match the provided answer
	acc = int(any([ans in str(pred_ans).lower() for ans in gt_ans_list]))

	# if the number is wrong or there is no answer, factual error
	err_fact = int(not acc)

	# faithfullness error if the answer is too long
	err_faith = int(pred_ans_len > max_ans_len['A2'])

	# hard accuracy: 
	acc_h = int(not (err_fact or err_faith))

	return acc, acc_h, err_fact, err_faith


# MC: what class?
def eval_q3(obj_id, gt_ans, ans_opts, pred_ans, pred_ans_len): 

	# check for exact match 
	acc = int(any([x.strip() in gt_ans for x in pred_ans.lower().split(',')]))

	# factual error if a wrong option is selected or more than one option is selected
	err_fact = int(not acc or (len(pred_ans.split(',')) > 1))

	# faithfullness error if the answer is too long or contains any items not in the options
	err_faith = int((any([x.strip() not in ans_opts for x in pred_ans.lower().split(',')])) or (pred_ans_len > max_ans_len['A3']))
	
	acc_h = int(not (err_fact or err_faith))

	return acc, acc_h, err_fact, err_faith

# MC How many objects?
def eval_q4(obj_id, gt_ans, ans_opts, pred_ans, pred_ans_len):

	regexp = r'([A-Z]?)\)?\s*([0-9]?)'

	match_object = re.match(regexp, pred_ans)
	pred_letter, pred_num = match_object.groups()

	try:
		pred_num = int(pred_num)
	except ValueError:
		for ans in ans_opts:
			pred_num = 0
			if pred_letter in ans:
				pred_num = int(ans.split(') ')[1])


	gt_ans_str = ') '.join([str(x) for x in eval(gt_ans)]) 
	pred_ans_str = ') '.join(match_object.groups())

	# for MC options, check if the prediction matches any of the gt answer strings
	acc = int(pred_ans_str in gt_ans_str)

	err_fact = int(not acc)

	err_faith = int((all([x in ans_opts for x in pred_ans])) or (pred_ans_len > max_ans_len['A4']))

	acc_h = int(not (err_fact or err_faith))

	return acc, acc_h, err_fact, err_faith

def get_ans_options(q_idx, q_str):
	
	# MC what kind of object
	if q_idx == 3:
		ans_opts = [x.strip() for x in q_str.split(':')[1].split(',')]
	# MC how many objects
	elif q_idx == 4:
		ans_opts = [x.strip() for x in q_str.split(':')[1].split('   ')]
	else:
		ans_opts = None

	return ans_opts

# evaluate on length, format
def eval_answers(gt_dict, pred_df):
	
	eval_res_df = []
	num_rows = len(pred_df)

	for idx, row in tqdm(pred_df.iterrows(), total=num_rows):
		
		img_path = row['Path']
		record = {}
		record['Path'] = img_path
		obj_id = img_path.split('/')[0].split('_')[1]

		for q_idx in range(1,5):
			
			ans_idx = f'A{q_idx}'

			try:
				if f'A{q_idx}_raw' in row:
					record[f'A{q_idx}_len'] = len(str(row[f'A{q_idx}_raw']))
				else: 
					record[f'A{q_idx}_len'] = len(str(row[ans_idx]))
			except TypeError:
				record[f'A{q_idx}_len'] = 0

			ans_opts = get_ans_options(q_idx, gt_dict[img_path][f'Q{q_idx}'])

			gt_ans = gt_dict[img_path][ans_idx]
			
			if f'A{q_idx}_raw' in row:
				record[f'A{q_idx}_pred_raw'] = row[f'A{q_idx}_raw'].replace('\n', ' ').strip() if isinstance(row[ans_idx], str) else row[ans_idx]

			# remove new lines from responses if present
			record[f'A{q_idx}_pred'] = row[ans_idx]
			record[f'A{q_idx}_gt'] = gt_ans

			# call functions eval_q3, eval_q4, etc based on the current question index
			acc, acc_hard, err_fact, err_faith = globals()[f'eval_q{q_idx}'](obj_id, gt_ans, ans_opts, str(row[ans_idx]), record[f'A{q_idx}_len'])
			
			record[f'A{q_idx}_acc'] = acc
			record[f'A{q_idx}_err1'] = err_fact
			record[f'A{q_idx}_err2'] = err_faith
			record[f'A{q_idx}_acc1'] = acc_hard


		eval_res_df.append(record)
	return pd.DataFrame.from_dict(eval_res_df)

raw_results = [x for x in os.listdir(raw_res_dir)]

# compute metrics by image

stats_by_image = {}
for metric in metrics:
	for i in qrange:
		stats_by_image[f'A{i}_{metric}'] = None
	stats_by_image[f'avg_{metric}'] = None

tot_stats_by_attr = {}

for res_idx, raw_res_path in enumerate(raw_results):
	print(f'Processing {os.path.basename(raw_res_path)}')
	# get model name from the filename
	model_name = os.path.basename(raw_res_path).replace(f'.xlsx', '')

	eval_res_path = join(eval_res_dir, raw_res_path)

	if isfile(eval_res_path):
		eval_res_df = pd.read_excel(eval_res_path, sheet_name='By image')
	else:
		# read model answers
		raw_res_df = pd.read_excel(join(raw_res_dir, raw_res_path), engine='openpyxl')

		# evaluate answers
		eval_res_df = eval_answers(question_dict, raw_res_df)

		# combine evaluation with data
		eval_res_df = pd.merge(eval_res_df, data_df, how='inner', on=['Path'])

	#stats_by_attr = {attr: None for attr in attrs}
	stats_by_attr = {}

	# aggregate statistics by attribute
	def summary_stats(eval_df, q_idx, by_attr='Category', metric='acc'):	
		col_name = f'A{q_idx}_{metric}'
		group = eval_df[[by_attr, col_name]].groupby(by=[by_attr], as_index=False)
		count_df = group.count()[col_name] # counts
		acc_df = group.sum()
		acc_df.insert(0, 'Num_images', count_df)
		acc_df[col_name] = acc_df[col_name]/acc_df['Num_images'] # accuracy
		return acc_df

	model_acc = {'Model': model_name}
	# compute accuracy for each answer
	for metric in metrics:
		avg_val = 0
		for q_idx in qrange:
			val = eval_res_df[f'A{q_idx}_{metric}'].sum()/len(eval_res_df)
			model_acc[f'A{q_idx}_{metric}'] = val
			avg_val += val
		model_acc[f'avg_{metric}'] = avg_val/len(qrange)
	
	tot_acc_all.append(model_acc)

	for metric in metrics:
		for attr in attrs:
			# compute summary statistics for each attribute per question
			avg_metric = None
			for q_idx in qrange:
				metric_df = summary_stats(eval_res_df, q_idx, by_attr=attr, metric=metric)
				# add a -- separator for the attribute + metric
				stats_by_attr[f'{attr}--A{q_idx}_{metric}'] = metric_df
				if avg_metric is None:
					avg_metric = metric_df.rename(columns={f'A{q_idx}_{metric}': f'avg_{metric}'})
				else:
					avg_metric[f'avg_{metric}'] += metric_df[f'A{q_idx}_{metric}']
				#avg_acc += avg_acc
			# average over all questions for attribute
			avg_metric[f'avg_{metric}'] /= len(qrange)
			stats_by_attr[f'{attr}--avg_{metric}'] = avg_metric

		avg_key = f'avg_{metric}'
		if stats_by_image[avg_key] is None:
			stats_by_image[avg_key] = pd.DataFrame(eval_res_df['Path'].to_list(), columns=['Path'])
	
		for q_idx in qrange:
			key = f'A{q_idx}_{metric}'
			if stats_by_image[key] is None:
				stats_by_image[key] = pd.DataFrame(eval_res_df['Path'].to_list(), columns=['Path'])
			stats_by_image[key][model_name] = eval_res_df[key].to_list()
			if model_name in stats_by_image[avg_key]:
				stats_by_image[avg_key][model_name] += eval_res_df[key].to_list()
			else:
				stats_by_image[avg_key][model_name] = eval_res_df[key].to_list()
		stats_by_image[avg_key][model_name] /= len(qrange)

	with pd.ExcelWriter(eval_res_path) as writer:
		eval_res_df.to_excel(writer, sheet_name='By image', index=False)
		for attr, attr_df in stats_by_attr.items():
			attr_df.to_excel(writer, sheet_name=attr, index=False)	

	for metric in metrics:
		for attr, attr_df in stats_by_attr.items():
			col_name = attr.split('--')[1]
			rename_dict = {col_name: f'{col_name}_{model_name}'}
			by_attr_t = attr_df.rename(columns=rename_dict).transpose()
			if attr in tot_stats_by_attr:
				tot_stats_by_attr[attr].append(by_attr_t)
			else:
				tot_stats_by_attr[attr] = [by_attr_t]

# write summary statistics for all data
tot_acc_all = pd.DataFrame.from_dict(tot_acc_all).sort_values(by='avg_acc', ascending=False).round(2)
sort_order = tot_acc_all.index

def get_metric_df(df_list, metric, new_sort_order):
	all_df = pd.concat(df_list, axis=0).reset_index().drop_duplicates(subset='index').set_index('index')
	metric_df = all_df[all_df.index.str.contains(metric)].astype(float).round(2)
	metric_df.index = metric_df.index.str.replace(f'{metric}_','')
	metric_df = metric_df.reset_index().reindex(labels=new_sort_order)
	metric_df = pd.concat([all_df.reset_index().head(n=2), metric_df], axis=0)
	return metric_df

# write to excel
writer = pd.ExcelWriter(join(eval_res_dir, f'all_models.xlsx'), engine="xlsxwriter")
tot_acc_all.to_excel(writer, index=False, sheet_name='all_data')
workbook = writer.book
#worksheet = writer.sheets['all_data']
(max_row, max_col) = tot_acc_all.shape
for m_idx, metric in enumerate(metrics):
	col_idx = m_idx + 1
	if metric == 'len':
		writer.sheets['all_data'].conditional_format(1, m_idx*4+1, max_row, (m_idx+1)*4+1, {'type': '3_color_scale', 'min_color': "#63BE7B", 'mid_color': "#FFEB84", 'max_color': "#F8696B", 'criteria': '<'})
	else:
		writer.sheets['all_data'].conditional_format(1, m_idx*4+1, max_row, (m_idx+1)*4+1, {'type': '3_color_scale', 'min_color': "#F8696B", 'mid_color': "#FFEB84", 'max_color': "#63BE7B", 'criteria': '<'})

# write per-image top1 accuracy and top1 confidence for all models
for metric in metrics:
	for i in qrange:
		sheet_name = f'By_image_A{i}_{metric}'
		stats_by_image[f'A{i}_{metric}'].to_excel(writer, index=False, sheet_name=sheet_name)
	stats_by_image[f'avg_{metric}'].to_excel(writer, index=False, sheet_name=f'By_image_avg_{metric}')

# write dataframes for each metric and attribute to a separate sheet
for metric in metrics:
	for attr, attr_df in tot_stats_by_attr.items():
		sheet_name = attr
		attr_metric_df = get_metric_df(attr_df, attr.split('--')[1], sort_order)

		if 'Shutter_speed' in attr:
			shutter_speed = attr_metric_df.iloc[1, 1:].to_list()
			# -1 is to keep the index in the 0th column
			shutter_speed_float = [-1] + [eval(str(x)) for x in shutter_speed]
			sort_idx = np.argsort(shutter_speed_float)
			attr_metric_df = attr_metric_df.iloc[:, sort_idx]

		attr_metric_df.to_excel(writer, index=False, sheet_name=sheet_name)
		(max_row, max_col) = attr_metric_df.shape
		# conditional format on the whole table
		if metric == 'acc':
			writer.sheets[sheet_name].conditional_format(3, 1, max_row, max_col, {'type': '3_color_scale', 'min_color': "#F8696B", 'mid_color': "#FFEB84", 'max_color': "#63BE7B", 'criteria': '<'})
		else:
			writer.sheets[sheet_name].conditional_format(3, 1, max_row, max_col, {'type': '3_color_scale', 'min_color': "#63BE7B", 'mid_color': "#FFEB84", 'max_color': "#F8696B", 'criteria': '<'})			

writer.close()

print(pd.DataFrame.from_dict(tot_acc_all).sort_values(by='avg_acc', ascending=False))