import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
from scipy import stats
import time
import os
from PIL import Image
from math import log10, log2
from matplotlib_venn import venn2
import operator
from mpl_toolkits.axes_grid1 import ImageGrid

plt.rcParams.update({'font.size': 12})

# a list of 1-stop sampled camera settings
shutter=['1/8000', '1/4000', '1/2000', '1/1000', '1/500', '1/250', '1/125', '1/60', '1/30', '1/15', '1/8', '1/4', '0.5', '1', '2', '4', '8', '15', '30', '60']
shutter_f = [round(eval(x), 5) for x in shutter]
iso=[12, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800, 409600]
fnumber=[1, 1.4, 2, 4, 5.6, 8, 11, 17, 22, 32, 45]

# list of all objects
obj_classes=['cup', 'laptop', 'phone', 'keyboard', 'comic book', 'backpack', 'mouse', 'remote', 'water bottle', 'tie']

def draw_error_band(ax, x, y, err, **kwargs):
    # Calculate normals via centered finite differences (except the first point
    # which uses a forward difference and the last point which uses a backward
    # difference).
    dx = np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]])
    dy = np.concatenate([[y[1] - y[0]], y[2:] - y[:-2], [y[-1] - y[-2]]])
    l = np.hypot(dx, dy)
    nx = dy / l
    ny = -dx / l

    # end points of errors
    xp = x + nx * err
    yp = y + ny * err
    xn = x - nx * err
    yn = y - ny * err

    vertices = np.block([[xp, xn[::-1]],
                         [yp, yn[::-1]]]).T
    codes = np.full(len(vertices), Path.LINETO)
    codes[0] = codes[len(xp)] = Path.MOVETO
    path = Path(vertices, codes)
    ax.add_patch(PathPatch(path, **kwargs))



class VisResults():
	
	def __init__(self, task='image_classification'):

		self.task = task

		self.paper_table_path = 'paper/paper_tables.xlsx'

		self.paper_table_df = pd.read_excel(self.paper_table_path, sheet_name=self.task)

		# filter the results only by the models to be analyzed
		self.data = {}

		self.annot_data_path = f'annotations/sensor_bias_data_v5.xlsx'
		self.human_data_path = f'eval_results/human/all_data_v5.xlsx'

		if task in ['image_classification', 'object_detection', 'vqa']:
			self.paper_table_df = self.paper_table_df[self.paper_table_df['plot'] == 'x']
			self.eval_data_path = f'eval_results/{self.task}/all_models_data_v5.xlsx'
			self.models = self.paper_table_df['Model id']
			self.model_labels = self.paper_table_df['Model name']
			
	
	def plot_top1_vs_AP(self):
		img_class_df = pd.read_excel(self.paper_table_path, sheet_name='image_classification')
		temp = pd.merge(self.paper_table_df, img_class_df, how='inner', left_on='Backbone', right_on='Model id')
		top1_vs_AP_df = temp[['Model name_x', 'Backbone', 'COCO AP', 'ImageNet_top1_acc']]
		fig, ax = plt.subplots(figsize=(5,5))
		top1 = top1_vs_AP_df['ImageNet_top1_acc']
		ap = top1_vs_AP_df['COCO AP']
		ax.scatter(top1, ap, marker='o', color='b', alpha=0.5)
		#plt.show()

		print('Pearson r', np.corrcoef(top1, ap))

		res=stats.pearsonr(top1, ap)
		print(res)
		ci = res.confidence_interval(confidence_level=0.9)
		print(ci)


	def line_metric_by_attr(self, attr='EV_offset', metric='top1_acc', human=False):

		fontdict={'alpha': 1, 'fontsize':'xx-small'}

		fig_all, ax_all = plt.subplots(figsize=(3,4))
		records = []

		# get results by attribute and metric
		if self.task == 'vqa':
			data_df = pd.read_excel(self.eval_data_path, sheet_name=f'{attr}--{metric}')
		else:
			data_df = pd.read_excel(self.eval_data_path, sheet_name=f'{attr}_{metric}')

		if human:
			if 'acc1' in metric:
				human_data_df = pd.read_excel(self.human_data_path, sheet_name=f'{attr}--{metric[:-1]}')
			else:
				human_data_df = pd.read_excel(self.human_data_path, sheet_name=f'{attr}--{metric}')

		attr_vals = data_df[data_df['index'] == attr].values.flatten().tolist()[1:]

		data_df = data_df[data_df['index'].isin(self.models)].reset_index(drop=True)

		cols = data_df.columns.to_list()[1:]
		if attr in ['EV_offset']:
			cols = [int(x) for x in cols]
			attr_vals = [str(int(x)) for x in attr_vals]
		else:
			sort_order = list(data_df[cols].mean(axis=0).sort_values().index)
			cols_mapper = {y:int(x) for x,y in enumerate(sort_order)}
			data_df = data_df[['index']+sort_order].rename(cols_mapper, axis=1)
			attr_vals = [attr_vals[x] for x in cols_mapper.keys()]

		data_df = self.paper_table_df[['Model name', 'Model id']].merge(data_df, how='inner', left_on='Model id', right_on='index')

		# get the overall results
		# this determines the sorting order in all plots
		all_data_df = pd.read_excel(self.eval_data_path, sheet_name='all_data')

		if 'LRP' in metric:
			data_df = all_data_df.merge(data_df, left_on='Model', right_on='Model id').sort_values(by='oLRP', ascending=False).reset_index(drop=True)
		elif 'A' in metric and 'acc' in metric:
			m = 'avg_' + metric.split('_')[1]
			data_df = all_data_df.merge(data_df, left_on='Model', right_on='Model id').sort_values(by=m, ascending=False).reset_index(drop=True)
		elif 'len' in metric:
			data_df = all_data_df.merge(data_df, left_on='Model', right_on='Model id').sort_values(by='avg_acc1', ascending=False).reset_index(drop=True)
		else:
			data_df = all_data_df.merge(data_df, left_on='Model', right_on='Model id').sort_values(by=metric, ascending=False).reset_index(drop=True)

		cmap = plt.get_cmap('tab20')
		num_colors = len(data_df)+1
		values = np.linspace(0, 1, num_colors)
		colors = [cmap(value) for value in values]

		x_0 = 0.6
		for idx, row in data_df.iterrows():
			p = ax_all.plot(row[len(row)-len(cols):], marker='o', color=colors[idx], 
						markerfacecolor=colors[idx], markersize=1, linewidth=0.5, aa=True)
			model_label = row['Model name']
			row = row.to_list()
			x = x_0-idx*0.03
			y = len(attr_vals)-0.5
			ax_all.text(y, x, model_label, **fontdict, color=colors[idx])

		if human:
			ax_all.plot(human_data_df[cols].iloc[-1,:], color='red', linestyle='-', marker='x', markersize=2, linewidth=1, aa=True)
			x = human_data_df[cols].iloc[-1,-5]
			y = len(attr_vals)-4.7
			ax_all.text(y, x, 'human', **fontdict, color='red')		

		ax_all.set_xticks(list(range(len(attr_vals))), labels=attr_vals)
		#ax_all.set_title(f'{metric} by {attr}')
		ax_all.set_xlabel(attr.replace('_', ' '))
		if metric == 'top1_acc':
			ax_all.set_ylabel('Top-1 (%)')
		elif 'acc1' in metric:
			ax_all.set_ylabel('Hard acc (%)')
		else:
			ax_all.set_ylabel(metric)

		ax_all.grid(axis='y', linewidth=0.25)
		ax_all.spines[['right', 'top']].set_visible(False)
		if 'len' not in metric:
			ax_all.set_ylim([0, 1])

		fig_all.savefig(f'paper/images/{self.task}_{attr}_{metric}_line.pdf', dpi=300, bbox_inches='tight')

	def box_metric_by_attr(self, attr='EV_offset', metric='top1_acc', data_metric=None, human=False, ps=None):

		fig_box, ax_box = plt.subplots(figsize=(8,4))
		
		# get the overall results
		# this determines the sorting order in all plots
		all_data_df = pd.read_excel(self.eval_data_path, sheet_name='all_data')

		# get results by attribute and metric
		if self.task == 'vqa':
			eval_data_df = pd.read_excel(self.eval_data_path, sheet_name=f'{attr}--{metric}')
		else:
			eval_data_df = pd.read_excel(self.eval_data_path, sheet_name=f'{attr}_{metric}')

		if human:
			if 'acc1' in metric:
				human_data_df = pd.read_excel(self.human_data_path, sheet_name=f'{attr}--{metric[:-1]}')
			else:
				human_data_df = pd.read_excel(self.human_data_path, sheet_name=f'{attr}--{metric}')

		attr_labels = list(eval_data_df[eval_data_df['index'] == attr].values[0])[1:]
		
		cols = eval_data_df.columns.to_list()[1:]
		if attr in ['EV_offset']:
			cols = [int(x) for x in cols]
		attr_vals = eval_data_df[eval_data_df['index'] == attr].values.flatten().tolist()[1:]
		attr_counts = eval_data_df[eval_data_df['index'] == 'Num_images'].values.flatten().tolist()[1:] # different number of images per attribute

		# combine with paper table to get the labels and imagenet accuracy
		if data_metric is None:
			data_df = self.paper_table_df[['Model name', 'Model id']].merge(eval_data_df, how='inner', left_on='Model id', right_on='index')
		else:
			data_df = self.paper_table_df[['Model name', 'Model id', data_metric]].merge(eval_data_df, how='inner', left_on='Model id', right_on='index')

		if ps is not None:
			data_df = data_df.merge(ps, how='inner', left_on='Model id', right_index=True)

		if 'LRP' in metric:
			data_df = all_data_df.merge(data_df, left_on='Model', right_on='Model id').sort_values(by='oLRP', ascending=False).reset_index(drop=True)
		elif 'A' in metric and 'acc' in metric:
			m = 'avg_' + metric.split('_')[1]
			data_df = all_data_df.merge(data_df, left_on='Model', right_on='Model id').sort_values(by=m).reset_index(drop=True)
		elif 'len' in metric:
			data_df = all_data_df.merge(data_df, left_on='Model', right_on='Model id').sort_values(by='avg_acc1').reset_index(drop=True)
		else:
			data_df = all_data_df.merge(data_df, left_on='Model', right_on='Model id').sort_values(by=metric).reset_index(drop=True)
		
		if attr in ['Class', 'Category']:
			data_df = data_df[cols]
			data_df = data_df.transpose()
			sort_order = [int(x) for x in data_df.mean(axis=1).sort_values(ascending='LRP' in metric).index]
			data_df = data_df.reindex(sort_order).reset_index(drop=True)
			attr_labels = [attr_labels[i] for i in sort_order]
			for idx, row in data_df.iterrows():
				ax_box.boxplot(row, positions=[idx], showmeans=True)
			ax_box.set_xticks([x for x in range(len(attr_labels))], labels=attr_labels, rotation=45, ha='right',rotation_mode='anchor') 
			ax_box.set_ylim(0, 1)
		else:
			model_labels = data_df['Model name'].to_list() + ['human']*human
			for idx, row in data_df[cols].iterrows():
				values = []
				for c, v in zip(attr_counts, row):
					values.extend([v]*int(c))

				ax_box.boxplot(values, positions=[idx], showmeans=True)
			
			if human:
				values = []
				counts = human_data_df.iloc[0,:].to_list()
				means = human_data_df.iloc[-1,:].to_list()
				for c, m in zip(counts, means):
					values.extend([m]*int(c))
				ax_box.boxplot(values, positions=[idx+1], showmeans=True)

			ax_box.set_xticks([x for x in range(len(model_labels))], labels=model_labels, rotation=35, ha='right',rotation_mode='anchor')

			if data_metric is not None:
				ax_box.plot([x for x in range(len(model_labels))], data_df[data_metric], 'rx')
		
			if ps is not None:
				ax_box.plot([x for x in range(len(model_labels)-human)], data_df['PS'], 'bo')

		ax_box.grid(axis='y', linewidth=0.25)
		if metric == 'top1_acc':
			ax_box.set_ylabel('Top-1 (%)')
		elif metric == 'avg_len':
			ax_box.set_ylabel('Avg. len (char)')
		fig_box.subplots_adjust(bottom=0.3)
		fig_box.savefig(f'paper/images/{self.task}_{attr}_{metric}_box.pdf', dpi=300, bbox_inches='tight')
		#plt.show()

	def bar_vqa_acc(self):
		# plot accuracy vs hard accuracy (with hallucinations)
		qrange = range(3, 7)
		data_df = pd.read_excel(self.eval_data_path, sheet_name='all_data').sort_values(by='avg_acc1')
		# combine with paper table to get the labels
		data_df = data_df.merge(self.paper_table_df[['Model name', 'Model id']], 
									 how='inner', right_on='Model id', left_on='Model')
		model_labels = data_df['Model name'].to_list()
		cols = [f'A{i}_acc' for i in qrange]
		metric = ['avg_acc'] + cols
		human_data_df = pd.read_excel(self.human_data_path, sheet_name=f'all_data')[cols]
		human_data_df['avg_acc'] = human_data_df.iloc[0].mean()
		for m in metric:
			fig_bar, ax_bar = plt.subplots(figsize=(8,4))
			acc1 = data_df[m+'1']
			acc = data_df[m] - data_df[m+'1']
			ax_bar.bar(model_labels, acc1, label='Hard acc (%)', color='g')
			ax_bar.bar(model_labels, acc, bottom=acc1, label='Soft acc (%)', color='b')
			ax_bar.set_xticks([x for x in range(len(model_labels))], labels=model_labels, rotation=35, ha='right', rotation_mode='anchor')
			fig_bar.subplots_adjust(bottom=0.3)
			ax_bar.legend()
			ax_bar.set_ylim([0,1])
			ax_bar.axhline(y=human_data_df.at[0,m], color='r', linewidth=0.75, linestyle='--')
			ax_bar.tick_params(labelsize=15)
			fig_bar.savefig(f'paper/images/{self.task}_{m}1_bar.pdf', dpi=300, bbox_inches='tight')

	# only for vqa
	def box_all(self, metric='acc'):
		qrange = range(3, 7)
		fig_box, ax_box = plt.subplots(figsize=(8,4))
		if 'len' in metric:
			data_df = pd.read_excel(self.eval_data_path, sheet_name=f'all_data').sort_values(by=f'avg_acc1')
		else:
			data_df = pd.read_excel(self.eval_data_path, sheet_name=f'all_data').sort_values(by=f'avg_{metric}')
		# combine with paper table to get the labels and imagenet accuracy
		data_df = data_df.merge(self.paper_table_df[['Model name', 'Model id']], 
									 how='inner', right_on='Model id', left_on='Model')
		cols = [f'A{i}_{metric}' for i in qrange]
		model_labels = data_df['Model name'].to_list()
		ax_box.boxplot(data_df[cols].transpose(), labels=model_labels, showmeans=True)
		ax_box.set_xticks([x+1 for x in range(len(model_labels))], labels=model_labels, rotation=35, ha='right', rotation_mode='anchor')
		ax_box.tick_params(labelsize=15)
		fig_box.subplots_adjust(bottom=0.3)
		fig_box.savefig(f'paper/images/{self.task}_all_{metric}_box.pdf', dpi=300, bbox_inches='tight')


	# performance on the most frequent training settings vs least frequent
	def ev_vs_freq(self, metric='top1_acc'):
		model_list = self.paper_table_df[self.paper_table_df['plot'] == 'x']['Model id'].to_list()

		# load eval data per image for all models
		print('Loading per-image results')
		all_data_df = pd.read_excel(self.eval_data_path, sheet_name=f'By_image_{metric}')
		all_data_df = all_data_df[['Path'] + model_list]

		# load annotations to get image info
		print('Loading annotations')
		annot_data_df = pd.read_excel(self.annot_data_path)
		data_df = pd.merge(all_data_df, annot_data_df, how='inner', on='Path')
		data_df.drop(columns=['Unnamed: 0'], inplace=True)

		fontdict={'alpha': 0.75, 'color': 'gray', 'fontsize':'x-small'}

		fig_all, ax_all = plt.subplots(figsize=(3,4))

		num_missed_df = pd.DataFrame()
		num_missed_df.index = model_list

		num_obj_df = pd.DataFrame()
		num_obj_df.index = model_list

		cols = model_list + ['Object_id']
		for ev_idx, ev_off in enumerate(range(14)):
			ev_df = data_df[data_df['EV_offset'] == ev_off-5]
			num_objects = len(ev_df['Object_id'].unique())


			mean = ev_df[cols].groupby(by='Object_id').mean()
			std = ev_df[cols].groupby(by='Object_id').std()
			temp = std/mean
			num_missed = temp > 1
			num_missed = num_missed.sum(axis=0)

			num_missed_df[ev_off-5] = num_missed
			num_obj_df[ev_off-5] = num_objects

		total_missed_df = num_missed_df.sum(axis=1)/num_obj_df.sum(axis=1)

		perc_missed_df = num_missed_df/num_obj_df
		cols = num_missed_df.columns.to_list()[1:]
		ax_all.plot(perc_missed_df[cols].transpose(), marker='o', markersize=1, linewidth=0.5, aa=True)


		# change model id to a more readable format
		model_name_df = pd.merge(perc_missed_df, self.paper_table_df[['Model id', 'Model name']], how='inner', left_index=True, right_on='Model id')[['Model id', 'Model name']]
		perc_missed_df.index = model_name_df['Model name']

		for idx, (model_label, row) in enumerate(perc_missed_df.iterrows()):
			x = row.to_list()[-1]
			y = len(row)-6
			ax_all.text(y, x, model_label, **fontdict)

		ax_all.set_xticks(list(range(-5,9)), labels=[str(x-5) for x in range(14)])
		#ax_all.set_title(f'{metric} by {attr}')
		ax_all.set_ylabel('Parameter sensitivity (PS)')
		ax_all.set_xlabel('EV offset')
		ax_all.grid(axis='y', linewidth=0.25)
		ax_all.spines[['right', 'top']].set_visible(False)
		if 'top1_acc' in metric:
			ax_all.set_ylim([0, 0.5])
		elif 'oLRP' in metric:
			ax_all.set_ylim([0, 0.35])

		fig_all.savefig(f'paper/images/{self.task}_{metric}_miss_line.pdf', dpi=300, bbox_inches='tight')
		return total_missed_df.to_frame('PS')



	def plot_model_properties(self, metric='top1_acc'):
		
		# get the overall results
		# this determines the sorting order in all plots
		all_data_df = pd.read_excel(self.eval_data_path, sheet_name='all_data')
		data_df = self.paper_table_df[['Model name', 'Model id', 'Model size (M params)', 'Training data size (M samples)']].merge(all_data_df, how='inner', left_on='Model id', right_on='Model')

		m = data_df[metric]
		model_size = data_df['Model size (M params)']
		data_size = data_df['Training data size (M samples)']

		fig, ax = plt.subplots(figsize=(6,3))

		fontdict={'alpha': 1, 'color': 'gray', 'fontsize':'small'}

		ax.scatter(model_size, m, s=[15 for x in data_size], alpha=0.75)
		ax.set_xscale('log')
		#ax.set_title('Model size vs. top-1 acc')
		ax.set_xlabel('Model size (M params)')
		ax.set_ylim([0,1])
		if 'top1' in metric:
			ax.set_ylabel('Top-1 (%)')
		else:
			ax.set_ylabel(metric)

		for ms, acc, ml in zip(model_size, m, self.model_labels):
			ax.text(ms, acc, ml, **fontdict)

		fig.savefig(f'paper/images/{self.task}_model_size_vs_{metric}.pdf', dpi=300, bbox_inches='tight')

		fig, ax = plt.subplots(figsize=(6,3))

		ax.scatter(data_size, m, s=[10 for x in model_size], alpha=0.75)
		ax.set_xscale('log')
		ax.set_xlabel('Training data size (M samples)')
		ax.set_ylabel(metric)
		ax.set_ylim([0,1])

		for ds, acc, ml in zip(data_size, m, self.model_labels):
			ax.text(ds, acc, ml, **fontdict)
		fig.savefig(f'paper/images/{self.task}_data_size_vs_{metric}.pdf', dpi=300, bbox_inches='tight')

		fig, ax = plt.subplots(figsize=(6,3))

		ax.scatter(model_size, data_size, s=m*100, alpha=0.5)
		ax.set_xscale('log')
		#ax.set_ylabel('Training data size (M samples)')
		ax.set_xlabel('Model size (M params)')
		ax.set_title('Model size vs. training data')
		ax.set_ylim([0,1])

		fig.savefig(f'paper/images/{self.task}_model_size_vs_data_size.pdf', dpi=300, bbox_inches='tight')	
		
		#plt.show()


def plot_sample_images():
	data_df = pd.read_excel('annotations/SNAP.xlsx')
	img_df = data_df[(data_df['Lux']==1000) & (data_df['EV_offset'] == 1)].drop_duplicates(subset=['Object_name'])
	img_by_class = img_df[['Path', 'Category']].groupby(by='Category').agg(lambda x: list(x))
	image_paths = reduce(operator.concat, img_by_class['Path'].to_list())
	ncols = 10
	nrows = 10

	fig = plt.figure(figsize=(8, 6))
	grid = ImageGrid(fig, 111, nrows_ncols=(10, 10), axes_pad=0.03)

	for im_idx, (ax, im_path) in enumerate(zip(grid, image_paths)):
		im = Image.open('data_v5/'+im_path)
		h = im.height
		w = im.width
		im.thumbnail((int(h/4), int(w/4)), Image.LANCZOS)
		ax.imshow(np.asarray(im))
		ax.set_xticks([])
		ax.set_yticks([])

		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['left'].set_visible(False)

		if im_idx % ncols == 0:
			class_label = im_path.split('/')[0].split('_')[1]
			ax.set_ylabel(class_label, fontsize=6)
			

	plt.show()
	fig.savefig(f'paper/images/SNAP_samples.pdf', dpi=300, bbox_inches='tight')



# run from the root sensor_bias directory
# python3 scripts/eval/generate_paper_figures.py

plot_sample_images()

visres = VisResults(task='image_classification')
visres.line_metric_by_attr(attr='EV_offset', metric='top1_acc')
top1_ps = visres.ev_vs_freq(metric='top1_acc')
visres.box_metric_by_attr(attr='EV_offset', metric='top1_acc', data_metric='ImageNet_top1_acc', ps=top1_ps)
visres.plot_model_properties(metric='top1_acc')


visres = VisResults(task='object_detection')

visres.line_metric_by_attr(attr='EV_offset', metric='AP')
visres.line_metric_by_attr(attr='EV_offset', metric='oLRP')
visres.line_metric_by_attr(attr='EV_offset', metric='oLRP FP')
visres.line_metric_by_attr(attr='EV_offset', metric='oLRP Loc')
visres.line_metric_by_attr(attr='EV_offset', metric='oLRP FN')

ps_oLRP = visres.ev_vs_freq(metric='oLRP')
ps_AP = visres.ev_vs_freq(metric='AP')
ps_oLRP_Loc = visres.ev_vs_freq(metric='oLRP Loc')
ps_oLRP_FN = visres.ev_vs_freq(metric='oLRP FN')
ps_oLRP_FP = visres.ev_vs_freq(metric='oLRP FP')

visres.box_metric_by_attr(attr='EV_offset', metric='oLRP', ps=ps_oLRP)
visres.box_metric_by_attr(attr='EV_offset', metric='AP', data_metric='COCO AP')
visres.box_metric_by_attr(attr='EV_offset', metric='oLRP FP', ps=ps_oLRP_FP)
visres.box_metric_by_attr(attr='EV_offset', metric='oLRP Loc', ps=ps_oLRP_Loc)
visres.box_metric_by_attr(attr='EV_offset', metric='oLRP FN', ps=ps_oLRP_FN)


visres.plot_model_properties(metric='AP')
visres.plot_model_properties(metric='oLRP')
visres.plot_model_properties(metric='oLRP FP')
visres.plot_model_properties(metric='oLRP Loc')

visres = VisResults(task='vqa')
visres.bar_vqa_acc()

ps_acc1 = visres.ev_vs_freq(metric='avg_acc1')
ps_acc = visres.ev_vs_freq(metric='avg_acc')

visres.box_all(metric='acc1')
visres.box_all(metric='len')

visres.line_metric_by_attr(attr='EV_offset', metric='avg_acc', human=True)
visres.line_metric_by_attr(attr='EV_offset', metric='avg_acc1', human=True)
visres.line_metric_by_attr(attr='EV_offset', metric='avg_len', human=True)


visres.box_metric_by_attr(attr='EV_offset', metric='avg_acc', data_metric=None, human=True)
visres.box_metric_by_attr(attr='EV_offset', metric='avg_len', data_metric=None, human=True)
visres.box_metric_by_attr(attr='EV_offset', metric='avg_acc1', data_metric=None, human=True)

for i in range(1, 5):
	ps_acc1 = visres.ev_vs_freq(metric=f'A{i}_acc1')
	visres.box_metric_by_attr(attr='EV_offset', metric=f'A{i}_acc1', data_metric=None, ps=ps_acc1, human=True)

for i in range(1, 5):
	visres.line_metric_by_attr(attr='EV_offset', metric=f'A{i}_acc', human=True)
	visres.line_metric_by_attr(attr='EV_offset', metric=f'A{i}_acc1', human=True)