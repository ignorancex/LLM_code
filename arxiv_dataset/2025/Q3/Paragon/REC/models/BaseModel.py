import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
from torch.nn.utils.rnn import pad_sequence
from typing import List
import json
from utils import utils
from reader import BaseReader
from functools import reduce
from time import time
class BaseModel(nn.Module):
	reader, runner = None, None  # choose helpers in specific model classes
	extra_log_args = []

	@staticmethod
	def init_weights(m):
		if 'Linear' in str(type(m)):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)
			if m.bias is not None:
				nn.init.normal_(m.bias, mean=0.0, std=0.01)
		elif 'Embedding' in str(type(m)):
			nn.init.normal_(m.weight, mean=0.0, std=0.01)

	def __init__(self, args, corpus: BaseReader):
		super(BaseModel, self).__init__()
		self.device = args.device
		self.model_path = args.model_path
		self.buffer = args.buffer
		self.optimizer = None
		self.check_list = list()  # observe tensors in check_list every check_epoch

	"""
	Key Methods
	"""
	def _define_params(self):
		pass

	def forward(self, feed_dict: dict) -> dict:
		"""
		:param feed_dict: batch prepared in Dataset
		:return: out_dict, including prediction with shape [batch_size, n_candidates]
		"""
		pass

	def loss(self, out_dict: dict) -> torch.Tensor:
		pass

	"""
	Auxiliary Methods
	"""
	def customize_parameters(self) -> list:
		# customize optimizer settings for different parameters
		weight_p, bias_p = [], []
		for name, p in filter(lambda x: x[1].requires_grad, self.named_parameters()):
			if 'bias' in name:
				bias_p.append(p)
			else:
				weight_p.append(p)
		optimize_dict = [{'params': weight_p}, {'params': bias_p, 'weight_decay': 0}]
		return optimize_dict

	def save_model(self, model_path=None):
		if model_path is None:
			model_path = self.model_path
		utils.check_dir(model_path) # check
		torch.save(self.state_dict(), model_path)
		# logging.info('Save model to ' + model_path[:50] + '...')
   
	def load_model(self, model_path=None):
		if model_path is None:
			model_path = self.model_path
		self.load_state_dict(torch.load(model_path))
		logging.info('Load model from ' + model_path)

	def count_variables(self) -> int:
		total_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
		return total_parameters

	def actions_after_train(self):  # e.g., save selected parameters
		pass

	"""
	Define Dataset Class
	"""
	class Dataset(BaseDataset):
		def __init__(self, model, corpus, phase: str , sample_ratio = 1.0):
			self.model = model  # model object reference
			self.corpus = corpus  # reader object reference
			self.phase = phase  # train / dev / test
			self.sample_ratio = sample_ratio
			self.buffer_dict = dict()
			#self.data = utils.df_to_dict(corpus.data_df[phase])#this raise the VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences warning
			self.data = corpus.data_df[phase].to_dict('list')
			if phase != 'train' and sample_ratio < 1.0:
				self.sample_data(sample_ratio)

		def sample_data(self, ratio):
			num_samples = int(len(self) * ratio)
			indices = np.random.choice(len(self), num_samples, replace=False)
			for key in self.data:
				self.data[key] = [self.data[key][i] for i in indices]

			# 检测
			# data= corpus.data_df["train"].to_dict('list')
			# neg_items = data['neg_items'][0]
			# ↑ DataFrame is not compatible with multi-thread operations

		def __len__(self):
			if type(self.data) == dict:
				for key in self.data:
					return len(self.data[key])
			return len(self.data)

		def __getitem__(self, index: int) -> dict:
			if self.model.buffer and self.phase != 'train':
				return self.buffer_dict[index]
			
			return self._get_feed_dict(index)

		# ! Key method to construct input data for a single instance
		def _get_feed_dict(self, index: int) -> dict:
			pass

		# Called after initialization
		def prepare(self):
			if self.model.buffer and self.phase != 'train':
				for i in tqdm(range(len(self)), leave=False, desc=('Prepare ' + self.phase)):
					self.buffer_dict[i] = self._get_feed_dict(i)

		# Called before each training epoch (only for the training dataset)
		def actions_before_epoch(self):
			pass

		# Collate a batch according to the list of feed dicts
		def collate_batch(self, feed_dicts: List[dict]) -> dict:
			feed_dict = dict()
			for key in feed_dicts[0]:
				if isinstance(feed_dicts[0][key], np.ndarray):
					tmp_list = [len(d[key]) for d in feed_dicts]
					if any([tmp_list[0] != l for l in tmp_list]):
						stack_val = np.array([d[key] for d in feed_dicts], dtype=object)
					else:
						stack_val = np.array([d[key] for d in feed_dicts])
				else:
					stack_val = np.array([d[key] for d in feed_dicts])
				if stack_val.dtype == object:  # inconsistent length (e.g., history)
					feed_dict[key] = pad_sequence([torch.from_numpy(x) for x in stack_val], batch_first=True)
				else:
					feed_dict[key] = torch.from_numpy(stack_val)
			feed_dict['batch_size'] = len(feed_dicts)
			feed_dict['phase'] = self.phase
			return feed_dict

class GeneralModel(BaseModel):
	reader, runner = 'BaseReader', 'BaseRunner'

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.user_num = corpus.n_users
		self.item_num = corpus.n_items
		self.num_neg = args.num_neg
		self.dropout = args.dropout
		self.test_all = args.test_all
		if args.dataset == 'MovieLens_1M':
			self.item_multihot_mapping = json.load(open(f'corpus/{args.dataset}/ML_1MTOPK/newid2multihot.json', 'r'))
		elif args.dataset == 'Grocery_and_Gourmet_Food':
			self.item_multihot_mapping = json.load(open(f'corpus/{args.dataset}/{args.dataset}/newid2multihot.json', 'r'))
		elif args.dataset == 'MIND_Large':
			self.item_multihot_mapping = json.load(open(f'corpus/{args.dataset}/MINDTOPK/newid2multihot.json', 'r'))
		elif args.dataset == 'xxx':
			self.item_multihot_mapping = json.load(open(f'corpus/{args.dataset}/xxx_TOPK/newid2multihot.json', 'r'))
		else:
			logging.error('No such multihot embedding of dataset: {}'.format(args.dataset))
			
		self.item_multihot_embeddings = self.multihot_embedding(self.item_multihot_mapping)

	def multihot_embedding(self, item_multihot_mapping):
		num_items = max(int(k) for k in self.item_multihot_mapping.keys()) + 1
		embedding_dim = len(next(iter(self.item_multihot_mapping.values())))
		# Create an empty tensor to store all the multithot vectors
		item_embeddings = torch.zeros((num_items, embedding_dim), dtype=torch.float32, device='cuda')
		for key, value in self.item_multihot_mapping.items():
			item_embeddings[int(key)] = torch.tensor(value, dtype=torch.float32)
		return item_embeddings
	
	def loss(self, out_dict: dict) -> torch.Tensor:
		"""
		BPR ranking loss with optimization on multiple negative samples (a little different now to follow the paper ↓)
		"Recurrent neural networks with top-k gains for session-based recommendations"
		:param out_dict: contain prediction with [batch_size, -1], the first column for positive, the rest for negative
		:return:
		"""
		predictions = out_dict['prediction']
		pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
		neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
		loss = -(((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1)).clamp(min=1e-8,max=1-1e-8).log().mean()
		# neg_pred = (neg_pred * neg_softmax).sum(dim=1)
		# loss = F.softplus(-(pos_pred - neg_pred)).mean()
		# ↑ For numerical stability, use 'softplus(-x)' instead of '-log_sigmoid(x)'
		return loss
	
	def fairness_loss(self, out_dict: dict, batch_user) -> torch.Tensor:
		"""
		Compute the fairness-aware regularization term based on the difference in average loss
		between male and female groups, as shown in the provided fairness objective.
		:param out_dict: Dictionary containing predictions and gender labels.
		:return: Total loss with fairness regularization.
		"""
		# Extract losses for the batch
		uids = batch_user.cpu().numpy().tolist()
		if self.dataset_name == 'MovieLens_1M':
			genders = [self.uid2gender[str(uid)] for uid in uids]# Assuming gender is a tensor where 0 = male and 1 = female
		predictions = out_dict['prediction']
		pos_pred, neg_pred = predictions[:, 0], predictions[:, 1:]
		neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
		batch_loss = -(((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1)).clamp(min=1e-8, max=1-1e-8).log()
		genders = genders = torch.tensor(genders, device=batch_loss.device) 
		# Separate losses by gender groups
		male_losses = batch_loss[genders == 0]
		female_losses = batch_loss[genders == 1]
		# Calculate mean losses for each group
		male_loss_mean = male_losses.mean() if len(male_losses) > 0 else torch.tensor(0.0, device=batch_loss.device)
		female_loss_mean = female_losses.mean() if len(female_losses) > 0 else torch.tensor(0.0, device=batch_loss.device)
		# Calculate fairness regularization term
		if self.is_fair == 0:
			fairness_reg = 0.1 * max(male_loss_mean, female_loss_mean) + 0.9 * min(female_loss_mean, male_loss_mean)
		else:
			fairness_reg = 0.9 * max(male_loss_mean, female_loss_mean) + 0.1 * min(female_loss_mean, male_loss_mean)
		# # Set lambda for the fairness regularization term
		# lambda_fairness = 0.1  # You can adjust this value as needed
		# # Compute the total loss with fairness term
		# total_loss = batch_loss.mean() + lambda_fairness * fairness_reg
		return fairness_reg
	
	def diversity_loss(self, out_dict: dict) -> torch.Tensor:
		start_time = time()
		predictions = out_dict['prediction']  # [B, N] preds 
		ids = out_dict['item_id']  # [B, N] item ids
		category = self.item_multihot_embeddings[ids]

		# Calculate the difference between the predicted values
		predictions_1 = predictions.unsqueeze(2)  # [B, N, 1]
		predictions_2 = predictions.unsqueeze(1)  # [B, 1, N]
		sub_prediction = (predictions_2 - predictions_1)  # [B, N, N]
		
		# Use broadcast to avoid duplicate tensors
		R = 0.5 + torch.sigmoid(sub_prediction / 0.1).sum(dim=2)  # [B, N]
		
		# Vectorization operations for processing categorical data
		category_expanded = category.unsqueeze(2)  # [B, N, 1, M]
		sub_prediction_expanded = sub_prediction.unsqueeze(3)  # [B, N, N, 1]
		sigmoid_sub = torch.sigmoid(sub_prediction_expanded / 0.1)  # [B, N, N, 1]
		category_product = (category_expanded * sigmoid_sub).sum(dim=1) - 0.5 * category  # [B, N, M]
		decay_cat = torch.pow(0.5, category_product)  # [B, N, M]
		sum_cat = (category * decay_cat).sum(dim=2)  # [B, N]

		loss = -torch.sum(sum_cat / (torch.log(R + 1) / torch.log(torch.tensor(2.0, device='cuda')))) / predictions.shape[0]

		return loss


	@staticmethod
	def alpha_ndcg( output_dict: dict, id_multihot: dict[int, list], alpha, k, normaliztion) -> float:
		batch_predictions = output_dict["prediction"].cpu().detach().numpy()
		batch_ids = output_dict["item_id"].cpu().detach().numpy()
		a_ndcg_list = []
		category =  []
		for i in range(batch_ids.shape[0]): # For each user in the batch
   			# 1. Sort the scores
			sorted_indices = np.argsort(batch_predictions[i])[::-1]  # Descending arrangement
			sorted_ids = batch_ids[i][sorted_indices]
			# 2. Map according to the dictionary
			grade_list = np.zeros((len(sorted_ids), len(id_multihot[str(sorted_ids[0])])))  # :param grade_list: a np.ndarry with shape [#subtopic, #doc]
			for j, id_ in enumerate(sorted_ids):
				grade_list[j] = id_multihot[str(id_)]  # :param grade_list: a np.ndarry withshape [#subtopic, #doc]
				# grade_list, subtopics, k, grade_max = div_metric_param(grade_list, subtopics, k, grade_max)
			grade_list = np.transpose(np.array(grade_list))

			alpha, n_subtopics, n_docs = 1 - alpha, grade_list.shape[0], grade_list.shape[1]
			grade_list = (grade_list>0).astype(int)
			cum = reduce(lambda t,i:  [t[0]+(grade_list[:,i]),
									t[1]+np.sum(np.dot(np.power(alpha,t[0]), grade_list[:,i]))/np.log2(i+2)],
						range(k), [np.zeros(n_subtopics), 0])[1]
			a_ndcg_list.append(cum)
			result = [a / b for a, b in zip(a_ndcg_list, normaliztion)]
		return  np.mean(result)	 #np.mean(category)   Each batch gets a value
	
	@staticmethod
	def best_alpha_nDCG(output_dict: dict, id_multihot: dict[int, list], alpha, k):
		batch_predictions = output_dict["prediction"].cpu().detach().numpy()
		batch_ids = output_dict["item_id"].cpu().detach().numpy()
		score_batch = []
		for i in range(batch_ids.shape[0]): 
   	
			sorted_indices = np.argsort(batch_predictions[i])[::-1] # Descending arrangement
			sorted_ids = batch_ids[i][sorted_indices]
		
			grade_list = np.zeros((len(sorted_ids), len(id_multihot[str(sorted_ids[0])])))  # :param grade_list: a np.ndarry with shape [#subtopic, #doc]
			for j, id_ in enumerate(sorted_ids):
				grade_list[j] = id_multihot[str(id_)]
			alpha, n_subtopics, n_docs = 1 - alpha, grade_list.shape[0], grade_list.shape[1]
			grade_list = (grade_list > 0).astype(int)
			mask = np.zeros(n_docs)
			discount = np.zeros(n_subtopics)
			score, rank, score_list = 0, [], []
			for i in range(k):
				scores = np.matmul(grade_list.T, np.power(alpha, discount)) + mask
				r = np.argmax(scores) #if not reverse else np.argmin(scores)
				discount += grade_list[:,r]
				score += scores[r]/np.log2(i+2)
				score_list.append(score)
				rank.append(r)
				mask[r] = np.finfo(np.float32).min #if not reverse else np.finfo(np.float32).max
			score_batch.append(score)		
		return score_batch
	
	class Dataset(BaseModel.Dataset):
		def _get_feed_dict(self, index):
			user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
			if self.phase != 'train' and self.model.test_all:
				neg_items = np.arange(1, self.corpus.n_items)
			elif self.phase != 'train':
				neg_items = self.data['neg_items'][index] # negative items are pre-sampled 99个
			elif self.phase == 'train':
				neg_items = self.data['neg_items'][index][:self.model.num_neg] # 9个
			item_ids = np.concatenate([[target_item], neg_items]).astype(int)
			feed_dict = {
				'user_id': user_id,
				'item_id': item_ids
			}
			
			return feed_dict

		def actions_before_epoch(self):
			neg_items = np.random.randint(1, self.corpus.n_items, size=(len(self), self.model.num_neg))
			for i, u in enumerate(self.data['user_id']):
				clicked_set = self.corpus.train_clicked_set[u]  # neg items are possible to appear in dev/test set
				# clicked_set = self.corpus.clicked_set[u]  # neg items will not include dev/test set
				for j in range(self.model.num_neg):
					while neg_items[i][j] in clicked_set:
						neg_items[i][j] = np.random.randint(1, self.corpus.n_items)
			self.data['neg_items'] = neg_items

class SequentialModel(GeneralModel):
	reader = 'SeqReader'

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.history_max = args.history_max

	class Dataset(GeneralModel.Dataset):  # 

		def __init__(self, model, corpus, phase, sample_ratio=1.0):
			super().__init__(model, corpus, phase)
			idx_select = np.array(self.data['position']) > 0  # history length must be non-zero
			for key in self.data:
				self.data[key] = np.array(self.data[key], dtype=object)[idx_select].tolist()
            
            # Apply sampling if it's not training phase and sample_ratio < 1
			if phase != 'train' and sample_ratio < 1.0:
				self.sample_data(sample_ratio)
		
		def sample_data(self, ratio):
			num_samples = int(len(self) * ratio)
			indices = np.random.choice(len(self), num_samples, replace=False)
			for key in self.data:
				self.data[key] = [self.data[key][i] for i in indices]

		def _get_feed_dict(self, index):
			feed_dict = super()._get_feed_dict(index)
			pos = self.data['position'][index]  #Use index to get the position of the user item pair in the user's history sequence
			user_seq = self.corpus.user_his[feed_dict['user_id']][:pos] #  It is open and closed, so if the current id is a target id, it will not appear in the history sequence
			if self.model.history_max > 0:
				user_seq = user_seq[-self.model.history_max:]
			feed_dict['history_items'] = np.array([x[0] for x in user_seq])
			feed_dict['history_times'] = np.array([x[1] for x in user_seq])
			feed_dict['lengths'] = len(feed_dict['history_items'])
			return feed_dict

class CTRModel(GeneralModel):
	reader, runner = 'BaseReader', 'CTRRunner'

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--loss_n',type=str,default='BCE',
							help='Type of loss functions.')
		return GeneralModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.loss_n = args.loss_n
		if self.loss_n == 'BCE':
			self.loss_fn = nn.BCELoss()

	def loss(self, out_dict: dict) -> torch.Tensor:
		"""
		MSE/BCE loss for CTR model, out_dict should include 'label' and 'prediction' as keys
		"""
		if self.loss_n == 'BCE':
			loss = self.loss_fn(out_dict['prediction'],out_dict['label'].float())
		elif self.loss_n == 'MSE':
			predictions = out_dict['prediction']
			labels = out_dict['label']
			loss = ((predictions-labels)**2).mean()
		else:
			raise ValueError('Undefined loss function: {}'.format(self.loss_n))
		return loss

	class Dataset(BaseModel.Dataset):
		def _get_feed_dict(self, index):
			user_id, item_id = self.data['user_id'][index], self.data['item_id'][index]
			feed_dict = {
				'user_id': user_id,
				'item_id': [item_id],
				'label':[self.data['label'][index]]
			}
			return feed_dict

		# Without negative sampling
		def actions_before_epoch(self):
			pass