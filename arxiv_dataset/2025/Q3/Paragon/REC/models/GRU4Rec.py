import torch
import torch.nn as nn

from models.BaseModel import SequentialModel
from models.BaseImpressionModel import ImpressionSeqModel

class GRU4RecBase(object):

	def _base_init(self, args, corpus):
		self.emb_size = args.emb_size
		self.hidden_size = args.hidden_size
		self._base_define_params()
		self.apply(self.init_weights)
		self.use_adapter = False
	
	def _base_define_params(self):	
		self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
		self.rnn = nn.GRU(input_size=self.emb_size, hidden_size=self.hidden_size, batch_first=True)
		# self.pred_embeddings = nn.Embedding(self.item_num, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.emb_size)

		self.adapter = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size // 8),
            nn.ReLU(),
            nn.Linear(self.emb_size // 8, self.emb_size)
        )

	def forward(self, feed_dict):
		self.check_list = []
		i_ids = feed_dict['item_id']  # [batch_size, -1]
		history = feed_dict['history_items']  # [batch_size, history_max]
		lengths = feed_dict['lengths']  # [batch_size]

		his_vectors = self.i_embeddings(history)

		# Sort and Pack
		sort_his_lengths, sort_idx = torch.topk(lengths, k=len(lengths))
		sort_his_vectors = his_vectors.index_select(dim=0, index=sort_idx)
		history_packed = torch.nn.utils.rnn.pack_padded_sequence(
			sort_his_vectors, sort_his_lengths.cpu(), batch_first=True)

		# RNN
		output, hidden = self.rnn(history_packed, None)

		# Unsort
		unsort_idx = torch.topk(sort_idx, k=len(lengths), largest=False)[1]
		rnn_vector = hidden[-1].index_select(dim=0, index=unsort_idx)

		# Predicts
		# pred_vectors = self.pred_embeddings(i_ids)
		pred_vectors = self.i_embeddings(i_ids)

		rnn_vector = self.out(rnn_vector)
		# Then apply adapter
		if self.use_adapter:
			rnn_vector = rnn_vector + self.adapter(rnn_vector)  # Residual connection

		prediction = (rnn_vector[:, None, :] * pred_vectors).sum(-1)
		
		u_v = rnn_vector.repeat(1,i_ids.shape[1]).view(i_ids.shape[0],i_ids.shape[1],-1)
		i_v = pred_vectors 
		
		return {'prediction': prediction.view(feed_dict['batch_size'], -1),'item_id': i_ids,
				'u_v': u_v, 'i_v': i_v}

class GRU4Rec(SequentialModel, GRU4RecBase):
	reader = 'SeqReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'hidden_size']

	@staticmethod
	def parse_model_args(parser):
		parser = GRU4RecBase.parse_model_args(parser)
		return SequentialModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		SequentialModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		out_dict = GRU4RecBase.forward(self, feed_dict)
		return { 'prediction': out_dict['prediction'], 'item_id': out_dict['item_id'] }
	
class GRU4RecImpression(ImpressionSeqModel, GRU4RecBase):
	reader = 'ImpressionSeqReader'
	runner = 'ImpressionRunner'
	extra_log_args = ['emb_size', 'hidden_size']

	@staticmethod
	def parse_model_args(parser):
		parser = GRU4RecBase.parse_model_args(parser)
		return ImpressionSeqModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		ImpressionSeqModel.__init__(self, args, corpus)
		self._base_init(args, corpus)

	def forward(self, feed_dict):
		return GRU4RecBase.forward(self, feed_dict)