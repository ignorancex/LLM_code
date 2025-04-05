import torch
import torch.nn as nn

from lib.word_vectors import obj_edge_vectors
from lib.concept_net_embeddings import load_concept_net
from einops import reduce


class MLP(nn.Module):
	def __init__(self, num_layers=2, obj_classes=37, action_class_num=157, semantic=False, concept_net=False):
		super(MLP, self).__init__()
		self.semantic = semantic

		if self.semantic:
			if not concept_net:
				self.input_size = 1936
			else:
				self.input_size = 2136
		else:
			self.input_size = 1536
		self.hidden_size = 2048
		self.obj_classes = obj_classes
		self.action_class_num = 157
		self.num_layers = num_layers

		self.union_func1 = nn.Conv2d(1024, 256, 1, 1)
		self.conv = nn.Sequential(
			nn.Conv2d(2, 256 //2, kernel_size=7, stride=2, padding=3, bias=True),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(256//2, momentum=0.01),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
			nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(256, momentum=0.01),
		)
		# fully connected layer
		self.subj_fc = nn.Linear(2048, 512)
		self.obj_fc = nn.Linear(2048, 512)
		self.vr_fc = nn.Linear(256*7*7, 512)
		
		if not concept_net:
			embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='data/', wv_dim=200)
			self.obj_embed = nn.Embedding(len(obj_classes), 200)
			self.obj_embed.weight.data = embed_vecs.clone()

			self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
			self.obj_embed2.weight.data = embed_vecs.clone()
		else:
			print("Loading concept net embeddings...")
			embed_vecs = load_concept_net(obj_classes, dim=300)

			self.obj_embed = nn.Embedding(len(obj_classes), 300)
			self.obj_embed.weight.data = embed_vecs.clone()

			self.obj_embed2 = nn.Embedding(len(obj_classes), 300)
			self.obj_embed2.weight.data = embed_vecs.clone()

			print("Loaded concept net embeddings...")

		if self.num_layers == 2:
			self.mlp = nn.Sequential(nn.Linear(self.input_size, self.hidden_size),
									 nn.ReLU(),
									 nn.Dropout(),
									 nn.Linear(self.hidden_size, self.action_class_num),
									 nn.Sigmoid())
		else:
			self.mlp = nn.Sequential(nn.Linear(self.input_size, self.hidden_size),
									 nn.ReLU(),
									 nn.Dropout(),
									 nn.Linear(self.hidden_size, self.hidden_size),
									 nn.ReLU(),
									 nn.Dropout(),
									 nn.Linear(self.hidden_size, self.action_class_num),
									 nn.Sigmoid())

	def forward(self, entry):
		entry['pred_labels'] = entry['labels']
		im_idx = entry['im_idx']
		b = int(im_idx[-1] + 1)
		l = torch.sum(im_idx == torch.mode(im_idx)[0])  # the highest box number in the single frame

		# get visual and semantic parts after using a fc-layer
		subj_rep = entry['features'][entry['pair_idx'][:, 0]]
		subj_rep = self.subj_fc(subj_rep)
		obj_rep = entry['features'][entry['pair_idx'][:, 1]]
		obj_rep = self.obj_fc(obj_rep)

		vr = self.union_func1(entry['union_feat'])+self.conv(entry['spatial_masks'])
		vr = self.vr_fc(vr.view(-1,256*7*7))

		# subject-object representations 
		x_visual = torch.cat((subj_rep, obj_rep, vr), 1)
		
		if not self.semantic:
			input = torch.zeros([b, x_visual.shape[1]]).to(x_visual.device)
			for i in range(b):
				if entry['pool_type'] == 'max':
					input[i, :] = reduce(x_visual[im_idx == i], 'n d -> d', 'max')
				elif entry['pool_type'] == 'avg':
					input[i, :] = reduce(x_visual[im_idx == i], 'n d -> d', 'mean')

		else:
			# add semantic part
			subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
			obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]
			subj_emb = self.obj_embed(subj_class) # 200 dim
			obj_emb = self.obj_embed2(obj_class) # 200 dim
			x_semantic = torch.cat((subj_emb, obj_emb), 1)

			rel_features = torch.cat((x_visual, x_semantic), dim=1)

			input = torch.zeros([b, rel_features.shape[1]]).to(rel_features.device)
			for i in range(b):
				if entry['pool_type'] == 'max':
					input[i, :] = reduce(rel_features[im_idx == i], 'n d -> d', 'max')
				elif entry['pool_type'] == 'avg':
					input[i, :] = reduce(rel_features[im_idx == i], 'n d -> d', 'mean')

			entry["unpooled_relational_feats"] = rel_features

		entry["action_class_distribution"] = self.mlp(input) # sigmoid already applied in nn.Sequential
		entry["relational_feats"] = input # max pooled features - hidden state for gru / query for transformer during sequence task

		return entry

class MLP_Veri(nn.Module):
	def __init__(self, num_layers=2, obj_classes=37, action_class_num=157, semantic=False):
		super(MLP_Veri, self).__init__()
		self.semantic = semantic
		if self.semantic:
			self.input_size = 1936
		else:
			self.input_size = 1536
		self.hidden_size = 2048
		self.obj_classes = obj_classes
		self.action_class_num = 157
		self.num_layers = num_layers

		self.union_func1 = nn.Conv2d(1024, 256, 1, 1)
		self.conv = nn.Sequential(
			nn.Conv2d(2, 256 //2, kernel_size=7, stride=2, padding=3, bias=True),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(256//2, momentum=0.01),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
			nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(256, momentum=0.01),
		)
		# fully connected layer
		self.subj_fc = nn.Linear(2048, 512)
		self.obj_fc = nn.Linear(2048, 512)
		self.vr_fc = nn.Linear(256*7*7, 512)

		embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='data/', wv_dim=200)
		self.obj_embed = nn.Embedding(len(obj_classes), 200)
		self.obj_embed.weight.data = embed_vecs.clone()

		self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
		self.obj_embed2.weight.data = embed_vecs.clone()


		if self.num_layers == 2:
			self.mlp = nn.Sequential(nn.Linear(self.input_size + 512, self.hidden_size),
									 nn.ReLU(),
									 nn.Dropout(),
									 nn.Linear(self.hidden_size, 1),
									 nn.Sigmoid())
		else:
			self.mlp = nn.Sequential(nn.Linear(self.input_size + 512, self.hidden_size),
									 nn.ReLU(),
									 nn.Dropout(),
									 nn.Linear(self.hidden_size, self.hidden_size),
									 nn.ReLU(),
									 nn.Dropout(),
									 nn.Linear(self.hidden_size, 1),
									 nn.Sigmoid())

	def forward(self, entry, class_enc = None):
		entry['pred_labels'] = entry['labels']
		im_idx = entry['im_idx']
		b = int(im_idx[-1] + 1)

		# get visual and semantic parts after using a fc-layer
		subj_rep = entry['features'][entry['pair_idx'][:, 0]]
		subj_rep = self.subj_fc(subj_rep)
		obj_rep = entry['features'][entry['pair_idx'][:, 1]]
		obj_rep = self.obj_fc(obj_rep)

		vr = self.union_func1(entry['union_feat'])+self.conv(entry['spatial_masks'])
		vr = self.vr_fc(vr.view(-1,256*7*7))

		# subject-object representations 
		x_visual = torch.cat((subj_rep, obj_rep, vr), 1)
		
		if not self.semantic:
			inputs = torch.zeros([b, x_visual.shape[1]]).to(x_visual.device)
			for i in range(b):
				if entry['pool_type'] == 'max':
					inputs[i, :] = reduce(x_visual[im_idx == i], 'n d -> d', 'max')
				elif entry['pool_type'] == 'avg':
					inputs[i, :] = reduce(x_visual[im_idx == i], 'n d -> d', 'mean')
				else:
					pass

		else:
			# add semantic part
			subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
			obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]
			subj_emb = self.obj_embed(subj_class) # 200 dim
			obj_emb = self.obj_embed2(obj_class) # 200 dim
			x_semantic = torch.cat((subj_emb, obj_emb), 1)

			rel_features = torch.cat((x_visual, x_semantic), dim=1)

			score = torch.zeros([b, 157]).to(rel_features.device)
			for i in range(b):
				if entry['pool_type'] == 'max':
					inputs = reduce(rel_features[im_idx == i], 'n d -> d', 'max')
					
				elif entry['pool_type'] == 'avg':
					inputs = reduce(rel_features[im_idx == i], 'n d -> d', 'mean')
				else:
					pass										
				score[i] = self.mlp(torch.cat([inputs.unsqueeze(0).repeat([157,1]),class_enc],dim=1)).squeeze()
				
		
		
		entry["action_class_distribution"] = score # sigmoid already applied in nn.Sequential

		return entry