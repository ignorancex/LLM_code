"""
    @file: model.py
    @author: Aditya Desai
    @desc: federated learning for collaborative filtering

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from FakeRoast.FakeRoast import *
import pdb


class NCF(nn.Module):
  def __init__(self, user_num, item_num, factor_num, num_layers,
          dropout):
    super(NCF, self).__init__()
    """
    modified from yihong-chen/neural-collaborative-filtering

    user_num: number of users;
    item_num: number of items;
    factor_num: number of predictive factors;
    num_layers: the number of layers in MLP model;
    dropout: dropout rate between fully connected layers;
    model: 'MLP', 'GMF', 'NeuMF-end';
    """    
    self.dropout = dropout
    self.GMF_model = None
    self.MLP_model = None

    self.embed_user_GMF = nn.Embedding(user_num, factor_num)
    self.embed_item_GMF = nn.Embedding(item_num, factor_num)
    self.embed_user_MLP = nn.Embedding(
        user_num, factor_num * (2 ** (num_layers - 1)))
    self.embed_item_MLP = nn.Embedding(
        item_num, factor_num * (2 ** (num_layers - 1)))

    MLP_modules = []
    for i in range(num_layers):
      input_size = factor_num * (2 ** (num_layers - i))
      MLP_modules.append(nn.Dropout(p=self.dropout))
      MLP_modules.append(nn.Linear(input_size, input_size//2))
      MLP_modules.append(nn.ReLU())
    self.MLP_layers = nn.Sequential(*MLP_modules)

    predict_size = factor_num * 2
    self.predict_layer = nn.Linear(predict_size, 1)

    self._init_weight_()

  def _init_weight_(self):
    """ We leave the weights initialization here. """
    nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
    nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
    nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
    nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

    for m in self.MLP_layers:
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    nn.init.kaiming_uniform_(self.predict_layer.weight, 
                a=1, nonlinearity='sigmoid')

    for m in self.modules():
      if isinstance(m, nn.Linear) and m.bias is not None:
        m.bias.data.zero_()

  def forward(self, user, item):
    embed_user_GMF = self.embed_user_GMF(user)
    embed_item_GMF = self.embed_item_GMF(item)
    output_GMF = embed_user_GMF * embed_item_GMF

    embed_user_MLP = self.embed_user_MLP(user)
    embed_item_MLP = self.embed_item_MLP(item)
    interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
    output_MLP = self.MLP_layers(interaction)

    concat = torch.cat((output_GMF, output_MLP), -1)

    prediction = self.predict_layer(concat)
    return prediction.view(-1)


  def bpr_loss(self, users, pos, neg):
    ''' 
        users: user ids (batch,)
        pos: (batch,)
        neg: (batch, num_neg)
        
    '''
    positive_scores = self(users, pos)
    neg_shape = neg.shape
    negative_scores = torch.mean(self(users.reshape(-1,1).repeat(1,neg_shape[1]).reshape(-1), neg.reshape(-1)).reshape(*neg_shape), dim=1)


    user_m = self.embed_user_MLP(users)
    user_g = self.embed_user_GMF(users)

    pos_m = self.embed_item_MLP(pos)
    neg_m = self.embed_item_MLP(neg)
    pos_g = self.embed_item_GMF(pos)
    neg_g = self.embed_item_GMF(neg)

    embs = [user_m, user_g, pos_m, neg_m, pos_g, neg_g]
    reg_loss = 0
    for emb in embs:
          reg_loss += user_m.norm(2).pow(2)  / 2
    reg_loss = reg_loss / len(users)

    #loss = - torch.sum(torch.log(1e-3 + torch.nn.functional.sigmoid(positive_scores - negative_scores)))
    loss = - torch.mean(torch.nn.LogSigmoid()(positive_scores - negative_scores))
    if torch.isinf(loss):
        pdb.set_trace()
    return loss, reg_loss

class NCFUser(NCF):
  def __init__(self, user_num, item_num, factor_num, num_layers,
          dropout, compression, compression_seed):
    super(NCFUser, self).__init__(user_num, item_num, factor_num, num_layers, dropout)

    self.compression_seed = compression_seed
    self.compression = compression

    self.embed_item_MLP = FakeRoastEmbedding(
                 num_embeddings=self.embed_item_MLP.num_embeddings,
                 embedding_dim=self.embed_item_MLP.embedding_dim,
                 is_global=False,
                 weight=None,
                 init_scale=None,
                 compression=compression,
                 padding_idx=self.embed_item_MLP.padding_idx,
                 max_norm=self.embed_item_MLP.max_norm,
                 norm_type=self.embed_item_MLP.norm_type,
                 scale_grad_by_freq=self.embed_item_MLP.scale_grad_by_freq,
                 sparse=False,
                 seed = compression_seed,
                 test = False,
                 hashmode = "random")

    self.embed_item_GMF = FakeRoastEmbedding(
                 num_embeddings=self.embed_item_GMF.num_embeddings,
                 embedding_dim=self.embed_item_GMF.embedding_dim,
                 is_global=False,
                 weight=None,
                 init_scale=None,
                 compression=compression,
                 padding_idx=self.embed_item_GMF.padding_idx,
                 max_norm=self.embed_item_GMF.max_norm,
                 norm_type=self.embed_item_GMF.norm_type,
                 scale_grad_by_freq=self.embed_item_GMF.scale_grad_by_freq,
                 sparse=False,
                 seed = compression_seed*3+1,
                 test = False,
                 hashmode = "random")



class NCFUserFedHM(NCF): # low rank repreesentation of embeddings
  def __init__(self, user_num, item_num, factor_num, num_layers,
          dropout, compression, compression_seed):
    super(NCFUserFedHM, self).__init__(user_num, item_num, factor_num, num_layers, dropout)

    self.compression_seed = compression_seed
    self.compression = compression

    self.embed_item_MLP = LowRankEmbedding(
                 num_embeddings=self.embed_item_MLP.num_embeddings,
                 embedding_dim=self.embed_item_MLP.embedding_dim,
                 compression=compression,
                 padding_idx=self.embed_item_MLP.padding_idx,
                 max_norm=self.embed_item_MLP.max_norm,
                 norm_type=self.embed_item_MLP.norm_type,
                 scale_grad_by_freq=self.embed_item_MLP.scale_grad_by_freq,
                 sparse=False)

    self.embed_item_GMF = LowRankEmbedding(
                 num_embeddings=self.embed_item_GMF.num_embeddings,
                 embedding_dim=self.embed_item_GMF.embedding_dim,
                 compression=compression,
                 padding_idx=self.embed_item_GMF.padding_idx,
                 max_norm=self.embed_item_GMF.max_norm,
                 norm_type=self.embed_item_GMF.norm_type,
                 scale_grad_by_freq=self.embed_item_GMF.scale_grad_by_freq,
                 sparse=False)



class MF(nn.Module):
  def __init__(self, user_num, item_num, embedding_dim):
    super(MF, self).__init__()
    self.embed_user = nn.Embedding(user_num, embedding_dim)
    self.embed_item = nn.Embedding(item_num, embedding_dim)

    self._init_weight_()

  def _init_weight_(self):
    """ We leave the weights initialization here. """
    nn.init.normal_(self.embed_user.weight, std=0.01)
    nn.init.normal_(self.embed_user.weight, std=0.01)

  def forward(self, user, item):
    embed_user = self.embed_user(user)
    embed_item = self.embed_item(item)
    output = torch.sum(embed_user * embed_item, dim=1)
    return output.view(-1)


  def bpr_loss(self, users, pos, neg):
    ''' 
        users: user ids (batch,)
        pos: (batch,)
        neg: (batch, num_neg)
        
    '''
    positive_scores = self(users, pos)
    neg_shape = neg.shape
    negative_scores = torch.mean(self(users.reshape(-1,1).repeat(1,neg_shape[1]).reshape(-1), neg.reshape(-1)).reshape(*neg_shape), dim=1)


    user_m = self.embed_user(users)
    pos_m = self.embed_item(pos)
    neg_m = self.embed_item(neg)

    embs = [user_m, pos_m, neg_m]
    reg_loss = 0
    for emb in embs:
          reg_loss += user_m.norm(2).pow(2)
    reg_loss = reg_loss / len(users)

    #loss = - torch.sum(torch.log(1e-3 + torch.nn.functional.sigmoid(positive_scores - negative_scores)))
    loss = - torch.mean(torch.nn.LogSigmoid()(positive_scores - negative_scores))
    if torch.isinf(loss):
        pdb.set_trace()
    return loss, reg_loss



if __name__ == '__main__':
    user_num = 100
    item_num = 1000
    factor_num=16
    num_layers = 3
    dropout = 0.1
    m = NCF(user_num, item_num, factor_num, num_layers,
          dropout)

    user = torch.arange(10)
    pos_items = torch.randint(1000, (10,))
    neg_items = torch.randint(1000, (10,10))
    loss, reg_loss = m.bpr_loss(user, pos_items, neg_items)
    
