# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:51:07 2020

@author: Andrei
"""

import numpy as np
import pandas as pd
from time import time
import textwrap
import os
from collections import OrderedDict
import pickle

from libraries_pub.generic_obj import LummetryObject

from prove import prove_utils
          
__EMBENG_VER__ = '0.1.0.2'

###############################################################################
###############################################################################
# BEGIN EmbedsEngine
###############################################################################
###############################################################################


class EmbedsEngine(LummetryObject):
  """
  This is the main workhorse for the product replacement and new product creating 
  process.
  """
  def __init__(self,
             np_embeddings,
             df_metadata,
             name_field,
             id_field,
             categ_fields,
             dct_categ_names,
             strict_relations=True,
             name='emb_eng',
             **kwargs,
             ):
    """
    Inputs:
       `np_embeddings`    : the embeddings matrix `(N, D)`
       
       `df_metadata`      : the metadata dataframe with `N` products
       
       `name_field`       : field name where item names are stored in metadata 
       
       `id_field`         : field name for the item ids
       
       `categ_fields`     : list of category fields in order of hierarchy levels
       
       `log` : Log object
       
       `strict_relations` : (boolean, default True) flag that decides whether positive
                            relations are based on any category or only on category intersection
                            between products
             
       `dct_categ_names`  : dictionary that maps each each of `categ_fields` to ids and names
      
    """
    assert type(np_embeddings) == np.ndarray
    assert len(np_embeddings.shape) == 2
    assert type(df_metadata) == pd.DataFrame
    assert df_metadata.shape[0] > 1
    assert id_field in df_metadata.columns, "Field {} not in meta-data {}".format(
        id_field, list(df_metadata.columns))
    for fld in categ_fields:
      assert fld in df_metadata.columns, "Field {} not in meta-data {}".format(
        fld, list(df_metadata.columns))
      
    self.version = __EMBENG_VER__
    
    self.dct_pos_edges = None
    self.dct_neg_edges = None
    self.strict_relations = strict_relations
    
    self.embeds = np_embeddings    
    self.df_meta = df_metadata
    self.name_fld = name_field
    self.id_fld = id_field
    self.categ_fields = categ_fields
    self.name = name
    self.is_categ_field_str = True
    self.dct_categ_names = dct_categ_names
    self.dct_categ_i2n = {}
    for categ_field in self.categ_fields:
      if type(self.df_meta[categ_field].iloc[0]) != str:
        self.is_categ_field_str = False
      self.dct_categ_i2n[categ_field] = {v:k for k,v in self.dct_categ_names[categ_field].items()}
    super().__init__(**kwargs)
    return
    
  def startup(self):
    super().startup()

    self.P("Initializing Embeddings processing Engine v{}.".format(self.version))
    self._setup_meta_data()
    return
  
  def _maybe_load_graph(self):
    data = self.log.load_pickle_from_models(self.name + '.pkl')
    if data is not None:
      self.dct_pos_edges, self.dct_neg_edges, self.dct_categ_prods, self.dct_prods_categs = data
    return True if data is not None else False
      

  def _save_graph(self):
    fn = self.name + '.pkl'
    data = (
        self.dct_pos_edges, 
        self.dct_neg_edges, 
        self.dct_categ_prods, 
        self.dct_prods_categs,
        )
    self.log.save_pickle_to_models(data, fn)
    return False
        
  
  def _setup_meta_data(self):
    self.P("Preparing product dicts")
    self.dct_i2n = {idx:name for idx, name in zip(self.df_meta[self.id_fld], self.df_meta[self.name_fld])}
    self.dct_n2i = {v:k for k, v in self.dct_i2n.items()}
    self.df_meta[self.name_fld] = self.df_meta[self.name_fld].str[:30]
    self.categs_names = []
    for categ_fld in self.categ_fields:
      name_field = categ_fld + '_name'      
      dct_rev = {v:k for k, v in self.dct_categ_names[categ_fld].items()} if self.dct_categ_names else None
      if name_field not in self.df_meta.columns:
        if self.df_meta[categ_fld].dtype != object:
          if dct_rev is None:
            raise ValueError("Please provide `dct_categ_names` to translate from '{}' to actual names".format(
                categ_fld))
          hn = self.df_meta[categ_fld].apply(lambda x: dct_rev[x])
          self.df_meta[name_field] = hn
          self.categs_names.append(name_field)
          self.P("Created category string in '{}'".format(name_field))
        else:
          self.categs_names.append(categ_fld)
          self.P("Found category string in '{}'".format(categ_fld))
      else:
        self.categs_names.append(name_field)
        self.P("Found category string in '{}'".format(name_field))
        
    if not self._maybe_load_graph():
      self._construct_graph_from_meta()
      self._save_graph()
    return


  def _get_item_negatives(self, item_id, k=100, max_dist=0.70):
    self.log.start_timer('neg_get_neigh')
    idxs, dists = prove_utils.neighbors_by_idx(item_id, self.embeds, k=k)    
    self.log.end_timer('neg_get_neigh')
    filtered = []
    filtered = idxs[dists < max_dist]
    filtered1 = []
    filtered2 = []
    if len(filtered) > 0:
#      self.log.start_timer('neg_df1')
#      self.log.start_timer('neg_items_to_df')
#      df_orig = self._items_to_df(filtered)
#      self.log.end_timer('neg_items_to_df')
#      df1 = df_orig
#      self.log.start_timer('neg_filter_by_categ')
#      for categ_field in self.categ_fields:
#        df1 = df1[df1[categ_field] != self.dct_prods_categs[item_id][categ_field]]
#      self.log.end_timer('neg_filter_by_categ')
#      self.log.end_timer('neg_df1')
#      filtered1 = df1[self.id_fld].tolist()
      
      self.log.start_timer('neg_categ_filter')
      filtered1 = set(filtered)
      for categ_field in self.categ_fields:
        categ_id = self.dct_prods_categs[item_id][categ_field]
        categ_items = self.dct_categ_prods[categ_field][categ_id]
        filtered1 = filtered1 - set(categ_items)        
      self.log.end_timer('neg_categ_filter')
      
      self.log.start_timer('neg_item_name')
      item_name = self.dct_i2n[item_id]
      item_name = item_name.replace(')' ,' ').replace('(',' ')
      tokens = [x for x in item_name.split() if len(x) > 2]
      filtered2 = []
      for idx in filtered1:
        _name = self.dct_i2n[idx]
        ok = True
        for token in tokens:
          if _name.find(token) >= 0:
            ok = False
        if ok:
          filtered2.append(idx)
      self.log.end_timer('neg_item_name')
      
    res = filtered2
    return res
  
  def _meta_to_dicts(self):
    dct_categ_prods = {}
    dct_prods_categs = {x : {} for x in range(self.embeds.shape[0])}
    self.P("Constructing items knowledge graph with `strict_relations={}`".format(
        self.strict_relations))    
    for categ_field in self.categ_fields:
      self.P("  Retrieving products for category '{}'".format(categ_field))
      dct_categ_prods[categ_field] = {}
      for categ_id in self.df_meta[categ_field].unique():
        dct_categ_prods[categ_field][categ_id] = self.df_meta[
            self.df_meta[categ_field] == categ_id][self.id_fld].unique().tolist()
        for item_id in dct_categ_prods[categ_field][categ_id]:
          dct_prods_categs[item_id][categ_field] = categ_id
    self.dct_categ_prods = dct_categ_prods
    self.dct_prods_categs = dct_prods_categs
    return
    
  
  def _construct_graph_from_meta(self, MAX_PROD_NEIGH=3000):
    dct_pos_edges = {}    
    dct_neg_edges = {}
    
    self._meta_to_dicts()
    no_neigh_list = []
    for emb_idx in range(self.embeds.shape[0]):
      prod_neighbors = []
      for categ_field in self.dct_categ_prods:
        prod_categ = self.dct_prods_categs[emb_idx][categ_field]
        prods = self.dct_categ_prods[categ_field][prod_categ]
        if self.strict_relations:
          if len(prod_neighbors) > 0:
            prod_neighbors = set(prod_neighbors) & set(prods)
          else:
            prod_neighbors = prods
        else:
          prod_neighbors += prods
      if len(prod_neighbors) > MAX_PROD_NEIGH:
        raise ValueError('Looks like item {} has over {} products!'.format(
            emb_idx, MAX_PROD_NEIGH))                  
      dct_pos_edges[emb_idx] = list(prod_neighbors)
      no_neigh_list.append(len(dct_pos_edges[emb_idx]))
      if (emb_idx % 1000) == 0:
        self.log.Pr("  Creating positive relations graph {:.1f}%".format(
            (emb_idx + 1) / self.embeds.shape[0] * 100))    
    self.log.Pr("  Created positive relations graph.")
    self.P("  Items nr of neigbors distribution:")
    self.P(textwrap.indent(str(pd.Series(no_neigh_list).describe()), ' ' * 4))


    no_neigh_list = []
    self.log.Pr("  Creating negative relations graph...")
    for emb_idx in range(self.embeds.shape[0]):
      self.log.start_timer("neg_iter")
      self.log.start_timer("neg_get_negs")
      prod_neg_neigh = self._get_item_negatives(emb_idx)
      self.log.end_timer("neg_get_negs")
      dct_neg_edges[emb_idx] = prod_neg_neigh
      no_neigh_list.append(len(dct_neg_edges[emb_idx]))
      self.log.end_timer("neg_iter")
      if ((emb_idx + 1) % 100) == 0:
        self.log.Pr("  Creating negative relations graph {:.1f}%".format(
            (emb_idx + 1) / self.embeds.shape[0] * 100))    
      
    self.log.P("  Created negative relations graph.\t\t\t")    
    self.P("  Items nr of negative neigbors distribution:")
    self.P(textwrap.indent(str(pd.Series(no_neigh_list).describe()), ' ' * 4))
    

    self.dct_pos_edges = {k:list(v) for k,v in dct_pos_edges.items()}
    self.dct_neg_edges = {k:list(v) for k,v in dct_neg_edges.items()}
    return
  
  def _items_to_df(self, items, dists=None):
    if dists is not None:
      fields = [self.id_fld, self.name_fld, 'DIST'] + self.categs_names + self.categ_fields
      df = pd.DataFrame({self.id_fld: items, 'DIST':dists}, index=items)
    else:
      fields = [self.id_fld, self.name_fld] + self.categs_names + self.categ_fields
      df = pd.DataFrame({self.id_fld: items}, index=items)
   
    df_out = df.join(self.df_meta, on=self.id_fld, rsuffix='r')[fields]
    return df_out
        
      
  def get_item_info(self, item_id, verbose=False, show_relations=False):
    self.log.start_timer('get_item_info')
    predefined_names = ['ID','NAME', 'POS_EDGES', 'NEG_EDGES']
    dct_info = OrderedDict({})
    dct_info['ID'] = item_id
    dct_info['NAME'] = self.df_meta[self.df_meta[self.id_fld] == item_id][[self.name_fld]].iloc[0,0]
    for i, categ_field in enumerate(self.categ_fields):
      dct_info[categ_field] = self.dct_prods_categs[item_id][categ_field]
      if self.categs_names[i] != categ_field:
        dct_info[self.categs_names[i]] = self.dct_categ_i2n[categ_field][dct_info[categ_field]]
    dct_info['POS_EDGES'] = self.dct_pos_edges[item_id]
    dct_info['NEG_EDGES'] = self.dct_neg_edges[item_id]
    if verbose:
      self.P("")
      self.P("Product '{}' info:".format(dct_info['ID']))
      for k in dct_info:
        if k not in predefined_names:
          _s = "  {}: {}".format(k, dct_info[k])
          self.P(_s)
        elif k != 'ID':
          val = dct_info[k]
          if type(val) != list:
            self.P("  {}: {}".format(k, val))
          else:
            # assume products
            df = self._items_to_df(dct_info[k])
            self.P("  {} ({} items):\n{}".format(
                k, df.shape[0],
                textwrap.indent(str(df.head()),' ' * 4)))
    self.log.end_timer('get_item_info')            
    return dct_info
  
  def analize_item(self, 
                   item_id,
                   positive_id,
                   negative_id,
                   embeds=None,
                   show_df=False,
                   embeds_name=None,
                   verbose=False,
                   k=10,
                   ):
    self.log.start_timer('analize_item')

    self.log.start_timer('analize_item_info')
    d_i = self.get_item_info(item_id)
    self.log.stop_timer('analize_item_info')

    self.log.start_timer('analize_item_neibs')
    if embeds is None:
      embeds = self.embeds      
    idxs, dists = prove_utils.neighbors_by_idx(item_id, embeds, k=None)
    self.log.stop_timer('analize_item_neibs')

    self.log.start_timer('analize_item_dists')
    p_rank = np.where(idxs==positive_id)[0][0]
    n_rank = np.where(idxs==negative_id)[0][0]
    p_dist = dists[p_rank]
    n_dist = dists[n_rank]
    self.log.stop_timer('analize_item_dists')
    
    self.log.start_timer('analize_item_categ')
    all_good = True
    for categ_field in self.categ_fields:
      categ = self.dct_prods_categs[item_id][categ_field]
      neigh_categs = np.array([self.dct_prods_categs[x][categ_field] for x in idxs[:k]])
      if (neigh_categs != categ).sum() > 0:
        all_good = False
    self.log.stop_timer('analize_item_categ')

    self.log.start_timer('analize_item_df_similar')
    if show_df:
      df_f = self.get_similar_items(item_id, embeds, filtered=True)
      df_n = self.get_similar_items(item_id, embeds, filtered=False)
      self.P("  Non-filtered neighbors:")
      self.P(textwrap.indent(str(df_n), "    "))
      self.P("  Filtered neighbors:")
      self.P(textwrap.indent(str(df_f), "    "))
    self.log.stop_timer('analize_item_df_similar')

    self.log.start_timer('analize_item_print')      
    if verbose:
      self.P("Analysis of {}: '{}' {}".format(
          item_id, d_i['NAME'],
          "using model {}".format(embeds_name) if embeds_name else '')
          )
      self.P("  All top {} neighbors {}in same categories".format(
          k, 'NOT ' if not all_good else ''))
      self.P("  Rank/dist from positive {:<7} {:>5}/{:.3f}".format(
          str(positive_id)+':', 
          p_rank, p_dist))
      self.P("  Rank/dist from negative {:<7} {:>5}/{:.3f}".format(
          str(negative_id)+':', 
          n_rank, n_dist))
    self.log.stop_timer('analize_item_print')      

    self.log.stop_timer('analize_item')
    d_res = OrderedDict({
        'POS_D' : p_dist,
        'POS_R' : p_rank,
        'NEG_R' : n_rank,
        'NEG_D' : n_dist,
        'CATEG' : all_good,
        })
    return d_res
    

  def get_similar_items(self, 
                        item_id, 
                        embeds=None, 
                        filtered=False, 
                        k=10, 
                        show=False,
                        name=None
                        ):
    self.log.start_timer('get_similar_items')
    if embeds is None:
      embeds = self.embeds
    if filtered:
      _k = max(5000, k)
    else:
      _k = max(100, k)
      
    dct_info = self.get_item_info(item_id, verbose=False)
    c1 = dct_info[self.categs_names[0]]
    c2 = dct_info[self.categs_names[1]]
    
    df_res = prove_utils.show_neighbors(
        idx=item_id, 
        embeds=embeds, 
        log=self.log,
        k=_k, 
        df=self.df_meta, 
        id_field=self.id_fld, 
        name_field=self.name_fld,
        h1fld=self.categs_names[0], 
        h2fld=self.categs_names[1]
        )
    if filtered:
      df_res = df_res[
          (df_res['H1'] == c1) |
          (df_res['H2'] == c2)
          ]
    df = df_res.iloc[:k,:]
    if show:
      if name is not None:
        title = "  Table: {}".format(name)
      else:
        title = "  Top neighbors for product {}:\n{}".format(item_id)
      self.P("Similarity report:\n{}\n{}".format(
          title,
          textwrap.indent(str(df), "  ")
          )
      )
    else:
      if name is not None:
        self.log.Pmdc("Table: {}".format(name))
    self.log.end_timer('get_similar_items')
    return df
  
  
  #############################################################################
  #############################################################################
  #############################################################################
  
  
  def cold_start_item(self, name, dct_categs, need_items=None, similar_items=None):
    item_id = self.df_meta[self.id_fld].max() + 1
    record = {
        self.id_fld : item_id,
        self.name_fld :  name,
        }
    for categ_field in dct_categs:
      record[categ_field] = dct_categs[categ_field]
    self.df_meta.append(record)
    embed = self.cold_start_embed(
        dct_categs=dct_categs,
        need_items=need_items,
        similar_items=similar_items
        )
    self.embeds = np.concatenate((self.embeds, embed.reshape(1,-1)))
    self.get_similar_items(
        item_id=item_id, 
        filtered=False, 
        show=True,
        name='Non-filtered neighbors of new item {}'.format(
            item_id))
    return item_id
  
  
  def cold_start_embed(self, dct_categs, need_items=None, similar_items=None):
    """
    Steps:
      1. USER: categ1, categ2...
      2. SYS: embed
      3. 
      
      
      1. USER: categ1, categ2, interests categ
    """
    
    idxs_from_categ = []
    for categ_field in dct_categs:
      categ_id = dct_categs[categ_field]
      idxs_from_categ += self.dct_categ_prods[categ_field][categ_id]
    embed = self.embeds[idxs_from_categ].mean(axis=0)
    if need_items is not None:
      assert type(need_items) in [list, np.ndarray], "`need_items` must be either list or ndarray"
      
      need_embeds = self.embeds[need_items]
      embed = self._retrofit_vector_to_embeddigs(embed, need_embeds)
    # end need_items
    if similar_items is not None:
      assert type(similar_items) in [list, np.ndarray], "`similar_items` must be either list or ndarray"
      
      similar_embeds = self.embeds[similar_items]
      embed = self._retrofit_vector_to_embeddigs(embed, similar_embeds)
    # end similar_items  
    return embed
  
  def get_item_replacement(self, item_id, k=5, as_dataframe=False, verbose=True):    
    idxs, dists = prove_utils.neighbors_by_idx(
        idx=item_id, 
        embeds=self.embeds, 
        k=10)
    cands = idxs[1:k+1]
    cands_dists = dists[:1:k+1]
    all_ok = True    
    for cand in cands:
      for categ_type in self.categ_fields:
        if self.dct_prods_categs[item_id][categ_type] != self.dct_prods_categs[cand][categ_type]:
          all_ok = False
    if not all_ok:
      self.P("Top k={} items do NOT match same categories as {}. Performing retrofit...".format(k, item_id))
      new_embeds = self.default_retrofit(item_ids=item_id)
      idxs, dists = prove_utils.neighbors_by_idx(
          idx=item_id, 
          embeds=new_embeds, 
          k=10)
      cands = idxs[1:k+1]
      cands_dists = dists[:1:k+1]
    else:
      self.P("Top k={} items match same categories as {}".format(k, item_id))
    # end need retrofit
    if as_dataframe:
      res = self._items_to_df(cands, dists=cands_dists)
    else:
      res = cands
    if verbose:
      self.P("Item top_k={} replacement for item {} - '{}':\n{}".format(          
          k, item_id, self.dct_i2n[item_id],
          textwrap.indent(str(res), "    ")))
    return res
      
      
      
  


  #############################################################################
  #############################################################################
  #############################################################################

  
  def default_retrofit(self, item_ids=None):
    return self.get_retrofitted_embeds(
        item_ids=item_ids,
        method='v4_th',
        dist='l1',
        lr=0.001,
        epochs=20,
        skip_negative=False,
        batch_size=256,
        fixed_weights=1,
        verbose=False,
        )

  
  
  def get_retrofitted_embeds(self, 
                             item_ids=None, 
                             method='v1', 
                             dct_negative=None, 
                             skip_negative=False,
                             full_edges=True, 
                             **kwargs):
    self.P("")
    self.P("Performing retrofitting on {} embedding matrix...".format(self.embeds.shape))
    self.P("  Product(s):  {}".format(item_ids))
    self.P("  Full edges:  {}".format(full_edges))
    if dct_negative is not None:
      self.P("  Negatives:   {}".format(len(dct_negative)))
    _dct = self.dct_pos_edges
    _dct_neg = self.dct_neg_edges if not skip_negative else {}
    if item_ids is not None:
      _dct = {}
      _dct_neg = {}
      if type(item_ids) == int:
        item_ids = [item_ids]
      for item_id in item_ids:
        related_prods = self.dct_pos_edges[item_id]
        _dct[item_id] = related_prods
        if full_edges:
          for related_id in related_prods:
            if related_id not in _dct:
              _dct[related_id] = self.dct_pos_edges[related_id]
            elif item_id not in _dct[related_id]:
              _dct[related_id].append(item_id)
        if dct_negative is None and not skip_negative: 
          # unless we give specific negative dict or no negative
          negative_prods = self.dct_neg_edges[item_id]
          _dct_neg[item_id] = negative_prods
          if full_edges:
            for neg_id in negative_prods:
              if neg_id not in _dct_neg:
                _dct_neg[neg_id] = [item_id]
            
    if dct_negative is not None and not skip_negative:
      _dct_neg = dct_negative.copy()
      if full_edges:
        for neg_id in dct_negative:
          neg_neigh = _dct_neg[neg_id]
          for nn in neg_neigh:
            if nn not in _dct_neg:
              _dct_neg[nn] = [neg_id]
            elif neg_id not in _dct_neg[nn]:
              _dct_neg[nn].append(neg_id)
            
    method_name = '_retrofit_embeds_' + method    
    func = getattr(self, method_name)
    self.P("  Method:      {}".format(func.__name__))
    self.P("  Pos edges: {}".format(len(_dct)))
    self.P("  Neg edges: {}".format(len(_dct_neg)))
    self.P("Starting `{}()` retrofit function".format(func.__name__))
    t1 = time()
    embeds = func(dct_edges=_dct, dct_negative=_dct_neg, **kwargs)
    self._clear_th()
    t2 = time()
    self.P("Retrofit with '{}' took {:.1f}s\t\t\t\t\t\t\t".format(func.__name__, t2-t1))
    return embeds
  
  
  def _retrofit_embeds_v0(self, dct_edges, **kwargs):
    embeds = self._retrofit_faruqui_fast(
        np_X=self.embeds,
        dct_edges=dct_edges,
        )
    return embeds
    
  
  
  @staticmethod
  def _measure_changes(Y, Y_prev):
    """
    this helper function measures changes in the embedding matrix 
    between previously step of the retrofiting loop and the current one
    """
    return np.abs(np.mean(np.linalg.norm(
                              np.squeeze(Y_prev) - np.squeeze(Y),
                              ord=2)))



  def _retrofit_faruqui_fast(
      self, 
      np_X, 
      dct_edges, 
      max_iters=100, 
      tol=5e-3, 
      alpha=None, 
      beta=None):
    """
    Implements retrofitting method of Faruqui et al. https://arxiv.org/abs/1411.4166
    however using the implementation from Potts et at / Dingwell et al
    
    Inputs:
    ======
    np_X : np.ndarray
      This is the input embedding matrix
    
    dct_edges: dict
      This is the dict that maps a certain vector to all its relatives 
      
    max_iters: int (default=100)
    
    alpha, beta: callbacks that return floats as per paper alpha/beta

    tol : float (default=1e-2)
      If the average distance change between two rounds is at or
      below this value, we stop. Default to 10^-2 as suggested
      in the paper.
      
    
    Outputs: 
    ======
      np.ndarray: the retrofitted version of np_X
      
    Original code by Dingwell et all:
      ```
        for iteration in range(1, max_iters+1):
            t1 = time()
            for i in dct_edges:
              neighbors = dct_edges[i]
              n_neighbors = len(neighbors)
              if n_neighbors > 0:
                a = alpha(i)
                b = beta(i)
                retro = np.array([b * np_Y[j] for j in neighbors])
                retro = retro.sum(axis=0) + (a * np_X[i])
                norm = np.array([b for j in neighbors])
                norm = norm.sum(axis=0) + a
                np_Y[i] = retro / norm
      ```
      
    """

    if alpha is None:
      alpha = lambda x: 1.0
    if beta is None:
      beta = lambda x: 1.0 / len(dct_edges[x])

    np_Y = np_X.copy()
    np_Y_prev = np_Y.copy()
    self.P("  Stop tol:    {:.1e}".format(tol))
    self.P("  Training log:")   
    for iteration in range(1, max_iters+1):
      t1 = time()
      for i in dct_edges:
        neighbors = dct_edges[i]
        n_neighbors = len(neighbors)
        if n_neighbors > 0:
          a = alpha(i)
          b = beta(i)
          retro = b * np_Y[neighbors]
          retro = retro.sum(axis=0) + a * np_X[i] # - b_neg * neg_retro...
          norm = n_neighbors * b + a
          np_Y[i] = retro / norm
        if (i % 1000) == 0:
          self.log.Pr("    Iteration {:02d} - {:.1f}%".format(
              iteration, (i+1)/np_X.shape[0]*100))
        # end if
      # end for matrix rows
      changes = self._measure_changes(np_Y, np_Y_prev)
      t2 = time()
      if changes <= tol:
        self.P("    Retrofiting converged at iteration {} - change was {:.1e} ".format(
                iteration, changes))
        break
      else:
        np_Y_prev = np_Y.copy()
        self.P("    Iteration {:02d} - change was {:.1e}, iter time: {:.2f}s".format(
            iteration, changes, t2 - t1))
    # end for iterations
    return np_Y

  
  
  def _retrofit_vector_to_embeddigs(self, np_start_vect, np_similar_vectors):
    """
    this function will retrofit a single vector (can be even a random vector but 
    preferably a centroid from the original latent space) to a pre-prepared matrix 
    of similar embeddings using the basic Faruqui et al approach
    """
    self.P("Creating new item embedding starting from a {} vector and {} similar items".format(
        np_start_vect.shape, np_similar_vectors.shape))
    self.P("  Current distance: {:.2f}".format(
        self._measure_changes(np_start_vect, np_similar_vectors)))
    n_simils = np_similar_vectors.shape[0]
    np_full = np.concatenate((np_start_vect.reshape(-1,1),
                              np_similar_vectors))
    dct_edges = {0:[x for x in range(1, n_simils)]}
#    for x in range(1, n_simils):
#      dct_edges[x] = 0
    np_new_embeds = self._retrofit_faruqui(np_full, dct_edges)
    np_new_embed = np_new_embeds[0]
    self.P("  New distance: {:.2f}".format(
        self._measure_changes(np_new_embed, np_similar_vectors)))
    return np_new_embed
  

  def _prepare_retrofit_data(self, dct_positive, 
                             dct_negative=None, 
                             split=False, 
                             pad_id=-1,
                             fixed_weights=None):
    self.P("  Preparing retrofit data based on dict({})...".format(len(dct_positive)))
    if len(dct_positive) <= 1:
      raise ValueError("`dct_positive` must have more than 1 item")

    if split:
      # split in ids(N,) positives(N, var) negatives(N, var), pos_w(N), neg_w(N)
      dct_negative = {} if dct_negative is None else dct_negative
      start_ids = list(set(
              list(dct_positive.keys()) + list(dct_negative.keys())
              ))
      self.P("    Positive edges: {}".format(len(dct_positive)))
      self.P("    Negative edges: {}".format(len(dct_negative)))
      pos_lists = [dct_positive.get(x,[]) for x in start_ids]
      pos_lens = [len(x) for x in pos_lists]
      
      neg_lists = [dct_negative.get(x,[]) for x in start_ids]
      neg_lens = [len(x) for x in neg_lists]
      
      max_pos_len = max([1] + pos_lens)
      max_neg_len = max([1] + neg_lens)
      
      if fixed_weights is not None:
        pos_w = [fixed_weights for _ in pos_lists]
        neg_w = [fixed_weights for _ in neg_lists]
      else:
        pos_w = [1/len(x) if x!=[] else 0 for x in pos_lists]
        neg_w = [1/len(x) if x!=[] else 0 for x in neg_lists]   
      
      for i,seq in enumerate(pos_lists):
        nn = max_pos_len - len(seq)
        pos_lists[i] = seq + [pad_id] * nn
      for i,seq in enumerate(neg_lists):
        nn = max_neg_len - len(seq)
        neg_lists[i] = seq + [pad_id] * nn
      return start_ids, pos_lists, pos_w, neg_lists, neg_w    
    else:
      all_data = []  
      n_pos = len(dct_positive)
      view_step = n_pos / 100
      for i, idx in enumerate(dct_positive):
        neibs = dct_positive[idx]
        pre_weight = 1 / len(neibs)
        rel_weight = 1 / len(neibs)
        pairs = [[idx, x, pre_weight, rel_weight] for x in neibs]
        all_data += pairs
        if (i % view_step) == 0:
          self.log.Pr("    Preparing positive edges {:.1f}%".format((i+1)/n_pos * 100))
      self.P("    Prepared positive edges - total {:,} relations".format(len(all_data)))
      if dct_negative is not None:
        all_negative = []
        n_neg = len(dct_negative)
        view_step = n_neg / 100
        for i, idx in enumerate(dct_negative):
          neg_neibs = dct_negative[idx]
          pre_weight = 1 / len(neg_neibs) 
          neg_weight = -1 / len(neg_neibs)
          neg_pairs = [[idx, x, pre_weight, neg_weight] for x in neg_neibs]
          all_negative += neg_pairs
          if (i % view_step) == 0:
            self.log.Pr("    Preparing negative edges {:.1f}%".format((i+1)/n_neg * 100))
        all_data += all_negative
        self.P("    Prepared negative edges - total {:,} relations".format(len(all_negative)))
      np_all_data = np.array(all_data, dtype='float32')
      self.P("    Dataset: {}".format(np_all_data.shape))
      return np_all_data
      
  
  def _retrofit_embeds_v5_tf(self, 
                                    dct_edges, 
                                    dct_negative=None,
                                    eager=False, 
                                    use_fit=False,
                                    epochs=99, 
                                    batch_size=256,
                                    gpu_optim=True,
                                    lr=0.001,
                                    patience=2,
                                    tol=1e-3,
                                    dist='l1',
                                    fixed_weights=None,
                                    **kwargs):
    """
    this method implements a similar approach to Dingwell et al
    """
    import tensorflow as tf

    
    vocab_size = self.embeds.shape[0]
    embedding_dim = self.embeds.shape[1]
    
    pad_id = vocab_size

    data = self._prepare_retrofit_data(
        dct_positive=dct_edges,
        dct_negative=dct_negative,
        fixed_weights=fixed_weights,
        pad_id=pad_id,
        split=True,
        )
    
    
    if dct_negative is not None and len(dct_negative) > 0:
      negative_margin = 128 
      if dist == 'cos':
        negative_margin = 1
    else:
      negative_margin = 0
    
    self.P("  Preparing model...")
    
    np_embeds = np.concatenate([self.embeds, np.zeros((1,embedding_dim))])
    

    embeds_old = tf.keras.layers.Embedding(
        vocab_size + 1, embedding_dim, 
        embeddings_initializer=tf.keras.initializers.Constant(np_embeds),
        trainable=False,
        dtype=tf.float32,
        name='org_emb')
    embeds_new = tf.keras.layers.Embedding(
        vocab_size + 1, embedding_dim, 
        embeddings_initializer=tf.keras.initializers.Constant(np_embeds),
        trainable=True,
        dtype=tf.float32,
        name='new_emb')
    
    def cosine_distance(t1, t2):
      t1 = tf.nn.l2_normalize(t1, axis=-1)
      t2 = tf.nn.l2_normalize(t2, axis=-1)
      return 1 - tf.reduce_sum(t1 * t2, axis=-1)      
    
    def identity_loss(y_true, y_pred):
      return tf.math.maximum(0.0, tf.reduce_sum(y_pred))
    
    
    lyr_p_l2 = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(tf.pow(x[0]-x[1], 2), axis=-1), name='preserve_l2')
    lyr_p_l1 = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(tf.math.abs(x[0]-x[1]), axis=-1), name='preserve_l1')

    lyr_r_l2 = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(tf.pow(x[0]-x[1], 2), axis=-1), name='relate_l2')
    lyr_r_l1 = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(tf.math.abs(x[0]-x[1]), axis=-1), name='relate_l1')
    lyr_r_cos = tf.keras.layers.Lambda(lambda x: cosine_distance(x[0], x[1]), name='relate_cos')

    lyr_n_l2 = tf.keras.layers.Lambda(
        lambda x: tf.reduce_sum(tf.pow(x[0]-x[1], 2), axis=-1),
        name='negative_l2')
    lyr_n_l1 = tf.keras.layers.Lambda(
        lambda x: tf.reduce_sum(tf.math.abs(x[0]-x[1]), axis=-1), 
        name='negative_l1')
    lyr_n_cos = tf.keras.layers.Lambda(
        lambda x: cosine_distance(x[0], x[1]), 
        name='negative_cos')
        
        
    lyr_n_margin = tf.keras.layers.Lambda(
        lambda x: tf.maximum(negative_margin - x, 0),
        name='negative_margin')
    
    if dist == 'l1':
      lyr_p_dist = lyr_p_l1
      lyr_r_dist = lyr_r_l1
      lyr_n_dist = lyr_n_l1
    elif dist == 'l2':
      lyr_p_dist = lyr_p_l2
      lyr_r_dist = lyr_r_l2
      lyr_n_dist = lyr_n_l2
    elif dist == 'cos':
      lyr_p_dist = lyr_p_l2
      lyr_r_dist = lyr_r_cos
      lyr_n_dist = lyr_n_cos
    
    lyr_mask_r = tf.keras.layers.Lambda(lambda x: tf.cast(x != pad_id, dtype='float32'), name='relate_mask')
    lyr_mask_n = tf.keras.layers.Lambda(lambda x: tf.cast(x != pad_id, dtype='float32'), name='negative_mask')
    
    lyr_r_masking = tf.keras.layers.Multiply(name='relate_masking')
    lyr_n_masking = tf.keras.layers.Multiply(name='negative_masking')
    
    lyr_r_weighting = tf.keras.layers.Multiply(name='relation_weighting')
    lyr_n_weighting = tf.keras.layers.Multiply(name='negative_weighting')
    
    lyr_r_reduce = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True), name='relate_line_sum')
    lyr_n_reduce = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True), name='negative_line_sum')
    
    final_add = tf.keras.layers.Add(name='add_p_r_n')
    
    tf_src_id = tf.keras.layers.Input((1,), name='item_id')
    tf_rel_id = tf.keras.layers.Input((None,), name='related_ids')
    tf_rel_w = tf.keras.layers.Input((1,), name='related_weights', dtype='float32')
    tf_neg_id = tf.keras.layers.Input((None,), name='negative_ids')
    tf_neg_w = tf.keras.layers.Input((1,), name='negative_weights', dtype='float32')
    
    inputs = [tf_src_id, tf_rel_id, tf_rel_w, tf_neg_id, tf_neg_w]
        
    
    tf_src_emb = embeds_old(tf_src_id)
    tf_new_emb = embeds_new(tf_src_id)
    tf_rel_emb = embeds_new(tf_rel_id)
    tf_neg_emb = embeds_new(tf_neg_id)
    
    tf_p_dist = lyr_p_dist([tf_src_emb, tf_new_emb])
    tf_r_dist = lyr_r_dist([tf_new_emb, tf_rel_emb])
    tf_n_dist_pm = lyr_n_dist([tf_new_emb, tf_neg_emb])
    tf_n_dist = lyr_n_margin(tf_n_dist_pm)
    
    tf_r_mask = lyr_mask_r(tf_rel_id)
    tf_n_mask = lyr_mask_n(tf_neg_id)
    
    tf_r_masked = lyr_r_masking([tf_r_mask, tf_r_dist])
    tf_n_masked = lyr_n_masking([tf_n_mask, tf_n_dist])
    
    tf_r_weighted = lyr_r_weighting([tf_r_masked, tf_rel_w])
    tf_n_weighted = lyr_n_weighting([tf_n_masked, tf_neg_w])

    tf_r_batch_loss = lyr_r_reduce(tf_r_weighted)
    tf_n_batch_loss = lyr_n_reduce(tf_n_weighted)
    
    tf_retro_loss_batch = final_add([tf_p_dist, tf_r_batch_loss, tf_n_batch_loss])
    
    model = tf.keras.models.Model(inputs, tf_retro_loss_batch)
    opt = tf.keras.optimizers.SGD(lr=lr)
    model.compile(optimizer=opt, loss=identity_loss)  
    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(self.log.get_models_folder(),'rmodel_v5_{}_tf.png'.format(
            dist)),
        show_shapes=True,
        show_layer_names=True,
        expand_nested=True,
        )
    if eager:
      gpu_optim = False
    losses = []
    best_loss = np.inf
    fails = 0
    last_embeds = self.embeds
    best_embeds = None
    self.P("  Preparing dataset...")
    data_np = [np.array(x) for x in data]
    data_np_reshape = []
    for np_data in data_np:
      if np_data.dtype == np.float64:
        np_data = np_data.astype('float32')
      if len(np_data.shape) == 1:
        data_np_reshape.append(np_data.reshape(-1,1))
      else:
        data_np_reshape.append(np_data)
    tensors = tuple([tf.constant(x) for x in data_np_reshape])
    ds = tf.data.Dataset.from_tensor_slices(tensors)
    n_batches = data_np_reshape[0].shape[0] // batch_size + 1
    ds = ds.batch(batch_size)
    if gpu_optim:
      ds = ds.apply(tf.data.experimental.copy_to_device("/gpu:0"))
      ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    else:
      ds = ds.prefetch(1)
    self.P("  Training tf model ep={}, pos_batch={}, lr={:.1e}, tol={:.1e}, dist={}, fix_w={}, neg_m={}".format(
        epochs, tuple(tensors[1].shape), lr, tol, 
        dist, fixed_weights,
        negative_margin))
    t_start = time()
    if eager:
      for epoch in range(1, epochs+1):
        epoch_losses = []
        for i, batch in enumerate(ds):
          tf_i, tf_r, tf_rw, tf_n, tf_nw = batch
          with tf.GradientTape() as tape:
            
            tf_src_emb = embeds_old(tf_i)
            tf_new_emb = embeds_new(tf_i)
            tf_rel_emb = embeds_new(tf_r)
            tf_neg_emb = embeds_new(tf_n)
            
            tf_p_dist    = lyr_p_dist([tf_src_emb, tf_new_emb])
            tf_r_dist    = lyr_r_dist([tf_new_emb, tf_rel_emb])
            tf_n_dist_pm = lyr_n_dist([tf_new_emb, tf_neg_emb])
            tf_n_dist    = lyr_n_margin(tf_n_dist_pm)
            
            tf_r_mask = lyr_mask_r(tf_r)
            tf_n_mask = lyr_mask_n(tf_n)
            
            tf_r_masked = lyr_r_masking([tf_r_mask, tf_r_dist])
            tf_n_masked = lyr_n_masking([tf_n_mask, tf_n_dist])
            
            tf_r_weighted = lyr_r_weighting([tf_r_masked, tf_rw])
            tf_n_weighted = lyr_n_weighting([tf_n_masked, tf_nw])
        
            tf_r_batch_loss = lyr_r_reduce(tf_r_weighted)
            tf_n_batch_loss = lyr_n_reduce(tf_n_weighted)
            
            tf_retro_loss_batch = final_add([tf_p_dist, tf_r_batch_loss, tf_n_batch_loss])      
            
            tf_loss = identity_loss(None, tf_retro_loss_batch)
            
          epoch_losses.append(round(tf_loss.numpy(),2))
          grads = tape.gradient(tf_loss, model.trainable_weights)
#          test = _convert(grads[0])
          opt.apply_gradients(zip(grads, model.trainable_weights))
          self.log.Pr("    Epoch {:03d} - {:.1f}% - loss: {:.2f}".format(
              epoch, i / n_batches * 100, np.mean(epoch_losses)))
        # end batch
        epoch_loss = np.mean(epoch_losses)
        losses.append(epoch_loss)          
      # end epoch
    # end if eager
    else:     
      @tf.function
      def _train_on_batch(batch):
        with tf.GradientTape() as tape:
          tf_out = model(batch)
          tf_loss = identity_loss(None, tf_out)
        grads = tape.gradient(tf_loss, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))
        return tf_loss

      for epoch in range(1, epochs+1):
        epoch_losses = []
        t1 = time()
        for i, batch in enumerate(ds):
#          loss = model.train_on_batch(x=batch, y=batch[0])
          loss = _train_on_batch(batch)
          epoch_losses.append(round(loss.numpy(),2))
          self.log.Pr("    Epoch {:02d} - {:.1f}% - loss: {:.2f}".format(
              epoch, i / n_batches * 100, np.mean(epoch_losses)))
        t2 = time()
        epoch_loss = np.mean(epoch_losses)
        losses.append(epoch_loss)
        new_embeds = embeds_new.get_weights()[0][:-1]
        if epoch_loss < best_loss:
          best_embeds = new_embeds
          best_loss = epoch_loss
          fails = 0
        else:
          fails += 1
        diff = self._measure_changes(last_embeds, new_embeds)
        self.P("    Epoch {:02d}/{} - loss: {:.2f}, change:{:.3f}, time: {:.1f}s, fails: {}".format(
            epoch, epochs, epoch_loss, diff, t2 - t1,  fails))
        if fails >= patience or diff <= tol:
          break
        last_embeds = new_embeds
        
    train_time = time() - t_start
    if epoch < epochs:
      self.P("  Training stopped at epoch {}/{} - {:.1f}s\t\t\t\t\t\t\t".format(
          epoch, epochs, train_time))
    else:
      self.P("  Training done in {} epochs - {:.1f}s\t\t\t\t\t\t\t".format(
          epoch, train_time))
    
    return best_embeds
  
    
    
  
  
  def _retrofit_embeds_v4_th(self, 
                                    dct_edges, 
                                    dct_negative=None,
                                    eager=False, 
                                    use_fit=False,
                                    epochs=99, 
                                    batch_size=256,
                                    gpu_optim=True,
                                    lr=0.001,
                                    tol=1e-3,
                                    patience=2,
                                    DEBUG=False,
                                    dist='l2',
                                    fixed_weights=None,  
                                    verbose=True,
                                    **kwargs):
    """
    this method implements a similar approach to Dingwell et al
    """
    import torch as th

    vocab_size = self.embeds.shape[0]
    embedding_dim = self.embeds.shape[1]

    pad_id = vocab_size
    
    data = self._prepare_retrofit_data(
        dct_positive=dct_edges,
        dct_negative=dct_negative,
        split=True,
        pad_id=pad_id,
        fixed_weights=fixed_weights,
        )
        
    self.P("  Preparing torch model...")
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    if th.cuda.is_available():
      th.cuda.empty_cache()
    
    tensors = [th.tensor(x, requires_grad=False, device=device) for x in data]

    th_embeds = th.tensor(self.embeds, dtype=th.float32, requires_grad=False, device=device)
    th_embeds_pad = th.cat((th_embeds, th.zeros((1,embedding_dim), device=device)))
    
    
    ds = th.utils.data.TensorDataset(*tensors)
    dl = th.utils.data.DataLoader(
        dataset=ds,
        batch_size=batch_size, 
        shuffle=not DEBUG,
        )
    
    emb_new = th.nn.Embedding(
        vocab_size + 1, 
        embedding_dim, 
        padding_idx=vocab_size).to(device)
    
    emb_new.weight.data.copy_(th_embeds_pad)
    opt = th.optim.SGD(params=emb_new.parameters(), lr=lr) 
    n_batches = len(data[0]) // batch_size + 1
    losses = []
    best_loss = np.inf
    fails = 0
    last_embeds = self.embeds

    if dct_negative is not None and len(dct_negative) > 0:
      negative_margin = 128 
      if dist == 'cos':
        negative_margin = 1
    else:
      negative_margin = 0
    self.P("  Training th ep={}, pos_batch={}, lr={:.1e}, tol={:.1e}, dist={}, fix_w={}, neg_m={}".format(
        epochs, (batch_size, tensors[1].shape[1]), lr, tol, 
        dist, fixed_weights,
        negative_margin))
    t_start = time()
    for epoch in range(1, epochs + 1):
      epoch_losses = []
      p_losses = []
      r_losses = []
      n_losses = []
      t1 = time()
      for i, batch in enumerate(dl):
        th_ids, th_pos, th_pos_w, th_neg, th_neg_w = batch
        th_pos_w = th_pos_w.unsqueeze(-1)
        th_neg_w = th_neg_w.unsqueeze(-1)
        
        th_org_embs_raw = th_embeds[th_ids].unsqueeze(1)
        th_new_embs_raw = emb_new(th_ids).unsqueeze(1)
        th_pos_embs_raw = emb_new(th_pos)
        th_neg_embs_raw = emb_new(th_neg)
        
        th_org_embs = th_org_embs_raw
        th_new_embs = th_new_embs_raw
        th_pos_embs = th_pos_embs_raw
        th_neg_embs = th_neg_embs_raw        

        if dist == 'l2':
          th_preserve_loss = (th_org_embs - th_new_embs).pow(2).sum(-1)
        elif dist == 'l1':
          th_preserve_loss = (th_org_embs - th_new_embs).abs().sum(-1)
        elif dist == 'cos':
          ### maybe huber?
          th_preserve_loss = (th_org_embs - th_new_embs).pow(2).sum(-1)                    
        
        if dist == 'l2':
          th_relate_nm = (th_pos_embs - th_new_embs).pow(2).sum(-1)
        elif dist == 'l1':
          th_relate_nm = (th_pos_embs - th_new_embs).abs().sum(-1)
        elif dist == 'cos':
          th_relate_nm = 1 - th.nn.functional.cosine_similarity(
              th_new_embs, th_pos_embs, dim=-1) 
          
        th_relate_mask = (th_pos != pad_id).float()
        th_relate_masked = th_relate_nm * th_relate_mask        
        th_relate_weighted = th_relate_masked * th_pos_w
        th_relate_loss = th_relate_weighted.sum(-1, keepdims=True)
        
        if dist == 'l2':
          th_neg_nm = (th_new_embs - th_neg_embs).pow(2).sum(-1)
        elif dist == 'l1':
          th_neg_nm = (th_new_embs - th_neg_embs).abs().sum(-1)  
        elif dist == 'cos':
          th_neg_nm = 1 - th.nn.functional.cosine_similarity(
              th_new_embs, th_neg_embs, dim=-1)
          negative_margin = 1
                
        th_neg_nm_d = th.clamp(negative_margin - th_neg_nm, min=0)
        th_neg_mask = (th_neg != pad_id).float()
        th_neg_masked = th_neg_mask * th_neg_nm_d        
        th_neg_weighted = th_neg_masked * th_neg_w
        th_neg_loss = th_neg_weighted.sum(-1, keepdims=True)
        
#        th_temp = th_preserve_loss + th_relate_loss +  th_neg_loss
        
        th_p_loss = th_preserve_loss.sum()
        th_r_loss = th_relate_loss.sum()
        th_n_loss = th_neg_loss.sum()
        
        th_loss = th_p_loss + th_r_loss + th_n_loss
        
        opt.zero_grad()
        th_loss.backward()
        opt.step()
        epoch_losses.append(round(th_loss.detach().cpu().item(),2))
        epoch_loss = np.mean(epoch_losses)
        if verbose:
          p_losses.append(round(th_p_loss.detach().cpu().item(),2))
          r_losses.append(round(th_r_loss.detach().cpu().item(),2))
          n_losses.append(round(th_n_loss.detach().cpu().item(),2))
          p_loss = np.mean(p_losses)
          r_loss = np.mean(r_losses)
          n_loss = np.mean(n_losses)
          self.log.Pr("    Epoch {:02d} - {:.1f}% - loss: {:.2f} (P/R/N: {:.2f}/{:.2f}/{:.2f})".format(
              epoch, i / n_batches * 100, 
              epoch_loss,
              p_loss, r_loss, n_loss))
      # end batch
      t2 = time()
      losses.append(epoch_loss)
      new_embeds = emb_new.weight.data.detach().cpu().numpy()[:-1]         
      if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_embeds = new_embeds
        fails = 0
      else:
        fails += 1
      diff = self._measure_changes(last_embeds, new_embeds)
      if verbose:
        self.P("    Epoch {:02d}/{} - loss: {:.2f} (P/R/N: {:.2f}/{:.2f}/{:.2f}), change:{:.3f}, time: {:.1f}s, fails: {}".format(
            epoch, epochs, 
            epoch_loss, p_loss, r_loss, n_loss,
            diff, t2 - t1, fails))
      else:
        print(".", end='', flush=True)
      if fails >= patience or diff <= tol:
        break
      last_embeds = new_embeds
    # end epoch
    train_time = time() - t_start
    if epoch < epochs:
      self.P("  Training stopped at epoch {}/{} - {:.1f}s\t\t\t\t\t\t\t".format(
          epoch, epochs, train_time))
    else:
      self.P("  Training done in {} epochs - {:.1f}s\t\t\t\t\t\t\t".format(
          epoch, train_time))
    return best_embeds
  
  def _clear_th(self):
    try:
      import torch as th
      if th.cuda.is_available():
        th.cuda.empty_cache()    
    except:
      pass
    return
        
    
  
  

###############################################################################
# END EmbedsEngine
###############################################################################

