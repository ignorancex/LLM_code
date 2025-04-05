# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:29:46 2020

@author: Andrei
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
print_df = True

DEBUG = True # in-development flag

DATA_FILE = 'df_tran_proc_top_13.5k.csv'
DATA_SLICE_FILE = 'df_tran_proc_top_13.5k_slice.csv'
META_FILE = 'df_items_top_13.5k.csv'
META_INFO = 'obfuscated_keys.txt'

MCO_OUT_FILE = 'exp_mco.npz'
EMB_OUT_FILE = '_embeds.npy'
EMB128_5k_FN = 'exp_v3_i5k.npy'
EMB128_10k_FN = 'exp_v2_i10k.npy'
EMB128_250k_FN = 'exp_v1_embeds.npy' 
EMB128_ES33_FN = 'exp_v4es_033.npy'
CHUNK_SIZE = 100 * 1024 ** 2 # read100MB chunks
MAX_N_TOP_PRODUCTS = 13000  # top sold products
_MCO_FILE  = 'mco_top_13.5k.npz' # debug pre-prepared mco

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 500)
pd.set_option('display.width', 1000)
pd.set_option('precision', 4)  
pd.set_option('display.float_format', lambda x: '%.3f' % x)
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=500)

plt.style.use('ggplot')


from libraries_pub.logger import Logger

from prove.prove_utils import load_categs_from_json
from prove.prove_utils import generate_sparse_mco
from prove.prove_utils import show_neighbors
from prove.prove_utils import filter_categs
from prove.prove_utils import show_categs
  

log = Logger("PROVE", config_file='config.txt')


## Understanding the data

def show_product(IDE, df):
    rec = df[df.IDE==IDE].iloc[0]
    log.P("   Name:  {}".format(rec.ItemName))
    log.P("   Id:    {}".format(rec.ItemId))
    log.P("   Freq:  {}".format(rec.Freq))
    log.P("   Hrchy: {}/{}".format(rec.Ierarhie1, rec.Ierarhie2))
    return
    
log.P("Metatada information:")    
df_meta = log.load_dataframe(META_FILE)
df_meta = df_meta[df_meta.IDE < MAX_N_TOP_PRODUCTS]
log.P("  Total no of products: {}".format(df_meta.ItemId.unique().shape[0]))
log.P("  Total no of level 1 hierarchies: {}".format(df_meta.Ierarhie1.unique().shape[0]))
log.P("  Total no of level 2 hierarchies: {}".format(df_meta.Ierarhie2.unique().shape[0]))
log.P("  Most sold individual product over period:")
show_product(df_meta[df_meta.Freq == df_meta.Freq.max()].iloc[0]['IDE'], df_meta)
log.P("  Least sold individual product over period:")
show_product(df_meta[df_meta.Freq == df_meta.Freq.min()].iloc[0]['IDE'], df_meta)
df_meta[df_meta.ItemName.str.len()<30].iloc[:15,:-1]

df_meta, dct_categories = load_categs_from_json(df_meta, META_INFO, log)
df_meta[df_meta.ItemName.str.len()<30].iloc[:15,[3,7,8]]

chunk_reader = pd.read_csv(log.get_data_file(DATA_SLICE_FILE), iterator=True) 
chunk_reader.get_chunk(15).iloc[:,:-2]


#################
chunk_reader.close()


## NLU meets BPA

data_file = DATA_FILE if log.get_data_file(DATA_FILE) else DATA_SLICE_FILE

if not print_df or 'csr_mco' not in globals():
  csr_mco, tran_sizes = generate_sparse_mco(
      data_file, mco_out_file=MCO_OUT_FILE,
      return_counts=True, log=log,
      DEBUG=DEBUG, mco_file=_MCO_FILE,
      plot=not print_df)


### Meta-information & categories

dct_i2n = {k:v for k,v in zip(df_meta.IDE, df_meta.ItemName)}
    
# we slice and convert the sparse matrix
np_mco = csr_mco.toarray()[:MAX_N_TOP_PRODUCTS,:MAX_N_TOP_PRODUCTS]
df_meta = df_meta.iloc[:MAX_N_TOP_PRODUCTS]
# raw sanity-check
for i in range(np_mco.shape[0]):
    assert i in df_meta.IDE

show_categs(df_meta, dct_categories, k=5, log=log)


### Analogy with word-vectors



if 'wemb' not in globals():
    log.P("Loading GloVe word embeddings")
    data = log.load_np('glove_words_and_embeds_100d.npz')
    if data != None:
      np_words = data['arr_0']
      wemb = data['arr_1']
      
      def show_word(word):
          show_neighbors(
              idx=word, 
              embeds=wemb, 
              dct_i2n=np_words, 
              log=log,
              )    
      
      show_word('beatles')


## Self-supervised training and supervised testing

proposed_test_categs = {
  'Ierarhie1': {
      'COMICS&MANGA': 21, 'GASTRONOMIE': 20, 'HOME&DECO AA': 10, 'LIFESTYLE': 12, 'MULTIMEDIA': 8,
      'MUZICA': 2, 'NOVELTY': 23,'VINURI': 14,
  },
  'Ierarhie2': {
      'AUDIO & VIDEOBOOK': 259, 'Accesorii apple': 193, 'Activity': 294, 'Audiobook CD': 111,
      'BLU-RAY': 36, 'BLU-RAY 3D': 224, 'BOARD GAMES': 140, 'Backtoschool': 108, 'Bakery': 269,
      'Bath': 262, 'Blu-Ray Audio': 165, 'Blu-Ray Video': 65, 'Body Care': 330, 'CD Clasica/Opera': 83,
      'COMICS': 87, 'Comics': 120, 'DRAW MANGA/COMICS/VIDEOGAMES': 72, 'FILOSOFIE': 43, 
      'Film Blu-Ray': 136, 'Film DVD': 52, 'Film VHS': 343, 'GASTRO/BAKERY': 314, 'Giftware': 107,
      'Gourmet': 192, 'Jazz': 308, 'Kitchen': 118, 'LIFESTYLE': 283, 'MANGA': 63, 'ROBOTICA': 300, 
      'SelfHelp': 198, 'TRAVELING COLLECTION': 139, 'Travel': 243, 'Travel Mug': 144, 
      'Travelling collection': 75, 'UHD': 183, 'Vinyl': 10,
   }
}      
dct_cat = filter_categs(
    proposed_test_categs, df_meta, dct_categories,
    max_n_top_products=MAX_N_TOP_PRODUCTS)


show_categs(df_meta, dct_cat, k=1000, log=log)
df_tpr = df_meta[
      df_meta.Ierarhie1.isin(list(dct_cat['Ierarhie1'].values())) |
      df_meta.Ierarhie2.isin(list(dct_cat['Ierarhie2'].values()))
      ]


# Metrics and overall evaluation

def mrr_k(np_cands_batch, np_gold_batch, k=5):
  """
  will compute Mean Reciprocal Rank for a batch of predictions
  inputs:
    np_cands_batch: [batch, no_products] - the candidates for each experiment in batch
    np_gold_batch: [batch,] or [batch, 1] - the actual gold target product replacement
    k: int (default 5) the max number of candidates evaluated
  """
  assert len(np_cands_batch.shape) == 2
  np_gold_batch  = np_gold_batch.reshape(-1,1)
  np_cands_batch = np_cands_batch[:,:k]
  n_obs, n_cands = np_cands_batch.shape
  matches = np_cands_batch == np_gold_batch  
  _x = np.repeat(np.arange(1,n_cands+1).reshape(1,-1),n_obs, axis=0) 
  _y = np.ones((n_obs,n_cands)) * np.inf
  np_res = 1 / np.where(matches, _x, _y).min(axis=1)
  return np_res.mean()
  


## Hyperparameters & hand-picked validation data

EMBED_SIZE = 128
MAX_FREQ = 250
MAX_EPOCHS = 10000
RETROFIT_TOL = 1e-5
FULL_EDGES = True

test_items = [
    (12071, 11418, 9745), 
    (10088, 5369, 8527), 
    (9251, 4671, 2433), 
    (9845, 7837, 6440), 
    (8956,6020 ,2599), 
    (1150, 129, 1361),
]

org_items = [x[0] for x in test_items]
pos_items = [x[1] for x in test_items]
neg_items = [x[2] for x in test_items]

## ProVe 

GENERATE = False

if GENERATE:
  from prove.prove_model import ProVe
  prove_model = ProVe(
      default_embeds_file=EMB128_ES33_FN,
      name='exp_v5es',
      log=log,
      )
  
  if prove_model.embeds is None:
      prove_model.fit(
          mco=np_mco,
          embed_size=EMBED_SIZE, 
          max_cooccure=MAX_FREQ, 
          epochs=5000, #MAX_EPOCHS,
          interactive_session=True,
          validation_data=(org_items, pos_items)
      )
  embeds_name = prove_model.name
  embeds = prove_model.embeds
else:
  embeds_name = EMB128_ES33_FN #prove_model.name
  embeds = log.load_np(embeds_name, folder='models') # prove_model.embeds


from prove.prove_engine import EmbedsEngine
prod_eng = EmbedsEngine(
    np_embeddings=embeds,
    df_metadata=df_meta,
    name_field='ItemName',
    id_field='IDE',
    log=log,
    categ_fields=['Ierarhie1', 'Ierarhie2'],
    dct_categ_names=dct_categories,
    )

## Refining evaluation data using trained ProVe


exp_id, pos_id, neg_id = test_items[0]


dct_prod_info = prod_eng.get_item_info(exp_id, verbose=True, show_relations=True)




prod_eng.get_similar_items(exp_id, filtered=False, show=print_df, 
                           name='Non-filtered neighbors of {} using {} model'.format(
                               exp_id, embeds_name))    

FULL_RETROFIT = True


dct_run_settings =   {
          'method' : 'v4_th', 
          'batch_size'   : 64, 
          'lr'           : 0.002, 
          'dist'         :'l2', 
          'fixed_weights': 1,
          'epochs'       : 1,
          'skip_negative': False,
          },


d_sim = prod_eng.analize_item(
    exp_id, 
    positive_id=pos_id, 
    negative_id=neg_id, 
    embeds_name=embeds_name,
    verbose=True
    )

new_embeds = prod_eng.get_retrofitted_embeds(
        **dct_run_settings,
        )
    
n_dif = (np.abs(new_embeds - embeds).sum(axis=1) > 1e-3).sum()
log.P("Total {} embeddings modified".format(n_dif))
    
d_sim = prod_eng.analize_item(
        exp_id, 
        positive_id=pos_id, 
        negative_id=neg_id, 
        embeds=new_embeds,
        embeds_name=embeds_name+'_RETRO',
        verbose=True,
        )   

d = prod_eng.get_similar_items(
    exp_id, 
    embeds=new_embeds, 
    filtered=False, 
    show=print_df,
    name='Non-filtered neighbors of {} using RETROFITTED {} model'.format(
                                   exp_id, embeds_name))    

log.show_timers()