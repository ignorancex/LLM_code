import json
import re
import numpy as np
import os
from model import *

device = 'cuda:0'
FILE = '../Data/electricity/e10kv2tl.json'
EFILE = '../Data/electricity/all_dict_correct_11_aoi.json'
BSFILE = '../Data/China_BS/bs_relation_combine.json'
ELE2BS_FILE = '../Data/China_BS/ele2bs_combined.json'
TFILE1 = '../Data/road/road_junc_map.json'
TFILE2 = '../Data/road/road_type_map.json'
TFILE3 = '../Data/road/tl_id_road2elec_map.json'
AOIFILE = '../Data/China_BS/aoi_combined.json'
ept = '../generate_failure_cascade/embedding/elec_feat.pt'
tpt = '../generate_failure_cascade/embedding/tra_feat.pt'
bspt = '../generate_failure_cascade/embedding/bs_feat.pt'
bpt = ('../generate_failure_cascade/embedding/bifeatures_aoi/bi_elec_feat.pt',
       '../generate_failure_cascade/embedding/bifeatures_aoi/bi_tra_feat.pt',
       '../generate_failure_cascade/embedding/bifeatures_aoi/bi_bs_feat.pt',
       '../generate_failure_cascade/embedding/bifeatures_aoi/bi_aoi_feat.pt')

EMBED_DIM = 64
HID_DIM = 128
FEAT_DIM = 64
KHOP = 5
BASE = 100000000

bsgraph = BSGraph(file=BSFILE,
                  embed_dim=EMBED_DIM,
                  hid_dim=HID_DIM,
                  feat_dim=FEAT_DIM,
                  khop=KHOP,
                  epochs=500,
                  pt_path=ept)  # 应该是bspt，为了先调通后面暂时改一下

egraph = ElecGraph(file=EFILE,
                   embed_dim=EMBED_DIM,
                   hid_dim=HID_DIM,
                   feat_dim=FEAT_DIM,
                   khop=KHOP,
                   epochs=500,
                   pt_path=ept)

tgraph = TraGraph(file1=TFILE1, file2=TFILE2, file3=TFILE3,
                  embed_dim=EMBED_DIM,
                  hid_dim=HID_DIM,
                  feat_dim=FEAT_DIM,
                  khop=KHOP,
                  epochs=300,
                  r_type='tertiary',
                  pt_path=tpt)

aoigraph = AOIGraph(file=AOIFILE,
                    embed_dim=EMBED_DIM,
                    hid_dim=HID_DIM,
                    feat_dim=FEAT_DIM,
                    khop=KHOP,
                    epochs=500,
                    pt_path=ept)

bigraph = Bigraph(efile=EFILE,
                  tfile1=TFILE1, tfile2=TFILE2, tfile3=TFILE3,
                  file=FILE,
                  bsfile=BSFILE,
                  ele2bsfile=ELE2BS_FILE,
                  aoifile=AOIFILE,
                  embed_dim=EMBED_DIM,
                  hid_dim=HID_DIM,
                  feat_dim=FEAT_DIM,
                  subgraph=(bsgraph, egraph, tgraph, aoigraph),
                  khop=KHOP,
                  epochs=1400,
                  r_type='tertiary',
                  pt_path=bpt)
print(1)
bi_nxgraph = bigraph.nxgraph
node_dict = {}
for k, v in bigraph.node_list.items():
    node_dict[v] = k
# with open('./GMNN_data/net.txt', 'a') as f:
#     for (u, v) in tqdm(bi_nxgraph.edges):
#         u = node_dict[u]
#         v = node_dict[v]
#         f.write('{0}\t{1}\t{2}\n'.format(u, v, 1))
#
#
# with open('./GMNN_data/feature.txt', 'a') as f:
#     index = 0
#     for node_type in ['power', 'junc', 'bs', 'aoi']:
#         features = bigraph.feat[node_type]
#         for node in trange(len(features)):
#             feat = features[node]
#             index_value_list = [(i, v) for i, v in enumerate(feat.view(-1))]
#             str_feat = ' '.join(['{}:{}'.format(i, v) for i, v in index_value_list])
#             f.write('{0}\t{1}\n'.format(index, str_feat))
#             index += 1


fpath = '../Data/ruin_cascades_for_different_type_and_size/500kv/cases'
for filename in os.listdir(fpath):
    with open('{0}/{1}'.format(fpath, filename), 'r') as f:
        data = json.load(f)
    source_nodes = data['source']
    source_nodes = [node_dict[v] for v in source_nodes]
    train_str = '\n'.join([str(x) for x in source_nodes])
    label_str = ''
    for index, node in enumerate(range(len(node_dict))):
        if not node in source_nodes:
            label_str = label_str + '{0}\t{1}\n'.format(index, -1)
        else:
            label_str = label_str + '{0}\t{1}\n'.format(index, 1)
    digits = re.findall('\d', filename)
    node_id = ''.join(digits)
    with open('./GMNN_data/train/train_{0}.txt'.format(node_id), 'w') as f:
        f.write(train_str)
    with open('./GMNN_data/label/label_{0}.txt'.format(node_id), 'w') as f:
        f.write(label_str)
