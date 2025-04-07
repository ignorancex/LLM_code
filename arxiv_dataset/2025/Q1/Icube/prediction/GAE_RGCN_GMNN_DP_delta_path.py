import dgl
import numpy as np

from model import *
from build_dataset import *

import torch.nn.functional as F
import re

device = 'cuda:1'
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
best_val = 0
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

dgl_egraph = dgl.from_networkx(egraph.nxgraph).to(device)
dgl_bsgraph = dgl.from_networkx(bsgraph.nxgraph).to(device)
dgl_tgraph = dgl.from_networkx(tgraph.nxgraph).to(device)

hetero_graph = build_hetero(nxgraph=bigraph.nxgraph,
                            embed_dim=EMBED_DIM,
                            hid_dim=HID_DIM,
                            feat_dim=FEAT_DIM,
                            subgraph=(bsgraph, egraph, tgraph, aoigraph),
                            pt_path=bpt).to(device)

model = GAE_GMNN_RGCN_DP_delta(EMBED_DIM + 1, 20, 2, hetero_graph.etypes).to(device)
fpath = '../Data/ruin_cascades_for_different_type_and_size/500kv/cases'
train_dataloader, test_dataloader = build_dataset(fpath)

opt = torch.optim.Adam(model.parameters(), lr=5e-4)

train_all_loss_list = []
train_auc_list = []
train_f1_list = []
train_pre_list = []
train_rec_list = []
train_rmse_list = []
test_all_loss_list = []
test_auc_list = []
test_f1_list = []
test_pre_list = []
test_rec_list = []
test_rmse_list = []

import os

spath = '../Data/plot_results_multinodes/GAE_RGCN_GMNN_DP_delta'
if not os.path.exists(spath):
    os.mkdir(spath)
best_checkpoint_path = "./{0}/best.pth".format(spath)
last_checkpoint_path = "./{0}/last.pth".format(spath)

if os.path.exists(last_checkpoint_path):
    checkpoint = torch.load(last_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    best_val = float(np.load('{0}/best_val.npy'.format(spath))[0])
    print('loaded checkpoint of epoch {0}'.format(checkpoint['epoch']))
else:
    print('No existing checkpoint found, start training from epoch 0')
    start_epoch = 0

for epoch in range(start_epoch + 1, 10000):
    model.train()
    train_all_loss = 0
    train_auc = []
    train_f1 = []
    train_pre = []
    train_rec = []
    train_rmse = []
    test_all_loss = 0
    test_auc = []
    test_f1 = []
    test_pre = []
    test_rec = []
    test_rmse = []
    for batch in train_dataloader:
        batch = random.sample(batch, len(batch))
        for batch_data in tqdm(batch, desc='training'):
            digits = re.findall('\d', batch_data)
            case_num = ''.join(digits)
            add_power, add_junc, add_bs, add_aoi, \
            end_power, end_junc, end_bs, end_aoi = get_dataset(
                batch_data,
                bigraph, egraph, tgraph, bsgraph, aoigraph,
                fpath)
            node_features = {
                'junc': torch.cat([hetero_graph.nodes['junc'].data['feature'],
                                   torch.tensor(add_junc).unsqueeze(1).to(device)], dim=1),
                'power': torch.cat([hetero_graph.nodes['power'].data['feature'],
                                    torch.tensor(add_power).unsqueeze(1).to(device)], dim=1),
                'bs': torch.cat([hetero_graph.nodes['bs'].data['feature'],
                                 torch.tensor(add_bs).unsqueeze(1).to(device)], dim=1),
                'aoi': torch.cat([hetero_graph.nodes['aoi'].data['feature'],
                                  torch.tensor(add_aoi).unsqueeze(1).to(device)], dim=1)
            }
            graph_list = [hetero_graph, dgl_egraph, dgl_tgraph, dgl_bsgraph]
            logits = model(graph_list, node_features, case_num)
            # 计算损失值
            save_output(end_power, end_junc, end_bs, end_aoi, logits, 'GAE_RGCN_GMNN_DP_delta', 'train')
            end_power = torch.tensor(end_power)
            end_junc = torch.tensor(end_junc)
            end_bs = torch.tensor(end_bs)
            end_aoi = torch.tensor(end_aoi)
            loss = nn.CrossEntropyLoss(weight=cal_weight(end_power).to(device))(logits['power'], end_power.to(device)) + \
                   nn.CrossEntropyLoss(weight=cal_weight(end_junc).to(device))(logits['junc'], end_junc.to(device)) + \
                   nn.CrossEntropyLoss(weight=cal_weight(end_bs).to(device))(logits['bs'], end_bs.to(device)) + \
                   nn.CrossEntropyLoss(weight=cal_weight(end_bs).to(device))(logits['aoi'], end_aoi.to(device))
            new_auc, new_f1, new_pre, new_rec, new_rmse = calculate_metrics(end_power, end_junc, end_bs, end_aoi,
                                                                            logits, epoch)
            train_auc.append(new_auc)
            train_f1.append(new_f1)
            train_pre.append(new_pre)
            train_rec.append(new_rec)
            train_rmse.append(new_rmse)
            train_all_loss += loss
    opt.zero_grad()
    train_all_loss = train_all_loss / len(batch)
    train_all_loss.backward()
    opt.step()
    print('epoch:', epoch, 'train:loss:', train_all_loss)
    print('auc:', np.mean(train_auc), 'f1:', np.mean(train_f1), 'precision:', np.mean(train_pre), 'recall:',
          np.mean(train_rec), 'rmse:', np.mean(train_rmse))
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': loss
    }, last_checkpoint_path)
    model.eval()
    for batch in test_dataloader:
        batch = random.sample(batch, len(batch))
        for batch_data in tqdm(batch, desc='testing'):
            digits = re.findall('\d', batch_data)
            case_num = ''.join(digits)
            add_power, add_junc, add_bs, add_aoi, \
            end_power, end_junc, end_bs, end_aoi = get_dataset(
                batch_data,
                bigraph, egraph, tgraph, bsgraph, aoigraph,
                fpath)
            node_features = {
                'junc': torch.cat([hetero_graph.nodes['junc'].data['feature'],
                                   torch.tensor(add_junc).unsqueeze(1).to(device)], dim=1),
                'power': torch.cat([hetero_graph.nodes['power'].data['feature'],
                                    torch.tensor(add_power).unsqueeze(1).to(device)], dim=1),
                'bs': torch.cat([hetero_graph.nodes['bs'].data['feature'],
                                 torch.tensor(add_bs).unsqueeze(1).to(device)], dim=1),
                'aoi': torch.cat([hetero_graph.nodes['aoi'].data['feature'],
                                  torch.tensor(add_aoi).unsqueeze(1).to(device)], dim=1)
            }
            graph_list = [hetero_graph, dgl_egraph, dgl_tgraph, dgl_bsgraph]
            logits = model(graph_list, node_features, case_num)
            save_output(end_power, end_junc, end_bs, end_aoi, logits, 'GAE_RGCN_GMNN_DP_delta', 'test')
            # 计算损失值
            end_power = torch.tensor(end_power)
            end_junc = torch.tensor(end_junc)
            end_bs = torch.tensor(end_bs)
            end_aoi = torch.tensor(end_aoi)
            loss = nn.CrossEntropyLoss(weight=cal_weight(end_power).to(device))(logits['power'], end_power.to(device)) + \
                   nn.CrossEntropyLoss(weight=cal_weight(end_junc).to(device))(logits['junc'], end_junc.to(device)) + \
                   nn.CrossEntropyLoss(weight=cal_weight(end_bs).to(device))(logits['bs'], end_bs.to(device)) + \
                   nn.CrossEntropyLoss(weight=cal_weight(end_bs).to(device))(logits['aoi'], end_aoi.to(device))
            new_auc, new_f1, new_pre, new_rec, new_rmse = calculate_metrics(end_power, end_junc, end_bs, end_aoi,
                                                                            logits, epoch)
            test_auc.append(new_auc)
            test_f1.append(new_f1)
            test_pre.append(new_pre)
            test_rec.append(new_rec)
            test_rmse.append(new_rmse)
            test_all_loss += loss
    test_all_loss = test_all_loss / len(batch)
    print('test:loss:', test_all_loss)
    print('auc:', np.mean(test_auc), 'f1:', np.mean(test_f1), 'precision:', np.mean(test_pre), 'recall:',
          np.mean(test_rec), 'rmse:', np.mean(test_rmse))
    print('--------------------------------------')
    if np.mean(test_f1) > best_val:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss
        }, best_checkpoint_path)
        best_val = np.mean(test_f1)
        np.save('{0}/best_val.npy'.format(spath), np.array([best_val]))
        print('*********************save for epoch {0}*********************'.format(epoch))
    train_all_loss_list, test_all_loss_list = plot_metrics(train_all_loss.item(),
                                                           test_all_loss.item(),
                                                           train_all_loss_list, test_all_loss_list, 'loss',
                                                           model='GAE_RGCN_GMNN_DP_delta')
    train_auc_list, test_auc_list = plot_metrics(np.mean(train_auc), np.mean(test_auc), train_auc_list, test_auc_list,
                                                 'auc', model='GAE_RGCN_GMNN_DP_delta')
    train_pre_list, test_pre_list = plot_metrics(np.mean(train_pre), np.mean(test_pre), train_pre_list, test_pre_list,
                                                 'precision', model='GAE_RGCN_GMNN_DP_delta')
    train_rec_list, test_rec_list = plot_metrics(np.mean(train_rec), np.mean(test_rec), train_rec_list, test_rec_list,
                                                 'recall', model='GAE_RGCN_GMNN_DP_delta')
    train_f1_list, test_f1_list = plot_metrics(np.mean(train_f1), np.mean(test_f1), train_f1_list, test_f1_list,
                                               'f1-score', model='GAE_RGCN_GMNN_DP_delta')
    train_rmse_list, test_rmse_list = plot_metrics(np.mean(train_rmse), np.mean(test_rmse), train_rmse_list,
                                                   test_rmse_list,
                                                   'rmse', model='GAE_RGCN_GMNN_DP_delta')
