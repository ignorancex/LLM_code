from Benchmarking.Customized_DataLoader import GraphData_Loader as data_loader
from Benchmarking.manager import graph_manager
import json 
import torch
import numpy as np
import argparse
import os

# '/projects/ovcare/classification/Ali/Heram/Dataset/NSCLC_dataset/cross_validation_splits/data_folds.json'
parser = argparse.ArgumentParser(description='MIL package for training and inference of Histopathology instances')
parser.add_argument('--model_name', type=str, required=True, help='which model to use for MIL (DeepMIL, Clam_SB, Clam_MB, VarMIl, TransMIL)')
parser.add_argument("--hidden_layers", nargs='+', type=int,  default = [256, 128], 
        help="<<<< --hidden_layers [256 128] >>>> keep it consistent!")
parser.add_argument('--split_name', type=str, required=True, help='fold-i for i in range (1,k)')
parser.add_argument("--batch_size", nargs='+', type=int, required=True, default = 1, 
        help="<<<< --batch_size 16 >>>> ")
parser.add_argument("--lr", nargs='+', type=float, default = 0.001, 
        help="<<<< --lr 0.001 >>>> ")
parser.add_argument("--mags", nargs='+', type=str, default=['5x', '10x', '20x'],
        help="<<<< --mags 5x 10x 20x >>>> ")
parser.add_argument("--weight_decay", nargs='+', type=float, default = 0.0001, 
        help="<<<< --weight_decay 0.0001 >>>> ")
parser.add_argument("--seed", nargs='+', type=int,  default = 256, 
        help="<<<< --seed 256 >>>> keep it consistent!")
parser.add_argument("--feature_size", nargs='+', type=int, required = True, default = 512, 
        help="<<<< --feature_size 512 >>>> ")
parser.add_argument("--epochs", nargs='+', type=int, default = 50, 
        help="<<<< --epochs 50 >>>> ")
parser.add_argument("--classes", nargs='+', type=str, required = True, 
        help="<<<< --classes UCC:0 MicroP:1 >>>> ")
parser.add_argument('--path_to_folds', type=str, required=True, help='path to the data fold json file')
parser.add_argument('--path_to_save', type=str, required=True, help='path to the folder to save output .pt files')
parser.add_argument('--path_to_load', type=str, required=True, help='path to the folder to load output .pt files')
parser.add_argument('--spatial_gcn', type=str, help='in case of patchGCN and DGCN')
parser.add_argument('--conv_layer', type=str, default='gcn', help='in case of GRASP: choose from [gcn, gat, gcn2conv, sageconv, sgconv]')

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #
print(device)

seed = args.seed[0]
torch.manual_seed(seed)
model_name = args.model_name
hidden_layers = args.hidden_layers
split_name = args.split_name
batch_size = args.batch_size[0]
mags = args.mags
spatial_gcn = args.spatial_gcn
#print(spatial_gcn)
print(mags)
lr = args.lr[0]
weight_decay = args.weight_decay[0]
epochs = args.epochs[0]
in_size = args.feature_size[0]
label_dict = {}
print(args.classes)
for item in args.classes:
    label = item.split(':')
    label_dict[label[0]] = int(label[1])
#print(label_dict)
out_size = len(label_dict)
# path_to_differentiate data
path_to_folds_json = args.path_to_folds #'/projects/ovcare/classification/Ali/Bladder_project/Dataset/scripts/data_folds.json'
f = open(path_to_folds_json)
data_dict = json.load(f)
# path_to_save
path_to_save = args.path_to_save
if model_name in ['GRASP', 'ZoomMIL', 'GRASP_1']:
    from dgl.dataloading import GraphDataLoader
elif model_name in ['PatchGCN', 'DGCN', 'H2MIL', 'HiGT']:
    from torch_geometric.loader import DataLoader as GraphDataLoader


# saving structure: model_name + '_' + split_name + '_' + S + '_' + B + '_' + LR + '_' + WD + '.pt'
random_seeds = torch.randint(0, 10000, (10,)).numpy()
for seed in random_seeds:
    print('random seed:', seed, flush=True)
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    #structure = model_name + '_' + split_name + '_' + 'S' + str(seed) + '_' + 'B' + str(batch_size) + '_' + 'LR' + str(lr) + '_' + 'WD' + str(weight_decay) + '_' + 'E' + str(epochs) + '.pt'
    # data_train = data_loader(data_dict[split_name], label_dict, phase='train', transform=None, magnifications=mags, super_patch=False, pooling='mean', graph_type=model_name)
    # data_val = data_loader(data_dict[split_name], label_dict, phase='val', transform=None, magnifications=mags, super_patch=False, pooling='mean', graph_type=model_name)
    data_test = data_loader(data_dict[split_name], label_dict, phase='test', transform=None, magnifications=mags, super_patch=False, pooling='mean', graph_type=model_name)
    # train_val_loader = {'train': GraphDataLoader(data_train, batch_size=batch_size, shuffle=True),
    #                     'val': GraphDataLoader(data_val, batch_size=batch_size, shuffle=False)}
    test_loader = GraphDataLoader(data_test, batch_size=1, shuffle=False, num_workers=1)
    # weight = np.zeros((len(label_dict)))
    # for label in label_dict.keys():
    #     weight[label_dict[label]] = (np.array(data_dict[split_name]['label']['train'])==label).sum()
    # weights = np.array(weight, np.float32)**(-1)
    # weights = weights / np.sum(weights)
    # weights = torch.tensor(weights, device=device)
    # print(weights)
    # print('The weights is: ' , weights)
    print(f"Here we go BRO!  --- Test: {len(data_test)} slides", flush=True)

    if model_name == 'GRASP':
        from Benchmarking.models.GRASP import GRASP
        model = GRASP(in_dim=in_size, hidden_dim_1 = hidden_layers[0], hidden_dim_2 = hidden_layers[1], n_classes=out_size, conv_layer=args.conv_layer) # in_size is size of the input, out_size is numbers of classes
        model_details = model_name + '_' + str(hidden_layers[0]) + '_' + str(hidden_layers[1])
        structure = model_details + '_' + split_name + '_' + 'S' + str(seed) + '_' + 'B' + str(batch_size) + '_' + 'LR' + str(lr) + '_' + 'WD' + str(weight_decay) + '_' + 'E' + str(epochs) + '.pt'
    elif model_name == 'GRASP_1':
        from Benchmarking.models.GRASP_1 import GRASP
        model = GRASP(in_dim=in_size, hidden_dim_1 = hidden_layers[0], hidden_dim_2 = hidden_layers[1], n_classes=out_size) # in_size is size of the input, out_size is numbers of classes
        model_details = model_name + '_' + str(hidden_layers[0]) + '_' + str(hidden_layers[1])
        structure = model_details + '_' + split_name + '_' + 'S' + str(seed) + '_' + 'B' + str(batch_size) + '_' + 'LR' + str(lr) + '_' + 'WD' + str(weight_decay) + '_' + 'E' + str(epochs) + '.pt'    
    elif model_name == 'ZoomMIL':
        from Benchmarking.models.ZoomMIL import ZoomMIL
        model = ZoomMIL(in_feat_dim=in_size, hidden_feat_dim=256, out_feat_dim=512, dropout=None, k_sample=12, k_sigma=0.002, n_cls=out_size)
        model_details = model_name + '_' + str(256) + '_' + str(512)
        structure = model_details + '_' + split_name + '_' + 'S' + str(seed) + '_' + 'B' + str(batch_size) + '_' + 'LR' + str(lr) + '_' + 'WD' + str(weight_decay) + '_' + 'E' + str(epochs) + '.pt'
    elif model_name == 'H2MIL':
        from Benchmarking.models.H2MIL import GCN
        model = GCN(in_feats=in_size,n_hidden=256,out_classes=out_size, drop_out_ratio=0.0, pool1_ratio=0.1,pool2_ratio=4,pool3_ratio=4,mpool_method="global_mean_pool")
        model_details = model_name + '_' + str(hidden_layers[0]) + '_' + str(hidden_layers[1])
        structure = model_details + '_' + split_name + '_' + 'S' + str(seed) + '_' + 'B' + str(batch_size) + '_' + 'LR' + str(lr) + '_' + 'WD' + str(weight_decay) + '_' + 'E' + str(epochs) + '.pt' 
    elif model_name == 'HiGT':
        from Benchmarking.models.HiGT import HiGT
        model = HiGT(gcn_in_channels=in_size, out_classes=out_size, gcn_hid_channels = in_size, gcn_out_channels=in_size)
        model_details = model_name + '_' + str(hidden_layers[0]) + '_' + str(hidden_layers[1])
        structure = model_details + '_' + split_name + '_' + 'S' + str(seed) + '_' + 'B' + str(batch_size) + '_' + 'LR' + str(lr) + '_' + 'WD' + str(weight_decay) + '_' + 'E' + str(epochs) + '.pt' 
    elif model_name == 'PatchGCN':
        from Benchmarking.models.PatchGCN import PatchGCN
        if spatial_gcn=="True":
            print('spatial triggered!')
            model = PatchGCN( input_dim=in_size, num_layers=4, edge_agg='spatial', multires=False, resample=0,
            fusion=None, num_features=in_size, hidden_dim=128, linear_dim=64, use_edges=False, pool=False, dropout=0.25, n_classes=out_size)
            model_details = model_name + '_' + 'spatial' + '_' + str(hidden_layers[0]) + '_' + str(hidden_layers[1])
        else:
            print('latent triggered!')
            model = PatchGCN( input_dim=in_size, num_layers=4, edge_agg='latent', multires=False, resample=0,
            fusion=None, num_features=in_size, hidden_dim=128, linear_dim=64, use_edges=False, pool=False, dropout=0.25, n_classes=out_size)
            model_details = model_name + '_' + 'latent' + '_' + str(hidden_layers[0]) + '_' + str(hidden_layers[1])
        structure = model_details + '_' + split_name + '_'  + mags[0] + '_' + 'S' + str(seed) + '_' + 'B' + str(batch_size) + '_' + 'LR' + str(lr) + '_' + 'WD' + str(weight_decay) + '_' + 'E' + str(epochs) + '.pt' 
    elif model_name == 'DGCN':
        from Benchmarking.models.DGCN import DGCN
        if spatial_gcn=="True":
            print('spatial triggered!')
            model = DGCN( edge_agg='spatial', resample=0, num_features=in_size, hidden_dim=256, linear_dim=256, use_edges=False, dropout=0.25, n_classes=out_size)
            model_details = model_name + '_' + 'spatial' + '_' + str(hidden_layers[0]) + '_' + str(hidden_layers[1])
        else:
            print('latent triggered!')
            model = DGCN( edge_agg='latent', resample=0, num_features=in_size, hidden_dim=256, linear_dim=256, use_edges=False, dropout=0.25, n_classes=out_size)
            model_details = model_name + '_' + 'latent' + '_' + str(hidden_layers[0]) + '_' + str(hidden_layers[1])
        structure = model_details + '_' + split_name + '_'  + mags[0] + '_' + 'S' + str(seed) + '_' + 'B' + str(batch_size) + '_' + 'LR' + str(lr) + '_' + 'WD' + str(weight_decay) + '_' + 'E' + str(epochs) + '.pt' 

    else:
        raise NotImplementedError(f"it's beyond God's creation ability dear sister!")
    
    print(model_name, 'loaded successfully!', flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model_manager = graph_manager(model, device, model_name=model_name)
    print('I am learning: be polite and quiet!', flush=True)
    #best_model, best_optimizer, best_val_acc, best_epoch = model_manager.train(train_val_loader, optimizer, epochs=epochs, weights=weights)
    checkpoint = torch.load(args.path_to_load + 'checkpoint_' + structure, map_location = device)
    #checkpoint = {'model': best_model, 'optimizer': best_optimizer, 'val_acc': best_val_acc, 'epoch': best_epoch}
    print('Learning Finished: Wanna challenge me?', flush=True)
    output = model_manager.test(test_loader, checkpoint['model'].to(device)) # test
    try:
        os.makedirs(path_to_save, exist_ok=True)
    except OSError as e:
        print('dir error passed')
    torch.save(output, path_to_save + 'output_external_' + structure)