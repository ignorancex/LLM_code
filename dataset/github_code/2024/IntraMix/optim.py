import torch
import copy
import argparse
import random
from tqdm import tqdm
import numpy as np
import optuna

from utils.conf import read_config
from utils.util import load_data,load_model,train_model_pyg_dynamic_data,generate_inclass_mixup,\
    generate_inclass_mixup_with_high_confidence_edge,generate_pseduo_label,generate_pseduo_label_same
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
parser=argparse.ArgumentParser(description="model training")
parser.add_argument('--dataset_name',type=str,default='Cora')
parser.add_argument('--model_name',type=str,default='gcn')
parser.add_argument('--cuda_device',type=int,default=0)
parser.add_argument('--n_trials',type=int,default=1000)
parser.add_argument('--step',type=int,default=1)
parser.add_argument('--epoch',type=int,default=400)
parser.add_argument('--lr',type=float,default=0.01)
parser.add_argument('--lr_mixup',type=float,default=0.02)
parser.add_argument('--hid_num',type=float,default=128)
parser.add_argument('--alpha',type=float,default=0.2)
parser.add_argument('--nb_heads',type=int,default=4)
parser.add_argument('--nb_heads2',type=int,default=1)
parser.add_argument('--dropout_rate1_before',type=float,default=0.9)
parser.add_argument('--dropout_rate2_before',type=float,default=0.005)
parser.add_argument('--dropout_rate1_high',type=float,default=0.5)
parser.add_argument('--dropout_rate1_mixup',type=float,default=0.9)
parser.add_argument('--dropout_rate2_mixup',type=float,default=0.005)
parser.add_argument('--weight_decay',type=float,default=0.0005)
parser.add_argument('--weight_decay_mixup',type=float,default=0.0001)
parser.add_argument('--with_test_data',type=str,default="yes")
parser.add_argument('--score_threshold_mixup',type=float,default=0.5)
parser.add_argument('--score_threshold_high',type=float,default=0.5)
parser.add_argument('--patient',type=int,default=40)
parser.add_argument('--dataset_root_path',type=str,default='./data')
parser.add_argument('--train_model_root_path',type=str,default='./store')
parser.add_argument('--config_root_path',type=str,default='./config')
args = parser.parse_args()
args=read_config(args)
device = torch.device('cuda:{}'.format(args.cuda_device) if torch.cuda.is_available() else 'cpu')
ori_x,ori_y,ori_edge_index,train_index,val_index,test_index,dataset=load_data(args.dataset_name,args.dataset_root_path,device)
ori_train_mask,ori_val_mask,ori_test_mask=torch.zeros(ori_x.shape[0], dtype=torch.bool).to(device),torch.zeros(ori_x.shape[0], dtype=torch.bool).to(device),torch.zeros(ori_x.shape[0], dtype=torch.bool).to(device)
ori_train_mask[train_index],ori_val_mask[val_index],ori_test_mask[test_index]=True,True,True
model=load_model(args,dataset.num_node_features,dataset.num_classes,args.hid_num,args.dropout_rate1_before,args.dropout_rate2_before).to(device)
loss = torch.nn.CrossEntropyLoss()
before_accs,vallina_mixup_accs,test_accs,test_accs_delta=[],[],[],[]
best_model_save_path="{}/{}_{}.pt".format(args.train_model_root_path,args.model_name,args.dataset_name)
if args.with_test_data=="no":
    unlabel_mask=~(ori_train_mask+ori_val_mask+ori_test_mask)
elif args.with_test_data=="yes":
    unlabel_mask=~(ori_train_mask+ori_val_mask)
unlabel_index=torch.nonzero(unlabel_mask==True,as_tuple=True)[0]
def objective(trial):
    x,y,train_y,train_mask,val_mask,test_mask,edge_index=copy.deepcopy(ori_x).to(device),copy.deepcopy(ori_y).to(device),\
    copy.deepcopy(ori_y).to(device),copy.deepcopy(ori_train_mask),copy.deepcopy(ori_val_mask),copy.deepcopy(ori_test_mask),copy.deepcopy(ori_edge_index)
    unlabel_index=torch.nonzero(unlabel_mask==True,as_tuple=True)[0]

    args.dropout_rate1_before=trial.suggest_float("dropout_rate1_before",0.0,0.99)
    args.dropout_rate2_before=trial.suggest_float("dropout_rate2_before",0.0,0.99)
    args.dropout_rate1_mixup=trial.suggest_float("dropout_rate1_mixup",0.0,0.99)
    args.dropout_rate2_mixup=trial.suggest_float("dropout_rate2_mixup",0.0,0.99)
    args.dropout_rate1_high=trial.suggest_float("dropout_rate1_high",0.0,0.99)
    args.lr=trial.suggest_float("lr",1e-4,1e-1)
    args.lr_mixup=trial.suggest_float("lr_mixup",1e-4,1e-1)
    args.weight_decay=trial.suggest_float("weight_decay",1e-5,5e-3)
    args.weight_decay_mixup=trial.suggest_float("weight_decay_mixup",1e-5,5e-3)
    args.score_threshold_mixup=trial.suggest_float("score_threshold_mixup",0.0,0.9)
    args.score_threshold_high=trial.suggest_float("score_threshold_high",0.0,0.9)
    args.alpha=trial.suggest_float("alpha",0.0,0.9)
    

    pretrain_model=load_model(args,dataset.num_node_features,dataset.num_classes,args.hid_num,args.dropout_rate1_before,args.dropout_rate2_before).to(device)
    model_intra_mixup=load_model(args,dataset.num_node_features,dataset.num_classes,args.hid_num,args.dropout_rate1_mixup,args.dropout_rate2_mixup).to(device)
    
    optimizer_pretrain=torch.optim.Adam(pretrain_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    optimizer_intra_mixup=torch.optim.Adam(model_intra_mixup.parameters(), lr=args.lr_mixup, weight_decay=args.weight_decay_mixup)

    train_model_pyg_dynamic_data(pretrain_model,optimizer_pretrain,x,edge_index,y,train_mask,val_mask,test_mask,args.epoch,best_model_save_path,patient=args.patient)
    pretrain_model.load_state_dict(torch.load(best_model_save_path))
    
    x,y,train_y,train_mask,val_mask,test_mask,edge_index=copy.deepcopy(ori_x).to(device),copy.deepcopy(ori_y).to(device),\
    copy.deepcopy(ori_y).to(device),copy.deepcopy(ori_train_mask),copy.deepcopy(ori_val_mask),copy.deepcopy(ori_test_mask),copy.deepcopy(ori_edge_index)
    unlabel_index=torch.nonzero(unlabel_mask==True,as_tuple=True)[0]
    train_mask,train_y,label_index_ori=generate_pseduo_label(pretrain_model,x,edge_index,train_mask,train_y,unlabel_index,score_threshold=args.score_threshold_mixup)
    train_mask_high,train_y_high,label_index_high=copy.deepcopy(ori_train_mask),copy.deepcopy(y),torch.nonzero(unlabel_mask==True,as_tuple=True)[0]
    train_mask_high,train_y_high,label_index_high=generate_pseduo_label_same(pretrain_model,x,edge_index,train_mask_high,train_y_high,label_index_high,model_dropout_rate=args.dropout_rate1_high,score_threshold=args.score_threshold_high)
    x,noise_y,clean_y,edge_index,train_mask,val_mask,test_mask=generate_inclass_mixup_with_high_confidence_edge(x,edge_index,train_y,train_y_high,y,label_index_ori,label_index_high,train_mask,val_mask,test_mask,dataset.num_classes,device)
    noise_y[label_index_high]=train_y_high[label_index_high]
    train_mask[label_index_high]=True
    best_test_acc=train_model_pyg_dynamic_data(model_intra_mixup,optimizer_intra_mixup,x,edge_index,noise_y,train_mask,val_mask,test_mask,args.epoch,best_model_save_path,test_y=clean_y)
    return best_test_acc

for i in tqdm(range(args.step)):
    seed=random.randint(0, 654321)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)
    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

