import torch
import numpy as np
import os
from sklearn.metrics import balanced_accuracy_score as bas
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import f1_score as f1
from sklearn.metrics import classification_report as cr
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
import argparse
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Reporting Cleaned Results for each model')
parser.add_argument("--models", nargs='+', type=str, required = True, help="<<<< --models GRASP ZoomMIL >>>> ")
parser.add_argument("--encoder", nargs='+', type=str, required = True, help="<<<< --encoder Phikon >>>> ")
parser.add_argument("--hidden_layers", nargs='+', type=int,  default = [256, 128], 
        help="<<<< --hidden_layers [256 128] >>>> keep it consistent!")
parser.add_argument("--mags", nargs='+', type=str, required = True, help="<<<< --models 5x 10x 20x >>>> ")
parser.add_argument("--batch_size", nargs='+', type=int, required=True, default = [1],  help="<<<< --batch_size 16 >>>> ")
parser.add_argument("--num_folds", nargs='+', type=int, required=True, default = [3],  help="<<<< --num_folds 3 >>>> ")
parser.add_argument("--lr", nargs='+', type=float, default = [0.001], help="<<<< --lr 0.001 >>>> ")
parser.add_argument("--weight_decay", nargs='+', type=float, default = [0.0001],  help="<<<< --weight_decay 0.0001 >>>> ")
parser.add_argument("--epochs", nargs='+', type=int, default = [50],  help="<<<< --epochs 50 >>>> ")
parser.add_argument('--path_to_outputs', type=str, required=True, help='path to the data fold json file')
parser.add_argument('--num_classes', nargs='+', type=int, default=[2], help='number of classes in the data')
#parser.add_argument('--path_to_save_fig', type=str, required=True, help='path to the data fold json file')
parser.add_argument('--is_external', type=bool, default=False, help='is your dataset an external evaluation')
args = parser.parse_args()

models = args.models
hidden_layers = args.hidden_layers
mags = args.mags
encoder = args.encoder[0]
batch_size = args.batch_size[0]
num_folds = args.num_folds[0]
lr = args.lr[0]
weight_decay = args.weight_decay[0]
epochs = args.epochs[0]
path_to_outputs = args.path_to_outputs + encoder +'/'
# path_to_save_fig = args.path_to_save_fig + encoder
# os.makedirs(path_to_save_fig, exist_ok=True)
torch.manual_seed(256)
random_seeds = torch.randint(0, 10000, (10,)).numpy()
multi_mag = ["GRASP", "ZoomMIL", "H2MIL", "HiGT", "GRASP_D", "GRASP_dropout", "GRASP_1"]
single_mag = ["DeepMIL", "VarMIL", "TransMIL", "Clam_SB", "Clam_MB", "PatchGCN_spatial", "PatchGCN_latent", "DGCN_spatial", "DGCN_latent"]
num_classes = args.num_classes[0]

def best_seed(result_dict, random_seeds, criterion = 'bacc'):
    x = result_dict[criterion]
    best_seed = np.argmax(np.mean(x, axis=1))
    the_best = {}
    for key in result_dict.keys():
        the_best[key] = result_dict[key][best_seed]
    #print(the_best)
    return random_seeds[best_seed], the_best

def avg_meter(data, num_folds):
    res_avg = {}
    res_std = {}
    for key in data.keys():
        res_avg[key] = np.mean(data[key])
        if len(data[key]) > num_folds:
            res_std[key] = np.std(np.mean(data[key], 0))
        elif len(data[key]) == num_folds:
            res_std[key] = np.std(data[key])
    
    return res_avg, res_std

def rank_founder(results, random_seeds, topk=3,criterion='bacc'):
    seed_ranking = []
    for seed, models_data in zip(results[list(results.keys())[0]].keys(), zip(*[models_data.values() for models_data in results.values()])):
        seed_score_per_model = {}
        for model, data in zip(results.keys(), models_data):
            avg_bacc = sum(data['bacc']) / len(data['bacc'])
            seed_score_per_model[model] = avg_bacc

        # Sort models based on 'bacc' values for the current seed
        sorted_models = sorted(seed_score_per_model.items(), key=lambda x: x[1], reverse=True)
        
        # Find the rank of each model and store the result
        seed_ranking.extend([(rank, model, seed) for rank, (model, _) in enumerate(sorted_models, start=1)])

    # Count how many times each model's seed appears in the top-3 ranking
    model_score = {}
    for rank, model, seed in seed_ranking:            
        if model not in model_score:
            model_score[model] = 0
        if rank <= topk:
            model_score[model] += 1

    # Print results
    print("Ranking of models based on the number of times their seed appears in the top-" + str(topk) + ":")
    for model, score in model_score.items():
        print(f"{model}: {score} times")
    # df = pd.DataFrame(list(model_score.items()), columns=['Model', 'Repetitions']).set_index('Model').sort_index()
    # ax = sns.barplot(x=df.index, y='Repetitions', data=df)
    # # Rotate x-axis labels by 45 degrees
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=-90, ha="right")
    # plt.title('The Frequency of model presence in top ' + str(topk) +  ' ranks')
    # # Format y-axis as integers
    # plt.tight_layout()
    # plt.savefig(path_to_save_fig + '/bar_plot_top'+ str(topk) + '.png')
    # plt.show()


results ={}
for model in models:
    results[model] = {}
    for seed in random_seeds:
        results[model][seed] = {'acc':[], 'bacc':[], 'auc':[], 'f1':[], 'time':[], 'recall':[], 'precision':[]}
        for fold in range(num_folds):
            split_name = 'fold-' + str(fold+1)
            if model in multi_mag:
                if model == 'ZoomMIL':
                    hidden_layers = [256, 512]
                    batch_size = 1
                elif model =='GRASP':
                    batch_size = batch_size
                model_details = model + '_' + str(hidden_layers[0]) + '_' + str(hidden_layers[1])
                if args.is_external:
                    structure = 'output_external_' + model_details + '_' + split_name + '_' + 'S' + str(seed) + '_' + 'B' + str(batch_size) + '_' + 'LR' + str(lr) + '_' + 'WD' + str(weight_decay) + '_' + 'E' + str(epochs) + '.pt'
                else:
                    structure = 'output_' + model_details + '_' + split_name + '_' + 'S' + str(seed) + '_' + 'B' + str(batch_size) + '_' + 'LR' + str(lr) + '_' + 'WD' + str(weight_decay) + '_' + 'E' + str(epochs) + '.pt'
            elif model in single_mag:
                if args.is_external:
                    structure = 'output_external_' + model + '_' + split_name + '_'  + mags[0] + '_' + 'S' + str(seed) + '_' + 'B' + str(batch_size) + '_' + 'LR' + str(lr) + '_' + 'WD' + str(weight_decay)+ '_' + 'E' + str(epochs) + '.pt'
                else:
                    if model in ["PatchGCN_spatial", "PatchGCN_latent", "DGCN_spatial", "DGCN_latent"]:
                        model_details = model + '_' + str(hidden_layers[0]) + '_' + str(hidden_layers[1])
                        structure = 'output_' + model_details + '_' + split_name + '_'  + mags[0] + '_' + 'S' + str(seed) + '_' + 'B' + str(batch_size) + '_' + 'LR' + str(lr) + '_' + 'WD' + str(weight_decay)+ '_' + 'E' + str(epochs) + '.pt'
                    else:
                        structure = 'output_' + model + '_' + split_name + '_'  + mags[0] + '_' + 'S' + str(seed) + '_' + 'B' + str(batch_size) + '_' + 'LR' + str(lr) + '_' + 'WD' + str(weight_decay)+ '_' + 'E' + str(epochs) + '.pt'
            data = torch.load(path_to_outputs + structure)
            accuracy = acc(data['labels'], data['preds'])
            balanced = bas(data['labels'], data['preds'])
            if num_classes == 2:
                avg_type = 'binary'
            else:
                avg_type = 'macro'
            f1_score = f1(data['labels'], data['preds'], average=avg_type)
            recall = recall_score(data['labels'], data['preds'], average=avg_type)
            #precision = precision_score(np.logical_not(data['labels']), np.logical_not(data['preds']), average='macro')
            precision = precision_score(data['labels'], data['preds'], average=avg_type)
            #kappa = cohen_kappa_score(data['labels'], data['preds'])
            #print(data['logits'][0].shape)
            #print(data['logits'])
            if len(data['logits'][-1].shape) == 1 and batch_size > 1: 
                data['logits'][-1] = data['logits'][-1].reshape(1,data['logits'][-1].shape[0])
            #print(data['logits'][-1].shape)#x = np.concatenate(data['logits'], axis=0)
            #print(data['logits'].reshape(-1,2))
            if model in multi_mag:
                preds = torch.nn.functional.softmax(torch.cat(data['logits'], dim=0).reshape(-1,num_classes), dim=1)
            elif model in single_mag:
                if model in [ "PatchGCN_spatial", "PatchGCN_latent", "DGCN_spatial", "DGCN_latent"]:
                    data['logits']=torch.cat(data['logits'], dim=0)
                #print(data['logits'])
                preds = torch.nn.functional.softmax(data['logits'].reshape(-1,num_classes), dim=1)
            preds = preds[:,1]#torch.max(preds, dim=1).values
            #logits = torch.cat(data['logits'], dim=0).reshape(-1,2)
            #logits = torch.max(logits, dim=1).values
            #print(preds)
            fpr, tpr, thresholds = roc_curve(data['labels'], preds, pos_label=1)
            #roc_auc_score
            #print(fpr, tpr, thresholds)
            area_uc = auc(fpr, tpr)
            #area_uc = roc_auc_score(data['labels'], preds, multi_class='ovr')
            #print(area_uc)
            
            #print(area_uc)
            results[model][seed]['acc'].append(accuracy)
            results[model][seed]['bacc'].append(balanced)
            results[model][seed]['f1'].append(f1_score)
            results[model][seed]['auc'].append(area_uc)
            results[model][seed]['recall'].append(recall)
            results[model][seed]['precision'].append(precision)
            results[model][seed]['time'].append(data['average_time']/batch_size)


for model in models:
    avg = {'acc':[], 'bacc':[], 'auc':[], 'f1':[], 'time':[], 'recall':[], 'precision':[]}
    stdv = {'acc':[], 'bacc':[], 'auc':[], 'f1':[], 'time':[], 'recall':[], 'precision':[]}
    print(model)
    #print(results[model])
    tmp = {'acc':[], 'bacc':[], 'auc':[], 'f1':[], 'time':[], 'recall':[], 'precision':[]}
    for seed in random_seeds:
        tmp['acc'].append(results[model][seed]['acc'])
        tmp['bacc'].append(results[model][seed]['bacc'])
        tmp['f1'].append(results[model][seed]['f1'])
        tmp['auc'].append(results[model][seed]['auc'])
        tmp['time'].append(results[model][seed]['time'])
        tmp['recall'].append(results[model][seed]['recall'])
        tmp['precision'].append(results[model][seed]['precision'])
    tmp['acc'] = np.array(tmp['acc'])
    tmp['bacc'] = np.array(tmp['bacc'])
    tmp['f1'] = np.array(tmp['f1'])
    tmp['auc'] = np.array(tmp['auc'])
    tmp['time'] = np.array(tmp['time'])
    tmp['recall'] = np.array(tmp['recall'])
    tmp['precision'] = np.array(tmp['precision'])
    best_seed_value, best_stat = best_seed(tmp, criterion='bacc', random_seeds=random_seeds)
    #print('best seed:', best_seed_value)
    avg, stdv = avg_meter(tmp, num_folds)
    avg_best, std_best = avg_meter(best_stat, num_folds)
    
    from prettytable import PrettyTable
    # Create PrettyTable for combined average and standard deviation
    combined_table = PrettyTable()

    # Add columns for Metric, Average Value, and Standard Deviation
    combined_table.field_names = ["Metric", "Average Value", "Standard Deviation", "Best Seed Avg", "Best Seed STD"]

    # Add rows for each metric
    for key in avg.keys():
        combined_table.add_row([key, "{:.4f}".format(avg[key]), "{:.4f}".format(stdv[key]), "{:.4f}".format(avg_best[key]), "{:.4f}".format(std_best[key])])

    # Print the table with metrics as separate columns
    print("Table of Results:")
    print(combined_table)
for topk in [1,3,5]:
    rank_founder(results, random_seeds, topk=topk)