import numpy as np
from utils import label2name
import scipy.ndimage as ndi
import sys

name2label = {v:i for i,v in enumerate(label2name)}


input_dir = f'../../models/{sys.argv[1]}/test/'


with open(input_dir + 'fnames.txt') as f: fnames = f.read().split('\n')


outputs = np.load(input_dir + 'outputs.npy')

names = [f.split('/')[-4] for f in fnames]
labels = np.array([name2label[n] if n in name2label else -1 for n in names])


def get_acc(outputs,names,labels):
    mask = labels >= 0
    outputs_filt = outputs[mask]
    labels_filt = labels[mask]
    y_est = np.argmax(outputs_filt,1)
    return np.mean(y_est==labels_filt)
    

def get_stats(model,attack,param,outputs,names,labels):

    if attack == 'FGSM':
        names_list = [f'{model}_FGSM_ss_{i}' for i in [1,2,3]]
    else:
        names_list = [f'{model}_PGD_ns_{n}_ss_{s}' for n in [8,12,16] for s in [1,2,3]]

    names_set = set(names_list)

    mask = np.array([n in names_set for n in names])
    label_mask = np.array([l in names_set for l in label2name])

    ind = -1 if param == 'ss' else -3

    gt_vals = np.array([int(n.split('_')[ind]) for n,v in zip(names,mask) if v])
    vals = np.array([int(n.split('_')[ind]) if v else 0 for n,v in zip(label2name,label_mask)])

    outputs_filt = outputs[mask][:,label_mask]
    vals_filt = vals[label_mask]

    outputs_filt = np.exp(outputs_filt) / np.exp(outputs_filt).sum(1)[:,None]
    vals_est = (outputs_filt * vals_filt[None,:]).sum(1)
    name2arr_idx = {n:i for i,n in enumerate(names_list)}
    ndi_labels = np.array([name2arr_idx[n] for n,m in zip(names,mask) if m])
    index = np.arange(len(names_list))
    avg_vals = ndi.mean(vals_est,ndi_labels,index)
    mse_vals = ndi.mean((vals_est-gt_vals)**2,ndi_labels,index)
    
    return names_list, avg_vals, mse_vals, np.mean(mse_vals)


def conf_mtx(outputs,names,labels):
    mask = labels >= 0

    L = outputs.shape[1]
    y_pred = np.argmax(outputs,1)
    return np.bincount(L*labels[mask]+y_pred[mask],minlength=L**2).reshape((L,L))


def conf_mtx_w_merges(merges,outputs,names,labels):
    mask = labels >= 0
    labels_q = labels[mask]
    output_q = outputs[mask]
    L = merges.max() + 1
    pred = np.argmax(output_q,1)
    new_preds = merges[pred]
    new_labels = merges[labels_q]
    conf_mtx = np.bincount(new_labels*L+new_preds,minlength=L**2).reshape((L,L)).astype(float)
    conf_mtx /= conf_mtx.sum(1)[:,None]
    return conf_mtx, np.mean(np.diag(conf_mtx))



def conf_mtx_diff_models(outputs,names,labels):

    merges = -1*np.ones(outputs.shape[1],dtype=np.int64)
    for i, l in enumerate(label2name):
        if i == 0: merges[i] = 0
        if l.split('_')[0] == 'resnet50': merges[i] = 1
        if l.split('_')[0] == 'vgg16': merges[i] = 2

    names = ['orig', 'resnet50', 'vgg16']
    return conf_mtx_w_merges(merges,outputs,names,labels), names



def conf_mtx_diff_attack_and_model(outputs,names,labels):

    merges = -1*np.ones(outputs.shape[1],dtype=np.int64)
    for i,l in enumerate(label2name):
        if i == 0: merges[i] = 0
        elif l.split('_')[0] == 'resnet50':
            if l.split('_')[1] == 'FGSM': merges[i] = 1
            if l.split('_')[1] == 'PGD': merges[i] = 2
        else:
            if l.split('_')[1] == 'FGSM': merges[i] = 3
            if l.split('_')[1] == 'PGD': merges[i] = 4
    names = ['orig','resnet_FGSM','resnet_PGD','vgg_FGSM','vgg_PGD']
    return conf_mtx_w_merges(merges,outputs,names,labels), names



def conf_mtx_diff_attacks(outputs,names,labels):

    merges = -1*np.ones(outputs.shape[1],dtype=np.int64)
    for i, l in enumerate(label2name):
        
        if i == 0: merges[i] = 0
        elif l.split('_')[1] == 'FGSM': merges[i] = 1
        elif l.split('_')[1] == 'PGD': merges[i] = 2


    names = ['orig', 'FGSM', 'PGD']
    return conf_mtx_w_merges(merges,outputs,names,labels), names


def conf_mtx_binary(outputs,names,labels):
    merges = np.ones(outputs.shape[1],dtype=np.int64)
    merges[0] = 0
    return conf_mtx_w_merges(merges,outputs,names,labels)


print('stats estimation')
for model in ['vgg16','resnet50']:
    for args in [[model,'FGSM','ss'],[model,'PGD','ss'],[model,'PGD','ns']]:
        print(args)
        print(get_stats(*args,outputs,names,labels))

for args in [[model,'FGSM','ss'],[model,'PGD','ss'],[model,'PGD','ns']]:
    print('\n\n')
    mse = 0
    for model in ['vgg16','resnet50']:
        print(args)
        stats = get_stats(*args,outputs,names,labels)
        print(stats)
        mse += stats[-1]
    print('avg rmse:',np.sqrt(mse/2))


print('acc')
print(get_acc(outputs,names,labels))

print('conf matrix')
print(conf_mtx(outputs,names,labels))
print(label2name)

print('diff model')
print(conf_mtx_diff_models(outputs,names,labels))

print('diff attacks')
print(conf_mtx_diff_attacks(outputs,names,labels))


print('5 class')
print(conf_mtx_diff_attack_and_model(outputs,names,labels))

print('binary')
print(conf_mtx_binary(outputs,names,labels))





def to_latex(X):
    A = ' & '.join([' '] + [str(i) for i in range(10)])
    A += ' \\\\\n\\hline\n'
    X_str = [[f'{v:0.3f}' for v in row] for i,row in enumerate(X)] 
    A += '\\\\\n'.join([' & '.join([str(i),] + r) for i,r in enumerate(X_str)])
    return A


def latex_conf_mtx(X,labels):
    labels = [l.replace('_','\\_') for l in labels]
    labels_h = [l[:l.find('_')] + 'newline ' + l[(l.find('_')+1):] for l in labels]
    
    
    A = ' & '.join([' ',] + labels_h) + ' \\\\\n\\hline\n'
    
    
    
    X = X.astype(float)
    X /= X.sum(1)[:,None]
    X_str = [[f'{v:0.3f}' for v in row] for i,row in enumerate(X)]
    A += '\\\\\n'.join([' & '.join([l,] + r) for l,r in zip(labels,X_str)])
    return A


C = conf_mtx(outputs,names,labels)

print(latex_conf_mtx(C,label2name))









