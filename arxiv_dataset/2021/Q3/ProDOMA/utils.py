import argparse
import sys
import os
import random
import time
import subprocess
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel.data_parallel import data_parallel
from apex import amp

from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, roc_auc_score, fbeta_score, accuracy_score, recall_score, precision_score
import pandas as pd

from Bio import SeqIO

import tqdm

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools

from settings import *


def argparser():
    parser = argparse.ArgumentParser()
    # for model
    parser.add_argument(
        '--filter_sizes',
        default=[8, 12, 16, 20, 24, 28, 32, 36],
        type=int,
        nargs='+',
        help='Space seperated list of motif filter lengths. (ex, --filter_sizes 4 8 12)'
    )
    parser.add_argument(
        '--num_filters',
        default=256,
        type=int,
        help='number of filters per kernel'
    )
    parser.add_argument(
        '--num_hidden',
        type=int,
        default=512,
        help='Number of neurons in hidden layer.'
    )
    parser.add_argument(
        '--l2',
        type=float,
        default=0.001,
        help='(Lambda value / 2) of L2 regularizer on weights connected to last layer (0 to exclude).'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Rate for dropout.'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=86,
        help='Number of classes (families).'
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=1000,
        help='Length of input sequences.'
    )
    # for learning
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        help='Number of epochs to train.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size. Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--train_file',
        type=str,
        default=os.path.join(DATA_DIR, default_train_file),
        help='Directory for input data.'
    )

    parser.add_argument(
        '--valid_file',
        type=str,
        default=os.path.join(DATA_DIR, default_valid_file),
        help='Directory for input data.'
    )

    parser.add_argument(
        '--test_file',
        type=str,
        default=os.path.join(DATA_DIR, default_test_file),
        help='Directory for input data.'
    )

    parser.add_argument(
        '--ood_file',
        type=str,
        default=os.path.join(DATA_DIR, default_train_file),
        help='Directory for input ood data.'
    )

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Path to write checkpoint file.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=DATA_DIR,
        help='Directory for log data.'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=100,
        help='Interval of steps for logging.'
    )
    parser.add_argument(
        '--save_interval',
        type=int,
        default=100,
        help='Interval of steps for save model.'
    )
    # test
    parser.add_argument(
        '--fine_tuning',
        type=bool,
        default=False,
        help='If true, weight on last layer will not be restored.'
    )
    parser.add_argument(
        '--fine_tuning_layers',
        type=str,
        nargs='+',
        default=["fc2"],
        help='Which layers should be restored. Default is ["fc2"].'
    )
    parser.add_argument(
        '--save_prediction',
        type=str,
        default=None,
        help='Path to save prediction'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=1,
        help='Top k prediction for predict'
    )
    parser.add_argument(
        '--predict_file',
        type=str,
        default=None,
        help='path for predict data.'
    )
    parser.add_argument(
        '--iter_val', default=100, type=int, help='start epoch'
    )

    parser.add_argument(
        '--model', default='PepCNN', type=str, help='model name'
    )

    parser.add_argument(
        '--openset_model', default='ThresholdBaseline', type=str, help='model name for openset'
    )

    parser.add_argument(
        '--filter_topk', default=100, type=int, help='top_filters to print'
    )

    parser.add_argument(
        '--suffix', default='', type=str, help='suffix of the output folder'
    )
    parser.add_argument(
        '--truth', default=None, type=str, help='ground truth of the simulated reads'
    )

    parser.add_argument(
        '--embed_dim',
        type=int,
        default=10,
        help='embedding dimension.'
    )

    parser.add_argument(
        '--conv1_filter',
        type=int,
        default=3,
        help='the filter size for the first convolution of CNNDeep.'
    )

    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.0012,
        help='epsilon for ODIN.'
    )

    parser.add_argument(
        '--temperature',
        type=int,
        default=1000,
        help='temperature for ODIN.'
    )

    parser.add_argument(
        '--repeat',
        type=int,
        default=5,
        help='repeat for experiments.'
    )

    parser.add_argument(
        '--threshold',
        type=str,
        default=None ,
        help='path to threshold file.'
    )

    parser.add_argument('--swap_dim', action='store_true')
    
    parser.add_argument('--apex', action='store_true')
    try:
        FLAGS, unparsed = parser.parse_known_args()

    except:
        parser.print_help()
        sys.exit(1)

    # check validity
    # assert (len(FLAGS.filter_sizes) == len(FLAGS.num_filters))

    return FLAGS


def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    # print('model saved to %s' % checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    if optimizer:
        optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)


def multi_f_measure(probs, labels, threshold=0.5, beta=1):
    SMALL = 1e-6  # 0  #1e-12
    batch_size = probs.size()[0]

    # weather
    l = labels
    p = Variable((probs > threshold).float())

    num_pos = torch.sum(p, 1)
    num_pos_hat = torch.sum(l, 1)
    tp = torch.sum(torch.mul(l, p), 1)
    precise = tp / (num_pos + SMALL)
    recall = tp / (num_pos_hat + SMALL)

    fs = (1 + beta * beta) * precise * recall / (beta * beta * precise + recall + SMALL)
    f = fs.sum() / batch_size
    return f


def plot_confusion_matrix(cm,
                          target_names=None,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if target_names is None:
        target_names = list(range(cm.shape[0] + 1))

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def metric(logit, truth):
    with torch.no_grad():
        prob = F.softmax(logit, 1)
        value, top = prob.topk(3, dim=1, largest=True, sorted=True)
        correct = top.eq(truth.view(-1, 1).expand_as(top))
    return correct


def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs


GPCR_label = {'Adenosine': 0, 'Adrenergic': 1, 'Adrenocorticotropic': 2, 'Adrenomedullin': 3, 'Adrenoreceptor': 4,
              'Allatostatin': 5, 'AlphaFac': 6, 'Anaphylatoxin': 7, 'Angiotensin': 8, 'BLT2': 9, 'BOSS': 10,
              'Bombesin': 11, 'Bradykinin': 12, 'BrainSpec': 13, 'C5A': 14, 'Cadherin': 15, 'CalcLike': 16,
              'Calcitonin': 17, 'Cannabinoid': 18, 'Chemokine': 19, 'Cholecystokinin': 20, 'Corticotropin': 21,
              'Dopamine': 22, 'Duffy': 23, 'EMR1': 24, 'Endothelin': 25, 'ExtraCalc': 26, 'FollicleStim': 27,
              'GABA': 28, 'GRHR': 29, 'Galanin': 30, 'Gastric': 31, 'Glucagon': 32, 'GlutaMeta': 33,
              'Gonadotrophin': 34, 'Growth': 35, 'GrowthHorm': 36, 'Histamine': 37, 'Interleukin8': 38, 'Kiss1': 39,
              'Latrophilin': 40, 'LysoEdg2': 41, 'MelaninConc': 42, 'Melanocortin': 43, 'Melanocyte': 44, 'Melaton': 45,
              'Methuselah': 46, 'MuscAcetyl': 47, 'Muscarinicacetylcholine': 48, 'Neuromedin': 49, 'NeuromedinB-U': 50,
              'Neuropeptide': 51, 'NeuropeptideFF': 52, 'Neurotensin': 53, 'Octopamine': 54, 'Olfactory': 55,
              'Opoid': 56, 'Orexin': 57, 'Oxytocin': 58, 'PACAP': 59, 'Parathyroid': 60, 'Pheromone': 61,
              'Platelet': 62, 'Prokineticin': 63, 'Prolactin': 64, 'Prostacyclin': 65, 'Prostaglandin': 66,
              'Proteinase': 67, 'Purinergic': 68, 'PutPher': 69, 'Secretin': 70, 'Serotonin': 71, 'Somatostatin': 72,
              'SubstanceK': 73, 'SubstanceP': 74, 'Tachykinin': 75, 'Taste': 76, 'Thrombin': 77, 'Thyro': 78,
              'Thyrotropin': 79, 'Traceamine': 80, 'UrotensinII': 81, 'Vasoactive': 82, 'Vasopressin': 83,
              'Vasotocin': 84, 'cAMP': 85}

GPCR_family = {'Adenosine': 'ClassA', 'Adrenergic': 'ClassA', 'Adrenocorticotropic': 'ClassA',
               'Adrenomedullin': 'ClassA', 'Adrenoreceptor': 'ClassA', 'Allatostatin': 'ClassA', 'AlphaFac': 'ClassD',
               'Anaphylatoxin': 'ClassA', 'Angiotensin': 'ClassA', 'BLT2': 'ClassA', 'BOSS': 'ClassC',
               'Bombesin': 'ClassA', 'Bradykinin': 'ClassA', 'BrainSpec': 'ClassB', 'C5A': 'ClassA',
               'Cadherin': 'ClassB', 'CalcLike': 'ClassC', 'Calcitonin': 'ClassB', 'Cannabinoid': 'ClassA',
               'Chemokine': 'ClassA', 'Cholecystokinin': 'ClassA', 'Corticotropin': 'ClassB', 'Dopamine': 'ClassA',
               'Duffy': 'ClassA', 'EMR1': 'ClassB', 'Endothelin': 'ClassA', 'ExtraCalc': 'ClassC',
               'FollicleStim': 'ClassA', 'GABA': 'ClassC', 'GRHR': 'ClassA', 'Galanin': 'ClassA', 'Gastric': 'ClassB',
               'Glucagon': 'ClassB', 'GlutaMeta': 'ClassC', 'Gonadotrophin': 'ClassA', 'Growth': 'ClassA',
               'GrowthHorm': 'ClassB', 'Histamine': 'ClassA', 'Interleukin8': 'ClassA', 'Kiss1': 'ClassA',
               'Latrophilin': 'ClassB', 'LysoEdg2': 'ClassA', 'MelaninConc': 'ClassA', 'Melanocortin': 'ClassA',
               'Melanocyte': 'ClassA', 'Melaton': 'ClassA', 'Methuselah': 'ClassB', 'MuscAcetyl': 'ClassA',
               'Muscarinicacetylcholine': 'ClassA', 'Neuromedin': 'ClassA', 'NeuromedinB-U': 'ClassA',
               'Neuropeptide': 'ClassA', 'NeuropeptideFF': 'ClassA', 'Neurotensin': 'ClassA', 'Octopamine': 'ClassA',
               'Olfactory': 'ClassA', 'Opoid': 'ClassA', 'Orexin': 'ClassA', 'Oxytocin': 'ClassA', 'PACAP': 'ClassB',
               'Parathyroid': 'ClassB', 'Pheromone': 'ClassC', 'Platelet': 'ClassA', 'Prokineticin': 'ClassA',
               'Prolactin': 'ClassA', 'Prostacyclin': 'ClassA', 'Prostaglandin': 'ClassA', 'Proteinase': 'ClassA',
               'Purinergic': 'ClassA', 'PutPher': 'ClassC', 'Secretin': 'ClassB', 'Serotonin': 'ClassA',
               'Somatostatin': 'ClassA', 'SubstanceK': 'ClassA', 'SubstanceP': 'ClassA', 'Tachykinin': 'ClassA',
               'Taste': 'ClassC', 'Thrombin': 'ClassA', 'Thyro': 'ClassA', 'Thyrotropin': 'ClassA',
               'Traceamine': 'ClassA', 'UrotensinII': 'ClassA', 'Vasoactive': 'ClassB', 'Vasopressin': 'ClassA',
               'Vasotocin': 'ClassA', 'cAMP': 'ClassE'}

GPCR_subfamily = {'Adenosine': 'ClassA_Nucleotide', 'Adrenergic': 'ClassA_Adrenergic',
                  'Adrenocorticotropic': 'ClassA_Peptide', 'Adrenomedullin': 'ClassA_Peptide',
                  'Adrenoreceptor': 'ClassA_Amine', 'Allatostatin': 'ClassA_Peptide', 'AlphaFac': 'ClassD_Pheromone',
                  'Anaphylatoxin': 'ClassA_Anaphylatoxin', 'Angiotensin': 'ClassA_Peptide', 'BLT2': 'ClassA_Leuko',
                  'BOSS': 'ClassC_BOSS', 'Bombesin': 'ClassA_Peptide', 'Bradykinin': 'ClassA_Peptide',
                  'BrainSpec': 'ClassB_BrainSpec', 'C5A': 'ClassA_Peptide', 'Cadherin': 'ClassB_Cadherin',
                  'CalcLike': 'ClassC_CalcSense', 'Calcitonin': 'ClassB_Calcitonin',
                  'Cannabinoid': 'ClassA_Cannabinoid', 'Chemokine': 'ClassA_Peptide',
                  'Cholecystokinin': 'ClassA_Peptide', 'Corticotropin': 'ClassB_Corticotropin',
                  'Dopamine': 'ClassA_Amine', 'Duffy': 'ClassA_Peptide', 'EMR1': 'ClassB_EMR1',
                  'Endothelin': 'ClassA_Peptide', 'ExtraCalc': 'ClassC_CalcSense', 'FollicleStim': 'ClassA_Hormone',
                  'GABA': 'ClassC_GABA', 'GRHR': 'ClassA_GRHR', 'Galanin': 'ClassA_Peptide',
                  'Gastric': 'ClassB_Gastric', 'Glucagon': 'ClassB_Glucagon', 'GlutaMeta': 'ClassC_GlutaMeta',
                  'Gonadotrophin': 'ClassA_Hormone', 'Growth': 'ClassA_Thyro', 'GrowthHorm': 'ClassB_GrowthHorm',
                  'Histamine': 'ClassA_Amine', 'Interleukin8': 'ClassA_Interleukin8', 'Kiss1': 'ClassA_Peptide',
                  'Latrophilin': 'ClassB_Latrophilin', 'LysoEdg2': 'ClassA_Lyso', 'MelaninConc': 'ClassA_Peptide',
                  'Melanocortin': 'ClassA_Peptide', 'Melanocyte': 'ClassA_Peptide', 'Melaton': 'ClassA_Melaton',
                  'Methuselah': 'ClassB_Methuselah', 'MuscAcetyl': 'ClassA_Amine',
                  'Muscarinicacetylcholine': 'ClassA_Amine', 'Neuromedin': 'ClassA_Peptide',
                  'NeuromedinB-U': 'ClassA_Peptide', 'Neuropeptide': 'ClassA_Peptide',
                  'NeuropeptideFF': 'ClassA_Peptide', 'Neurotensin': 'ClassA_Peptide', 'Octopamine': 'ClassA_Amine',
                  'Olfactory': 'ClassA_Olfactory', 'Opoid': 'ClassA_Peptide', 'Orexin': 'ClassA_Peptide',
                  'Oxytocin': 'ClassA_Peptide', 'PACAP': 'ClassB_PACAP', 'Parathyroid': 'ClassB_Parathyroid',
                  'Pheromone': 'ClassC_CalcSense', 'Platelet': 'ClassA_Platelet', 'Prokineticin': 'ClassA_Peptide',
                  'Prolactin': 'ClassA_Peptide', 'Prostacyclin': 'ClassA_Prostanoid',
                  'Prostaglandin': 'ClassA_Prostanoid', 'Proteinase': 'ClassA_Peptide',
                  'Purinergic': 'ClassA_Nucleotide', 'PutPher': 'ClassC_PutPher', 'Secretin': 'ClassB_Secretin',
                  'Serotonin': 'ClassA_Amine', 'Somatostatin': 'ClassA_Peptide', 'SubstanceK': 'ClassA_Peptide',
                  'SubstanceP': 'ClassA_Peptide', 'Tachykinin': 'ClassA_Peptide', 'Taste': 'ClassC_Taste',
                  'Thrombin': 'ClassA_Peptide', 'Thyro': 'ClassA_Thyro', 'Thyrotropin': 'ClassA_Hormone',
                  'Traceamine': 'ClassA_Amine', 'UrotensinII': 'ClassA_Peptide', 'Vasoactive': 'ClassB_Vasocactive',
                  'Vasopressin': 'ClassA_Peptide', 'Vasotocin': 'ClassA_Peptide', 'cAMP': 'ClassE_cAMP'}

idx_to_GPCR_name = {0: 'ClassA_Nucleotide_Adenosine', 1: 'ClassA_Adrenergic_Adrenergic',
                    2: 'ClassA_Peptide_Adrenocorticotropic', 3: 'ClassA_Peptide_Adrenomedullin',
                    4: 'ClassA_Amine_Adrenoreceptor', 5: 'ClassA_Peptide_Allatostatin', 6: 'ClassD_Pheromone_AlphaFac',
                    7: 'ClassA_Anaphylatoxin_Anaphylatoxin', 8: 'ClassA_Peptide_Angiotensin', 9: 'ClassA_Leuko_BLT2',
                    10: 'ClassC_BOSS_BOSS', 11: 'ClassA_Peptide_Bombesin', 12: 'ClassA_Peptide_Bradykinin',
                    13: 'ClassB_BrainSpec_BrainSpec', 14: 'ClassA_Peptide_C5A', 15: 'ClassB_Cadherin_Cadherin',
                    16: 'ClassC_CalcSense_CalcLike', 17: 'ClassB_Calcitonin_Calcitonin',
                    18: 'ClassA_Cannabinoid_Cannabinoid', 19: 'ClassA_Peptide_Chemokine',
                    20: 'ClassA_Peptide_Cholecystokinin', 21: 'ClassB_Corticotropin_Corticotropin',
                    22: 'ClassA_Amine_Dopamine', 23: 'ClassA_Peptide_Duffy', 24: 'ClassB_EMR1_EMR1',
                    25: 'ClassA_Peptide_Endothelin', 26: 'ClassC_CalcSense_ExtraCalc',
                    27: 'ClassA_Hormone_FollicleStim', 28: 'ClassC_GABA_GABA', 29: 'ClassA_GRHR_GRHR',
                    30: 'ClassA_Peptide_Galanin', 31: 'ClassB_Gastric_Gastric', 32: 'ClassB_Glucagon_Glucagon',
                    33: 'ClassC_GlutaMeta_GlutaMeta', 34: 'ClassA_Hormone_Gonadotrophin', 35: 'ClassA_Thyro_Growth',
                    36: 'ClassB_GrowthHorm_GrowthHorm', 37: 'ClassA_Amine_Histamine',
                    38: 'ClassA_Interleukin8_Interleukin8', 39: 'ClassA_Peptide_Kiss1',
                    40: 'ClassB_Latrophilin_Latrophilin', 41: 'ClassA_Lyso_LysoEdg2', 42: 'ClassA_Peptide_MelaninConc',
                    43: 'ClassA_Peptide_Melanocortin', 44: 'ClassA_Peptide_Melanocyte', 45: 'ClassA_Melaton_Melaton',
                    46: 'ClassB_Methuselah_Methuselah', 47: 'ClassA_Amine_MuscAcetyl',
                    48: 'ClassA_Amine_Muscarinicacetylcholine', 49: 'ClassA_Peptide_Neuromedin',
                    50: 'ClassA_Peptide_NeuromedinB-U', 51: 'ClassA_Peptide_Neuropeptide',
                    52: 'ClassA_Peptide_NeuropeptideFF', 53: 'ClassA_Peptide_Neurotensin',
                    54: 'ClassA_Amine_Octopamine', 55: 'ClassA_Olfactory_Olfactory', 56: 'ClassA_Peptide_Opoid',
                    57: 'ClassA_Peptide_Orexin', 58: 'ClassA_Peptide_Oxytocin', 59: 'ClassB_PACAP_PACAP',
                    60: 'ClassB_Parathyroid_Parathyroid', 61: 'ClassC_CalcSense_Pheromone',
                    62: 'ClassA_Platelet_Platelet', 63: 'ClassA_Peptide_Prokineticin', 64: 'ClassA_Peptide_Prolactin',
                    65: 'ClassA_Prostanoid_Prostacyclin', 66: 'ClassA_Prostanoid_Prostaglandin',
                    67: 'ClassA_Peptide_Proteinase', 68: 'ClassA_Nucleotide_Purinergic', 69: 'ClassC_PutPher_PutPher',
                    70: 'ClassB_Secretin_Secretin', 71: 'ClassA_Amine_Serotonin', 72: 'ClassA_Peptide_Somatostatin',
                    73: 'ClassA_Peptide_SubstanceK', 74: 'ClassA_Peptide_SubstanceP', 75: 'ClassA_Peptide_Tachykinin',
                    76: 'ClassC_Taste_Taste', 77: 'ClassA_Peptide_Thrombin', 78: 'ClassA_Thyro_Thyro',
                    79: 'ClassA_Hormone_Thyrotropin', 80: 'ClassA_Amine_Traceamine', 81: 'ClassA_Peptide_UrotensinII',
                    82: 'ClassB_Vasocactive_Vasoactive', 83: 'ClassA_Peptide_Vasopressin',
                    84: 'ClassA_Peptide_Vasotocin', 85: 'ClassE_cAMP_cAMP'}
