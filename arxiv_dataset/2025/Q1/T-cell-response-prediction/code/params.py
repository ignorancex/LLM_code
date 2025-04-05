import os
import itertools

"""
This file specifies all hyperparameters.
Hyperparameters can be given as list to try out all possible combinations.
"""
mhc_alleles_I = ['default allele I', 'HLA-C*07:02', 'HLA-A*02:01', 'HLA-A*11:01', 'HLA-B*44:02', 'HLA-B*07:02', 'HLA-A*29:02',
                 'HLA-A*24:07', 'HLA-B*08:01', 'HLA-A*33:01']
mhc_alleles_II = ['default allele II', 'HLA-DRA1*01:01-DRB1*16:02', 'HLA-DRA1*01:01-DRB3*02:02', 'HLA-DPA1*01:03-DPB1*02:01',
                  'HLA-DRA1*01:01-DRB1*11:01', 'HLA-DRA1*01:01-DRB1*09:01', 'HLA-DRA1*01:01-DRB1*03:01', 'HLA-DRA1*01:01-DRB1*12:01',
                  'HLA-DQA1*01:02-DQB1*06:02', 'HLA-DRA1*01:01-DRB5*01:01']

peptide_sources = ['Human gammaherpesvirus 4', 'Human betaherpesvirus 6B', 'Dengue virus', 'Homo sapiens', 'unknown',
                   'Human betaherpesvirus 5', 'Severe acute respiratory syndrome coronavirus 2', 'Phleum pratense', 'Influenza A virus',
                   'Hepacivirus C', 'Vaccinia virus', 'Mycobacterium tuberculosis', 'Alphapapillomavirus 9', 'Plasmodium falciparum']

tensorboard_hparams_metrics = {
                               'early_stop_epochs': 'early stop epochs',

                               'ROC AUC - MHC I - MHC corrected - Human - no defaults/Validation': 'Human auROC I',
                               'ROC AUC - MHC II - MHC corrected - Human - no defaults/Validation': 'Human auROC II',

                               'AP - MHC I+II - MHC corrected - Human - no defaults/Validation': 'Human AP I+II',
                               'AP - MHC I - MHC corrected - Human - no defaults/Validation': 'Human AP I',
                               'AP - MHC II - MHC corrected - Human - no defaults/Validation': 'Human AP II',

                               'ROC AUC - MHC I+II - MHC+source corrected - no defaults/Validation': 'Valid auROC I+II',

                               'ROC AUC - MHC I - MHC+source corrected - no defaults/Validation': 'Valid auROC I',
                               'ROC AUC - MHC II - MHC+source corrected - no defaults/Validation': 'Valid auROC II',
                               'AP - MHC I - MHC+source corrected - no defaults/Validation': 'Valid AP I',
                               'AP - MHC II - MHC+source corrected - no defaults/Validation': 'Valid AP II',
                               'ROC AUC - All selected sources/Validation': 'All selected sources',

                               'AP - MHC I - All sources/Validation': 'AP all sources I',
                               'ROC AUC - MHC I - All sources/Validation': 'All sources I',
                               'ROC AUC - MHC I - Human betaherpesvirus 6B/Validation': 'Human betaherpes 6B I',
                               'ROC AUC - MHC I - Dengue virus/Validation': 'Dengue I',
                               'ROC AUC - MHC I - Homo sapiens/Validation': 'Homo sapiens I',
                               'ROC AUC - MHC I - Human betaherpesvirus 5/Validation': 'Human betaherpes 5 I',

                               'AP - MHC II - All sources/Validation': 'AP all sources II',
                               'ROC AUC - MHC II - All sources/Validation': 'All sources II',
                               'ROC AUC - MHC II - Human betaherpesvirus 6B/Validation': 'Human betaherpes 6B II',
                               'ROC AUC - MHC II - Dengue virus/Validation': 'Dengue II',
                               'ROC AUC - MHC II - Homo sapiens/Validation': 'Homo sapiens II',
                               'ROC AUC - MHC II - Human betaherpesvirus 5/Validation': 'Human betaherpes 5 II',

                            }

for mhc_allele in mhc_alleles_I + mhc_alleles_II:
    tensorboard_hparams_metrics[f'ROC AUC - {mhc_allele}/Validation'] = mhc_allele


def get_early_stopping_metric(params):
    mhc_class = params['mhc_class']
    if params['early_stopping_criterion'] == 'Human' or params['pep_source_train_selection'] == 'Homo sapiens':
        return f'ROC AUC - MHC {mhc_class} - MHC corrected - Human - no defaults/Validation'
    else:
        return f'ROC AUC - MHC {mhc_class} - MHC+source corrected - no defaults/Validation'

outer_k = 5
inner_k = 4
random_seed = 0
train_eval_frac = 1.0
eval_interval = 10

configurations = {
    # Fig 3 a)
    'transformer_da_grl_64dim_lr': {
        'description': 'transformer_da_grl_64dim_lr',

        # Data
        'train_eval_frac': 1,
        'dataset_folder': ['nested_cv_thesis'],
        'mhc_class': ['I+II'],
        'early_stopping_criterion': 'all',
        'pep_source_train_selection': ['all'],
        'one_allele_per_pep': False,
        'select_two_pep_sources': False,
        'select_pep_sources': ['All'],
        'evaluate_pep_sources': False,
        'select_alleles': ['All'],
        'eval_human': False,
        'evaluate_alleles': False,
        'MHC_class_eval': True,
        'sample_size': 'none',
        'features': 'OneHotPeptide',
        'feature_selector': 'peptide',

        # Input
        'embedding_norm': ['None'],
        'mhc_class_input': [False],
        'pep_source_input': [False],
        'allele_input': [False],
        'separate_embeddings': [False],
        # 'separate_aa_embeddings': [False],
        # 'separate_pos_embeddings': [True],
        # 'allele_pos_embeddings': [True, False],
        # 'source_pos_embeddings': [True, False],
        'mhc_class_pos_embeddings': [False],
        'positional_encoding': ['both'],


        'combine_sources': False,
        'human_pep_weight': [0],

        'human_domain_adapt': False,
        'epochs_adapt_human': [100],
        'human_source_classifier': ['2-layer mlp'],
        'human_source_hidden_dim': [10],
        'human_source_learning_rate': [0.01],
        'human_source_l2_reg': 0,
        'human_pos_weight': [1],
        'non_human_aa_embeddings': False,

        'adversary_input': ['transformer_output'],  #, 'final_activations'],
        'train_mhc_class': ['mhc_class'],

        'human_adv_domain_adapt': False,
        'source_params_only_debiasing': [True],
        'source_adaptation_weight': [0],
        'source_adaptation_tcell_weight': 0.05,


        # General model params
        'CNN': False,
        'model': ['PeptideTransformer'],
        'random_seed': [1, 2, 3],
        'l2_reg': [0],
        'dropout': [0.2],
        'learning_rate': [0.0005],
        'lr_step_size': 1,  # for learning rate decay
        'lr_gamma': [0.99],
        'eval_interval': 10,
        'epochs': 250,
        'embedding_dim': [64],
        'batch_size': 500,
        'batch_size_human': [100],

        # Bag-Of-AA
        'normalize_features': True,
        'hidden_dim': [20],

        # Protein LM embeddings
        'protein_embeddings': [False],
        'protein_l1_reg': [0.0005],
        'protein_l2_reg': [0],
        'protein_dropout': [0.5],
        'protein_aa_embeddings': [False],
        'protein_model': ['tape'],

        # Pretraining
        'pretrain_model': [False],
        'pretraining_epochs': [150],
        'reuse_pretrained_model': [True],

        # Old params
        'data_weighting': False,

        # Transformer
        'layer_norm_transformer': [False],
        'half_feedforward_dim': [False],
        'mhc_presentation_feature': False,
        'pep_source_layer': [False],
        'debiasing_layer': [False],
        'attention_allele_input': False,
        'attention_mhc_class_input': False,
        'attention_pep_source_input': False,
        'attention_mhc_source': [False],
        'apply_transformer': [True],
        'output_transform': ['2-layer mlp'],  # <-
        'num_attention_layers': [1],
        'attention_heads': [8],
        'concat_positional_encoding': [True],

        # Within subgroup loss
        'default_padding': [1],
        'ce_objective_weight': [1],
        'allele_objective_weight': [0],
        'source_objective_weight': [0],

        # Debiasing
        'debiasing': True,
        'separate_adversaries': [False],
        'adv_grad_reversal_loss': ['negate_gradient'],
        'embedding_debiasing': [False],
        'neg_adv_uniform_loss': [True],
        'clip_uniform_loss': [True],
        # 'adv_input': ['tcell_last_layer_pred_simple'],
        'evaluate_adv': False,
        'adversary_l2_reg': [0],
        'adversary_learning_rate': [0.005],
        'allele_adversary_weight': [0],
        'pep_source_adversary_weight': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26],
        'adv_tcell_label_input': [False],
        'increasing_adv_weights': [True],
        'adversary_hidden_dim': [32],
        # 'debiasing_technique': ['adv'],
        # 'tcell_adversary_learning_rate': [0.01],
        # 'allele_tcell_adv_weight': [0],
        # 'source_tcell_adv_weight': [0],
        # 'tcell_adversary_hidden_dim': [10],
        'project_gradients': [False],
        'shortcut_learning_rate': [0.05],
        'shortcut_epochs': 20,
        'shortcut_skip_connection': [False],
        'shortcut_hidden_dim': [10],
    },
    # Fig 3 b) + c)
    'transformer_da_grl_64dim_lr_tsne_0_da': {
        'description': 'transformer_da_grl_64dim_lr_tsne_0_da, fold 1',

        'tsne_subset': False,
        'tsne_visualization': True,
        'pred_visualization': True,

        # Data
        'train_eval_frac': 1,
        'dataset_folder': ['nested_cv_thesis'],
        'mhc_class': ['I+II'],
        'early_stopping_criterion': 'all',
        'pep_source_train_selection': ['all'],
        'one_allele_per_pep': False,
        'select_two_pep_sources': False,
        'select_pep_sources': ['All'],
        'evaluate_pep_sources': False,
        'select_alleles': ['All'],
        'eval_human': False,
        'evaluate_alleles': False,
        'MHC_class_eval': True,
        'sample_size': 'none',
        'features': 'OneHotPeptide',
        'feature_selector': 'peptide',

        # Input
        'embedding_norm': ['None'],
        'mhc_class_input': [False],
        'pep_source_input': [False],
        'allele_input': [False],
        'separate_embeddings': [False],
        # 'separate_aa_embeddings': [False],
        # 'separate_pos_embeddings': [True],
        # 'allele_pos_embeddings': [True, False],
        # 'source_pos_embeddings': [True, False],
        'mhc_class_pos_embeddings': [False],
        'positional_encoding': ['both'],


        'combine_sources': False,
        'human_pep_weight': [0],

        'human_domain_adapt': False,
        'epochs_adapt_human': [100],
        'human_source_classifier': ['2-layer mlp'],
        'human_source_hidden_dim': [10],
        'human_source_learning_rate': [0.01],
        'human_source_l2_reg': 0,
        'human_pos_weight': [1],
        'non_human_aa_embeddings': False,

        'adversary_input': ['transformer_output'],  #, 'final_activations'],
        'train_mhc_class': ['mhc_class'],

        'human_adv_domain_adapt': False,
        'source_params_only_debiasing': [True],
        'source_adaptation_weight': [0],
        'source_adaptation_tcell_weight': 0.05,


        # General model params
        'CNN': False,
        'model': ['PeptideTransformer'],
        'random_seed': [1],
        'l2_reg': [0],
        'dropout': [0.2],
        'learning_rate': [0.0005],
        'lr_step_size': 1,  # for learning rate decay
        'lr_gamma': [0.99],
        'eval_interval': 10,
        'epochs': 250,
        'embedding_dim': [64],
        'batch_size': 500,
        'batch_size_human': [100],

        # Bag-Of-AA
        'normalize_features': True,
        'hidden_dim': [20],

        # Protein LM embeddings
        'protein_embeddings': [False],
        'protein_l1_reg': [0.0005],
        'protein_l2_reg': [0],
        'protein_dropout': [0.5],
        'protein_aa_embeddings': [False],
        'protein_model': ['tape'],

        # Pretraining
        'pretrain_model': [False],
        'pretraining_epochs': [150],
        'reuse_pretrained_model': [True],

        # Old params
        'data_weighting': False,

        # Transformer
        'layer_norm_transformer': [False],
        'half_feedforward_dim': [False],
        'mhc_presentation_feature': False,
        'pep_source_layer': [False],
        'debiasing_layer': [False],
        'attention_allele_input': False,
        'attention_mhc_class_input': False,
        'attention_pep_source_input': False,
        'attention_mhc_source': [False],
        'apply_transformer': [True],
        'output_transform': ['2-layer mlp'],  # <-
        'num_attention_layers': [1],
        'attention_heads': [8],
        'concat_positional_encoding': [True],

        # Within subgroup loss
        'default_padding': [1],
        'ce_objective_weight': [1],
        'allele_objective_weight': [0],
        'source_objective_weight': [0],

        # Debiasing
        'debiasing': True,
        'separate_adversaries': [False],
        'adv_grad_reversal_loss': ['negate_gradient'],
        'embedding_debiasing': [False],
        'neg_adv_uniform_loss': [True],
        'clip_uniform_loss': [True],
        # 'adv_input': ['tcell_last_layer_pred_simple'],
        'evaluate_adv': False,
        'adversary_l2_reg': [0],
        'adversary_learning_rate': [0.005],
        'allele_adversary_weight': [0],
        'pep_source_adversary_weight': [0],
        'adv_tcell_label_input': [False],
        'increasing_adv_weights': [True],
        'adversary_hidden_dim': [32],
        # 'debiasing_technique': ['adv'],
        # 'tcell_adversary_learning_rate': [0.01],
        # 'allele_tcell_adv_weight': [0],
        # 'source_tcell_adv_weight': [0],
        # 'tcell_adversary_hidden_dim': [10],
        'project_gradients': [False],
        'shortcut_learning_rate': [0.05],
        'shortcut_epochs': 20,
        'shortcut_skip_connection': [False],
        'shortcut_hidden_dim': [10],
    },
    'transformer_da_grl_64dim_lr_tsne': {
        'description': 'transformer_da_grl_64dim_lr_tsne, fold 1',

        'tsne_subset': False,
        'tsne_visualization': True,
        'pred_visualization': True,

        # Data
        'train_eval_frac': 1,
        'dataset_folder': ['nested_cv_thesis'],
        'mhc_class': ['I+II'],
        'early_stopping_criterion': 'all',
        'pep_source_train_selection': ['all'],
        'one_allele_per_pep': False,
        'select_two_pep_sources': False,
        'select_pep_sources': ['All'],
        'evaluate_pep_sources': False,
        'select_alleles': ['All'],
        'eval_human': False,
        'evaluate_alleles': False,
        'MHC_class_eval': True,
        'sample_size': 'none',
        'features': 'OneHotPeptide',
        'feature_selector': 'peptide',

        # Input
        'embedding_norm': ['None'],
        'mhc_class_input': [False],
        'pep_source_input': [False],
        'allele_input': [False],
        'separate_embeddings': [False],
        # 'separate_aa_embeddings': [False],
        # 'separate_pos_embeddings': [True],
        # 'allele_pos_embeddings': [True, False],
        # 'source_pos_embeddings': [True, False],
        'mhc_class_pos_embeddings': [False],
        'positional_encoding': ['both'],


        'combine_sources': False,
        'human_pep_weight': [0],

        'human_domain_adapt': False,
        'epochs_adapt_human': [100],
        'human_source_classifier': ['2-layer mlp'],
        'human_source_hidden_dim': [10],
        'human_source_learning_rate': [0.01],
        'human_source_l2_reg': 0,
        'human_pos_weight': [1],
        'non_human_aa_embeddings': False,

        'adversary_input': ['transformer_output'],  #, 'final_activations'],
        'train_mhc_class': ['mhc_class'],

        'human_adv_domain_adapt': False,
        'source_params_only_debiasing': [True],
        'source_adaptation_weight': [0],
        'source_adaptation_tcell_weight': 0.05,


        # General model params
        'CNN': False,
        'model': ['PeptideTransformer'],
        'random_seed': [1],
        'l2_reg': [0],
        'dropout': [0.2],
        'learning_rate': [0.0005],
        'lr_step_size': 1,  # for learning rate decay
        'lr_gamma': [0.99],
        'eval_interval': 10,
        'epochs': 250,
        'embedding_dim': [64],
        'batch_size': 500,
        'batch_size_human': [100],

        # Bag-Of-AA
        'normalize_features': True,
        'hidden_dim': [20],

        # Protein LM embeddings
        'protein_embeddings': [False],
        'protein_l1_reg': [0.0005],
        'protein_l2_reg': [0],
        'protein_dropout': [0.5],
        'protein_aa_embeddings': [False],
        'protein_model': ['tape'],

        # Pretraining
        'pretrain_model': [False],
        'pretraining_epochs': [150],
        'reuse_pretrained_model': [True],

        # Old params
        'data_weighting': False,

        # Transformer
        'layer_norm_transformer': [False],
        'half_feedforward_dim': [False],
        'mhc_presentation_feature': False,
        'pep_source_layer': [False],
        'debiasing_layer': [False],
        'attention_allele_input': False,
        'attention_mhc_class_input': False,
        'attention_pep_source_input': False,
        'attention_mhc_source': [False],
        'apply_transformer': [True],
        'output_transform': ['2-layer mlp'],  # <-
        'num_attention_layers': [1],
        'attention_heads': [8],
        'concat_positional_encoding': [True],

        # Within subgroup loss
        'default_padding': [1],
        'ce_objective_weight': [1],
        'allele_objective_weight': [0],
        'source_objective_weight': [0],

        # Debiasing
        'debiasing': True,
        'separate_adversaries': [False],
        'adv_grad_reversal_loss': ['negate_gradient'],
        'embedding_debiasing': [False],
        'neg_adv_uniform_loss': [True],
        'clip_uniform_loss': [True],
        # 'adv_input': ['tcell_last_layer_pred_simple'],
        'evaluate_adv': False,
        'adversary_l2_reg': [0],
        'adversary_learning_rate': [0.005],
        'allele_adversary_weight': [0],
        'pep_source_adversary_weight': [10],
        'adv_tcell_label_input': [False],
        'increasing_adv_weights': [True],
        'adversary_hidden_dim': [32],
        # 'debiasing_technique': ['adv'],
        # 'tcell_adversary_learning_rate': [0.01],
        # 'allele_tcell_adv_weight': [0],
        # 'source_tcell_adv_weight': [0],
        # 'tcell_adversary_hidden_dim': [10],
        'project_gradients': [False],
        'shortcut_learning_rate': [0.05],
        'shortcut_epochs': 20,
        'shortcut_skip_connection': [False],
        'shortcut_hidden_dim': [10],
    },

    # Fig 4 a) + S5 (top and bottom)
    'transformer_multi_source_paper_valid_re': {
        'description': 'Batch size 500, decay every 2 epochs',

        # Data
        'train_eval_frac': 1,
        'dataset_folder': ['nested_cv_thesis'],
        'mhc_class': ['I+II'],
        'early_stopping_criterion': 'all',
        'pep_source_train_selection': ['all'],
        'one_allele_per_pep': False,
        'select_two_pep_sources': False,
        'select_pep_sources': ['All'],
        'evaluate_pep_sources': True,
        'select_alleles': ['All'],
        'eval_human': True,
        'evaluate_alleles': False,
        'MHC_class_eval': True,
        'sample_size': 'none',
        'features': 'OneHotPeptide',
        'feature_selector': 'peptide',

        'embedding_norm': ['None'],
        'mhc_class_input': [False],
        'pep_source_input': [False],
        'allele_input': [False],
        'separate_aa_embeddings': [False],
        'separate_pos_embeddings': [False],
        'allele_pos_embeddings': [False],
        'source_pos_embeddings': [False],
        'mhc_class_pos_embeddings': [False],
        'positional_encoding': ['both'],

        'combine_sources': False,
        'human_pep_weight': [0],

        'human_domain_adapt': 'False',
        'epochs_adapt_human': [0],
        'human_source_classifier': ['2-layer mlp'],
        'human_source_hidden_dim': [10],
        'human_source_learning_rate': [0.01],
        'human_source_l2_reg': 0,
        'human_pos_weight': [1],
        'non_human_aa_embeddings': False,

        'adversary_input': ['final_activations'],
        'train_mhc_class': ['mhc_class'],

        'human_adv_domain_adapt': False,
        'source_params_only_debiasing': [True],
        'source_adaptation_weight': [0],
        'source_adaptation_tcell_weight': 0.05,


        # General model params
        'CNN': False,
        'model': ['PeptideTransformer'],
        'random_seed': [1, 2, 3],
        'l2_reg': [0],
        'dropout': [0.1, 0.2, 0.3, 0.4],
        'learning_rate': [0.01],
        'lr_step_size': [1],  # for learning rate decay
        'lr_gamma': [1.],
        'early_stopping': True,
        'early_stopping_moving_average': False,
        'epochs': 250,
        'eval_interval': 10,
        'embedding_dim': [32, 48],  # <-
        'batch_size': 500,
        'batch_size_human': [100],

        # Bag-Of-AA
        'normalize_features': True,
        'hidden_dim': [20],

        # Protein LM embeddings
        'protein_embeddings': [False],
        'protein_l1_reg': [0.0005],
        'protein_l2_reg': [0],
        'protein_dropout': [0.5],
        'protein_aa_embeddings': [False],
        'protein_model': ['tape'],

        # Pretraining
        'pretrain_model': [False],
        'pretraining_epochs': [100],
        'reuse_pretrained_model': [True],

        # Old params
        'data_weighting': False,

        # Transformer
        'layer_norm_transformer': [False],
        'half_feedforward_dim': [False],
        'mhc_presentation_feature': False,
        'pep_source_layer': [False],
        'pep_source_pos_input': [True],
        'debiasing_layer': [False],
        'attention_allele_input': False,
        'attention_mhc_class_input': False,
        'attention_pep_source_input': False,
        'attention_mhc_source': [False],
        'apply_transformer': [True],
        'output_transform': ['2-layer mlp'],
        'num_attention_layers': [1],
        'attention_heads': [16],
        'concat_positional_encoding': [True],

        # Within subgroup loss
        'default_padding': [1],
        'ce_objective_weight': [1],
        'allele_objective_weight': [0],
        'source_objective_weight': [0],

        # Debiasing
        'debiasing': False,
        'shortcut_learning_rate': [0.05],
        'shortcut_epochs': 20,
        'adversary_l2_reg': [0],
        'evaluate_adv': False,
        'adv_input': ['tcell_last_layer_pred_simple'],  #, 'tcell_last_layer_pred_label', 'tcell_last_layer_label', 'tcell_last_layer'],
        'debiasing_technique': ['adv'],
        'adversary_learning_rate': [0.01],
        'tcell_adversary_learning_rate': [0.01],
        'allele_adversary_weight': [0],
        'pep_source_adversary_weight': [0],
        'allele_tcell_adv_weight': [0],
        'source_tcell_adv_weight': [0],
        'increasing_adv_weights': [False],
        'project_gradients': [False],
        'adv_tcell_label_input': [False],
        'shortcut_skip_connection': [False],
        'adversary_hidden_dim': [32],
        'tcell_adversary_hidden_dim': [10],
        'shortcut_hidden_dim': [10],
    },
    # transformer trained on one peptide source with pretraining on all sources
    'transformer_pretrained_one_source_paper_valid_re_I': { # h
        'description': 'transformer_pretrained_one_source_paper_I',

        # Data
        'train_eval_frac': 1,
        'dataset_folder': ['nested_cv_thesis'],
        'mhc_class': ['I'],
        'early_stopping_criterion': 'All',
        'pep_source_train_selection': ['Human immunodeficiency virus 1', 'Trypanosoma cruzi', 'Homo sapiens', 'unknown',
                                       'Human betaherpesvirus 5', 'Hepacivirus C', 'Severe acute respiratory syndrome coronavirus 2'],
        'one_allele_per_pep': False,
        'select_two_pep_sources': False,
        'select_pep_sources': ['All'],
        'evaluate_pep_sources': False,
        'select_alleles': ['All'],
        'eval_human': True,
        'evaluate_alleles': False,
        'MHC_class_eval': True,
        'sample_size': 'none',
        'features': 'OneHotPeptide',
        'feature_selector': 'peptide',

        'embedding_norm': ['none'],
        'mhc_class_input': [False],
        'pep_source_input': [False],
        'allele_input': [False],
        'separate_aa_embeddings': [False],
        'separate_pos_embeddings': [False],
        'allele_pos_embeddings': [False],
        'source_pos_embeddings': [False],
        'mhc_class_pos_embeddings': [False],
        'positional_encoding': ['both'],

        'combine_sources': False,
        'human_pep_weight': [0],

        'human_domain_adapt': 'False',
        'epochs_adapt_human': [0],
        'human_source_classifier': ['2-layer mlp'],
        'human_source_hidden_dim': [10],
        'human_source_learning_rate': [0.01],
        'human_source_l2_reg': 0,
        'human_pos_weight': [1],
        'non_human_aa_embeddings': False,

        'adversary_input': ['final_activations'],
        'train_mhc_class': ['I+II'],

        'human_adv_domain_adapt': False,
        'source_params_only_debiasing': [True],
        'source_adaptation_weight': [0],
        'source_adaptation_tcell_weight': 0.05,


        # General model params
        'CNN': False,
        'model': ['PeptideTransformer'],
        'random_seed': [1, 2, 3],
        'l2_reg': [0],
        'dropout': [0.1, 0.2, 0.3, 0.4],
        'learning_rate': [0.01],
        'lr_step_size': [1],  # for learning rate decay
        'lr_gamma': [1.],
        'early_stopping': True,
        'early_stopping_moving_average': False,
        'eval_interval': 10,
        'epochs': 250,
        'embedding_dim': [32, 48],  # <-
        'batch_size': 500,
        'batch_size_human': [100],

        # Bag-Of-AA
        'normalize_features': True,
        'hidden_dim': [20],

        # Protein LM embeddings
        'protein_embeddings': [False],
        'protein_l1_reg': [0.0005],
        'protein_l2_reg': [0],
        'protein_dropout': [0.5],
        'protein_aa_embeddings': [False],
        'protein_model': ['tape'],

        # Pretraining
        'pretrain_model': [True],
        'pretraining_epochs': [100],
        'reuse_pretrained_model': [True],
        'pretrain_no_early_stop': True,

        # Old params
        'data_weighting': False,

        # Transformer
        'layer_norm_transformer': [False],
        'half_feedforward_dim': [False],
        'mhc_presentation_feature': False,
        'pep_source_layer': [False],
        'pep_source_pos_input': [True],
        'debiasing_layer': [False],
        'attention_allele_input': False,
        'attention_mhc_class_input': False,
        'attention_pep_source_input': False,
        'attention_mhc_source': [False],
        'apply_transformer': [True],
        'output_transform': ['2-layer mlp'],
        'num_attention_layers': [1],
        'attention_heads': [16],
        'concat_positional_encoding': [True],

        # Within subgroup loss
        'default_padding': [1],
        'ce_objective_weight': [1],
        'allele_objective_weight': [0],
        'source_objective_weight': [0],

        # Debiasing
        'debiasing': False,
        'shortcut_learning_rate': [0.05],
        'shortcut_epochs': 20,
        'adversary_l2_reg': [0],
        'evaluate_adv': False,
        'adv_input': ['tcell_last_layer_pred_simple'],  #, 'tcell_last_layer_pred_label', 'tcell_last_layer_label', 'tcell_last_layer'],
        'debiasing_technique': ['adv'],
        'adversary_learning_rate': [0.01],
        'tcell_adversary_learning_rate': [0.01],
        'allele_adversary_weight': [0],
        'pep_source_adversary_weight': [0],
        'allele_tcell_adv_weight': [0],
        'source_tcell_adv_weight': [0],
        'increasing_adv_weights': [False],
        'project_gradients': [False],
        'adv_tcell_label_input': [False],
        'shortcut_skip_connection': [False],
        'adversary_hidden_dim': [32],
        'tcell_adversary_hidden_dim': [10],
        'shortcut_hidden_dim': [10],
    },
    'transformer_pretrained_one_source_paper_valid_re_II': { # h
        'description': 'transformer_pretrained_one_source_paper_II',

        # Data
        'train_eval_frac': 1,
        'dataset_folder': ['nested_cv_thesis'],
        'mhc_class': ['II'],
        'early_stopping_criterion': 'All',
        'pep_source_train_selection': ['Vaccinia virus', 'unknown', 'Human betaherpesvirus 6B', 'Dengue virus',
                                       'Severe acute respiratory syndrome coronavirus 2', 'Phleum pratense',
                                       'Human betaherpesvirus 5', 'Homo sapiens'],
        'one_allele_per_pep': False,
        'select_two_pep_sources': False,
        'select_pep_sources': ['All'],
        'evaluate_pep_sources': False,
        'select_alleles': ['All'],
        'eval_human': True,
        'evaluate_alleles': False,
        'MHC_class_eval': True,
        'sample_size': 'none',
        'features': 'OneHotPeptide',
        'feature_selector': 'peptide',

        'embedding_norm': ['none'],
        'mhc_class_input': [False],
        'pep_source_input': [False],
        'allele_input': [False],
        'separate_aa_embeddings': [False],
        'separate_pos_embeddings': [False],
        'allele_pos_embeddings': [False],
        'source_pos_embeddings': [False],
        'mhc_class_pos_embeddings': [False],
        'positional_encoding': ['both'],

        'combine_sources': False,
        'human_pep_weight': [0],

        'human_domain_adapt': 'False',
        'epochs_adapt_human': [0],
        'human_source_classifier': ['2-layer mlp'],
        'human_source_hidden_dim': [10],
        'human_source_learning_rate': [0.01],
        'human_source_l2_reg': 0,
        'human_pos_weight': [1],
        'non_human_aa_embeddings': False,

        'adversary_input': ['final_activations'],
        'train_mhc_class': ['I+II'],

        'human_adv_domain_adapt': False,
        'source_params_only_debiasing': [True],
        'source_adaptation_weight': [0],
        'source_adaptation_tcell_weight': 0.05,


        # General model params
        'CNN': False,
        'model': ['PeptideTransformer'],
        'random_seed': [1, 2, 3],
        'l2_reg': [0],
        'dropout': [0.1, 0.2, 0.3, 0.4],
        'learning_rate': [0.01],
        'lr_step_size': [1],  # for learning rate decay
        'lr_gamma': [1.],
        'early_stopping': True,
        'early_stopping_moving_average': False,
        'eval_interval': 10,
        'epochs': 250,
        'embedding_dim': [32, 48],  # <-
        'batch_size': 500,
        'batch_size_human': [100],

        # Bag-Of-AA
        'normalize_features': True,
        'hidden_dim': [20],

        # Protein LM embeddings
        'protein_embeddings': [False],
        'protein_l1_reg': [0.0005],
        'protein_l2_reg': [0],
        'protein_dropout': [0.5],
        'protein_aa_embeddings': [False],
        'protein_model': ['tape'],

        # Pretraining
        'pretrain_model': [True],
        'pretraining_epochs': [100],
        'reuse_pretrained_model': [True],
        'pretrain_no_early_stop': True,

        # Old params
        'data_weighting': False,

        # Transformer
        'layer_norm_transformer': [False],
        'half_feedforward_dim': [False],
        'mhc_presentation_feature': False,
        'pep_source_layer': [False],
        'pep_source_pos_input': [True],
        'debiasing_layer': [False],
        'attention_allele_input': False,
        'attention_mhc_class_input': False,
        'attention_pep_source_input': False,
        'attention_mhc_source': [False],
        'apply_transformer': [True],
        'output_transform': ['2-layer mlp'],
        'num_attention_layers': [1],
        'attention_heads': [16],
        'concat_positional_encoding': [True],

        # Within subgroup loss
        'default_padding': [1],
        'ce_objective_weight': [1],
        'allele_objective_weight': [0],
        'source_objective_weight': [0],

        # Debiasing
        'debiasing': False,
        'shortcut_learning_rate': [0.05],
        'shortcut_epochs': 20,
        'adversary_l2_reg': [0],
        'evaluate_adv': False,
        'adv_input': ['tcell_last_layer_pred_simple'],  #, 'tcell_last_layer_pred_label', 'tcell_last_layer_label', 'tcell_last_layer'],
        'debiasing_technique': ['adv'],
        'adversary_learning_rate': [0.01],
        'tcell_adversary_learning_rate': [0.01],
        'allele_adversary_weight': [0],
        'pep_source_adversary_weight': [0],
        'allele_tcell_adv_weight': [0],
        'source_tcell_adv_weight': [0],
        'increasing_adv_weights': [False],
        'project_gradients': [False],
        'adv_tcell_label_input': [False],
        'shortcut_skip_connection': [False],
        'adversary_hidden_dim': [32],
        'tcell_adversary_hidden_dim': [10],
        'shortcut_hidden_dim': [10],
    },
    'transformer_pretrained_one_source_paper_valid_re_I_save': { # h
        'description': 'transformer_pretrained_one_source_paper_valid_re_I for model saving',

        # Data
        'train_eval_frac': 1,
        'dataset_folder': ['nested_cv_thesis'],
        'mhc_class': ['I'],
        'early_stopping_criterion': 'All',
        'pep_source_train_selection': ['Human immunodeficiency virus 1', 'Trypanosoma cruzi', 'Homo sapiens', 'unknown',
                                       'Human betaherpesvirus 5', 'Hepacivirus C', 'Severe acute respiratory syndrome coronavirus 2'],
        'one_allele_per_pep': False,
        'select_two_pep_sources': False,
        'select_pep_sources': ['All'],
        'evaluate_pep_sources': False,
        'select_alleles': ['All'],
        'eval_human': True,
        'evaluate_alleles': False,
        'MHC_class_eval': True,
        'sample_size': 'none',
        'features': 'OneHotPeptide',
        'feature_selector': 'peptide',

        'embedding_norm': ['none'],
        'mhc_class_input': [False],
        'pep_source_input': [False],
        'allele_input': [False],
        'separate_aa_embeddings': [False],
        'separate_pos_embeddings': [False],
        'allele_pos_embeddings': [False],
        'source_pos_embeddings': [False],
        'mhc_class_pos_embeddings': [False],
        'positional_encoding': ['both'],

        'combine_sources': False,
        'human_pep_weight': [0],

        'human_domain_adapt': 'False',
        'epochs_adapt_human': [0],
        'human_source_classifier': ['2-layer mlp'],
        'human_source_hidden_dim': [10],
        'human_source_learning_rate': [0.01],
        'human_source_l2_reg': 0,
        'human_pos_weight': [1],
        'non_human_aa_embeddings': False,

        'adversary_input': ['final_activations'],
        'train_mhc_class': ['I+II'],

        'human_adv_domain_adapt': False,
        'source_params_only_debiasing': [True],
        'source_adaptation_weight': [0],
        'source_adaptation_tcell_weight': 0.05,


        # General model params
        'CNN': False,
        'model': ['PeptideTransformer'],
        'random_seed': [1],
        'l2_reg': [0],
        'dropout': [0.1, 0.2, 0.3, 0.4],
        'learning_rate': [0.01],
        'lr_step_size': [1],  # for learning rate decay
        'lr_gamma': [1.],
        'early_stopping': True,
        'early_stopping_moving_average': False,
        'eval_interval': 10,
        'epochs': 250,
        'embedding_dim': [32, 48],  # <-
        'batch_size': 500,
        'batch_size_human': [100],

        # Bag-Of-AA
        'normalize_features': True,
        'hidden_dim': [20],

        # Protein LM embeddings
        'protein_embeddings': [False],
        'protein_l1_reg': [0.0005],
        'protein_l2_reg': [0],
        'protein_dropout': [0.5],
        'protein_aa_embeddings': [False],
        'protein_model': ['tape'],

        # Pretraining
        'pretrain_model': [True],
        'pretraining_epochs': [100],
        'reuse_pretrained_model': [True],
        'pretrain_no_early_stop': True,

        # Old params
        'data_weighting': False,

        # Transformer
        'layer_norm_transformer': [False],
        'half_feedforward_dim': [False],
        'mhc_presentation_feature': False,
        'pep_source_layer': [False],
        'pep_source_pos_input': [True],
        'debiasing_layer': [False],
        'attention_allele_input': False,
        'attention_mhc_class_input': False,
        'attention_pep_source_input': False,
        'attention_mhc_source': [False],
        'apply_transformer': [True],
        'output_transform': ['2-layer mlp'],
        'num_attention_layers': [1],
        'attention_heads': [16],
        'concat_positional_encoding': [True],

        # Within subgroup loss
        'default_padding': [1],
        'ce_objective_weight': [1],
        'allele_objective_weight': [0],
        'source_objective_weight': [0],

        # Debiasing
        'debiasing': False,
        'shortcut_learning_rate': [0.05],
        'shortcut_epochs': 20,
        'adversary_l2_reg': [0],
        'evaluate_adv': False,
        'adv_input': ['tcell_last_layer_pred_simple'],  #, 'tcell_last_layer_pred_label', 'tcell_last_layer_label', 'tcell_last_layer'],
        'debiasing_technique': ['adv'],
        'adversary_learning_rate': [0.01],
        'tcell_adversary_learning_rate': [0.01],
        'allele_adversary_weight': [0],
        'pep_source_adversary_weight': [0],
        'allele_tcell_adv_weight': [0],
        'source_tcell_adv_weight': [0],
        'increasing_adv_weights': [False],
        'project_gradients': [False],
        'adv_tcell_label_input': [False],
        'shortcut_skip_connection': [False],
        'adversary_hidden_dim': [32],
        'tcell_adversary_hidden_dim': [10],
        'shortcut_hidden_dim': [10],
    },
    'baseline_mlp_human': {
        'description': '',

        # Data
        'train_eval_frac': 0.5,
        'dataset_folder': ['nested_cv_thesis'],
        'mhc_class': ['I', 'II'],
        'early_stopping_criterion': 'all',
        'pep_source_train_selection': 'all',
        'one_allele_per_pep': False,
        'select_alleles': ['All'],
        'select_two_pep_sources': False,
        'select_pep_sources': ['Homo sapiens'],
        'evaluate_pep_sources': False,
        'eval_human': True,
        'evaluate_alleles': False,
        'MHC_class_eval': True,
        'sample_size': 'none',
        'features': 'BagOfAA',
        'feature_selector': 'peptide',

        # General model params
        'CNN': False,
        'model': ['MLP'],
        'random_seed': [1, 2, 3],
        'l2_reg': [0],
        'dropout': [0.1, 0.2, 0.3, 0.4],
        'learning_rate': [0.01, 0.1, 0.2],
        'lr_step_size': 2,  # for learning rate decay
        'lr_gamma': [0.99, 1.],
        'eval_interval': 10,
        'epochs': 250,
        'embedding_dim': [16],
        'batch_size': 100,
        'batch_size_human': [100],

        # Bag-Of-AA
        'normalize_features': True,
        'hidden_dim': [64, 96, 128],

        'protein_embeddings': False,
        'pretrain_model': False,
        'pretraining_epochs': [150],
        'combine_sources': False,
        'human_domain_adapt': False,
        'human_adv_domain_adapt': False,
        'train_mhc_class': 'mhc_class',
        'shortcut_skip_connection': False,

        'debiasing': False,
        'evaluate_adv': False,

    },

    # Fig 5
    # transformer on human peptides with pretraining on all sources
    'transformer_pretrained_human_paper': {
        'description': 'Batch size 100, decay every 2 epochs, Training on human peptides',

        # Data
        'train_eval_frac': 1,
        'dataset_folder': ['nested_cv_thesis'],
        'mhc_class': ['II', 'I'],
        'early_stopping_criterion': 'Human',
        'pep_source_train_selection': ['Homo sapiens'],
        'one_allele_per_pep': False,
        'select_two_pep_sources': False,
        'select_pep_sources': ['All'],
        'evaluate_pep_sources': False,
        'select_alleles': ['All'],
        'eval_human': True,
        'evaluate_alleles': False,
        'MHC_class_eval': True,
        'sample_size': 'none',
        'features': 'OneHotPeptide',
        'feature_selector': 'peptide',

        'embedding_norm': ['none'],
        'mhc_class_input': [False],
        'pep_source_input': [False],
        'allele_input': [False],
        'separate_aa_embeddings': [False],
        'separate_pos_embeddings': [False],
        'allele_pos_embeddings': [False],
        'source_pos_embeddings': [False],
        'mhc_class_pos_embeddings': [False],
        'positional_encoding': ['both'],

        'combine_sources': False,
        'human_pep_weight': [0],

        'human_domain_adapt': 'False',
        'epochs_adapt_human': [0],
        'human_source_classifier': ['2-layer mlp'],
        'human_source_hidden_dim': [10],
        'human_source_learning_rate': [0.01],
        'human_source_l2_reg': 0,
        'human_pos_weight': [1],
        'non_human_aa_embeddings': False,

        'adversary_input': ['final_activations'],
        'train_mhc_class': ['mhc_class', 'I+II'],

        'human_adv_domain_adapt': False,
        'source_params_only_debiasing': [True],
        'source_adaptation_weight': [0],
        'source_adaptation_tcell_weight': 0.05,


        # General model params
        'CNN': False,
        'model': ['PeptideTransformer'],
        'random_seed': [1, 2, 3],
        'l2_reg': [0],
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [0.01],
        'lr_step_size': [1],  # for learning rate decay
        'lr_gamma': [1.],
        'early_stopping': True,
        'epochs': 250,
        'embedding_dim': [32, 48],  # <-
        'batch_size': 500,
        'batch_size_human': [100],
        'eval_interval': 10,

        # Bag-Of-AA
        'normalize_features': True,
        'hidden_dim': [20],

        # Protein LM embeddings
        'protein_embeddings': [False],
        'protein_l1_reg': [0.0005],
        'protein_l2_reg': [0],
        'protein_dropout': [0.5],
        'protein_aa_embeddings': [False],
        'protein_model': ['tape'],

        # Pretraining
        'pretrain_model': [True],
        'pretraining_epochs': [100],
        'reuse_pretrained_model': [True],

        # Old params
        'data_weighting': False,

        # Transformer
        'layer_norm_transformer': [False],
        'half_feedforward_dim': [False],
        'mhc_presentation_feature': False,
        'pep_source_layer': [False],
        'pep_source_pos_input': [True],
        'debiasing_layer': [False],
        'attention_allele_input': False,
        'attention_mhc_class_input': False,
        'attention_pep_source_input': False,
        'attention_mhc_source': [False],
        'apply_transformer': [True],
        'output_transform': ['2-layer mlp'],
        'num_attention_layers': [1],
        'attention_heads': [16],
        'concat_positional_encoding': [True],

        # Within subgroup loss
        'default_padding': [1],
        'ce_objective_weight': [1],
        'allele_objective_weight': [0],
        'source_objective_weight': [0],

        # Debiasing
        'debiasing': False,
        'shortcut_learning_rate': [0.05],
        'shortcut_epochs': 20,
        'adversary_l2_reg': [0],
        'evaluate_adv': False,
        'adv_input': ['tcell_last_layer_pred_simple'],  #, 'tcell_last_layer_pred_label', 'tcell_last_layer_label', 'tcell_last_layer'],
        'debiasing_technique': ['adv'],
        'adversary_learning_rate': [0.01],
        'tcell_adversary_learning_rate': [0.01],
        'allele_adversary_weight': [0],
        'pep_source_adversary_weight': [0],
        'allele_tcell_adv_weight': [0],
        'source_tcell_adv_weight': [0],
        'increasing_adv_weights': [False],
        'project_gradients': [False],
        'adv_tcell_label_input': [False],
        'shortcut_skip_connection': [False],
        'adversary_hidden_dim': [32],
        'tcell_adversary_hidden_dim': [10],
        'shortcut_hidden_dim': [10],
    },

    # S4
    # paper: transformer on individual MHC alleles
    'transformer_mhc_alleles_paper_ma_es': {
        'description': 'transformer_mhc_alleles_paper_ma_es, moving average for early stopping',

        # Data
        'train_eval_frac': 0.5,
        'dataset_folder': ['nested_cv_thesis'],
        'mhc_class': ['I+II'],
        'early_stopping_criterion': 'All',
        'pep_source_train_selection': ['all'],
        'one_allele_per_pep': False,
        'select_two_pep_sources': False,
        'select_pep_sources': ['All'],
        'select_alleles': mhc_alleles_I + mhc_alleles_II + ['MHC Class I+II', 'MHC Class I', 'MHC Class II'],
        'evaluate_pep_sources': False,
        'eval_human': False,
        'evaluate_alleles': True,
        'MHC_class_eval': True,
        'sample_size': 'none',
        'features': 'OneHotPeptide',
        'feature_selector': 'peptide',

        # Input
        'embedding_norm': ['None'],
        'mhc_class_input': [False],
        'pep_source_input': [False],
        'allele_input': [False],
        'separate_aa_embeddings': [False],
        'separate_pos_embeddings': [False],
        'allele_pos_embeddings': [False],
        'source_pos_embeddings': [False],
        'mhc_class_pos_embeddings': [False],
        'positional_encoding': ['both'],

        'combine_sources': False,
        'human_pep_weight': [0],

        'human_domain_adapt': False,
        'epochs_adapt_human': [100],
        'human_source_classifier': ['2-layer mlp'],
        'human_source_hidden_dim': [10],
        'human_source_learning_rate': [0.01],
        'human_source_l2_reg': 0,
        'human_pos_weight': [1],

        'adversary_input': ['final_activations'],
        'train_mhc_class': ['mhc_class'],

        'human_adv_domain_adapt': False,
        'source_params_only_debiasing': [True],
        'source_adaptation_weight': [0],
        'source_adaptation_tcell_weight': 0.05,
        'non_human_aa_embeddings': False,

        # General model params
        'CNN': False,
        'model': ['PeptideTransformer'],
        'random_seed': [1, 2, 3],
        'l2_reg': [0],
        'dropout': [0, 0.1, 0.3],  # <-
        'learning_rate': [0.01],  # <-
        'lr_step_size': 2,  # for learning rate decay
        'lr_gamma': [0.99],
        'eval_interval': 10,
        'epochs': 100,
        'early_stopping': True,
        'early_stopping_moving_average': True,
        'embedding_dim': [32],  # <-
        'batch_size': 100,
        'batch_size_human': [100],


        # Bag-Of-AA
        'normalize_features': True,
        'hidden_dim': [20],

        # Protein LM embeddings
        'protein_embeddings': [False],
        'protein_l1_reg': [0.0005],
        'protein_l2_reg': [0],
        'protein_dropout': [0.5],
        'protein_aa_embeddings': [False],
        'protein_model': ['tape'],

        # Pretraining
        'pretrain_model': [False],
        'pretraining_epochs': [150],
        'reuse_pretrained_model': [True],

        # Old params
        'data_weighting': False,


        # Transformer
        'layer_norm_transformer': [False],
        'half_feedforward_dim': [False],
        'mhc_presentation_feature': False,
        'pep_source_layer': [False],
        'debiasing_layer': [False],
        'attention_allele_input': False,
        'attention_mhc_class_input': False,
        'attention_pep_source_input': False,
        'attention_mhc_source': [False],
        'apply_transformer': [True],
        'output_transform': ['2-layer mlp'],  # <-
        'num_attention_layers': [1],
        'attention_heads': [8],  # <-

        # Within subgroup loss
        'default_padding': [1],
        'ce_objective_weight': [1],
        'allele_objective_weight': [0],
        'source_objective_weight': [0],

        # Debiasing
        'debiasing': False,
        'shortcut_learning_rate': [0.05],
        'shortcut_epochs': 20,
        'adversary_l2_reg': [0],
        'evaluate_adv': False,
        'adv_input': ['tcell_last_layer_pred_simple'],  #, 'tcell_last_layer_pred_label', 'tcell_last_layer_label', 'tcell_last_layer'],
        'debiasing_technique': ['adv'],
        'adversary_learning_rate': [0.01],
        'tcell_adversary_learning_rate': [0.01],
        'allele_adversary_weight': [0],
        'pep_source_adversary_weight': [0],
        'allele_tcell_adv_weight': [0],
        'source_tcell_adv_weight': [0],
        'increasing_adv_weights': [False],
        'project_gradients': [False],
        'adv_tcell_label_input': [False],
        'shortcut_skip_connection': [False],
        'adversary_hidden_dim': [32],
        'tcell_adversary_hidden_dim': [10],
        'shortcut_hidden_dim': [10],
    },
    # paper: transformer on individual peptide sources
    'transformer_peptide_sources_paper_ma_es': {
        'description': 'Batch size 100, lr decay every second epoch, Training peptide sources without source representations',

        # CNN
        'out_channels': [1],
        'max_pooling': [False],
        'fc_per_position': [False],
        'max_kernel_width': [1],
        'batch_norm': [True],
        'ff_dim': [5],

        # Data
        'train_eval_frac': 1,
        'dataset_folder': ['nested_cv_thesis'],
        'mhc_class': ['I+II'],
        'early_stopping_criterion': 'all',
        'pep_source_train_selection': ['all'],
        'one_allele_per_pep': False,
        'select_two_pep_sources': False,
        'select_pep_sources': peptide_sources + ['All'],
        'evaluate_pep_sources': True,
        'select_alleles': ['All'],
        'eval_human': False,
        'evaluate_alleles': False,
        'MHC_class_eval': True,
        'sample_size': 'none',
        'features': 'OneHotPeptide',
        'feature_selector': 'peptide',

        'embedding_norm': ['None'],
        'mhc_class_input': [False],
        'pep_source_input': [False],
        'allele_input': [False],
        'separate_aa_embeddings': [False],
        'separate_pos_embeddings': [False],
        'allele_pos_embeddings': [False],
        'source_pos_embeddings': [False],
        'mhc_class_pos_embeddings': [False],
        'positional_encoding': ['both'],

        'combine_sources': False,
        'human_pep_weight': [0],

        'human_domain_adapt': False,
        'epochs_adapt_human': [100],
        'human_source_classifier': ['2-layer mlp'],
        'human_source_hidden_dim': [10],
        'human_source_learning_rate': [0.01],
        'human_source_l2_reg': 0,
        'human_pos_weight': [1],
        'non_human_aa_embeddings': False,

        'adversary_input': ['final_activations'],
        'train_mhc_class': ['mhc_class'],

        'human_adv_domain_adapt': False,
        'source_params_only_debiasing': [True],
        'source_adaptation_weight': [0],
        'source_adaptation_tcell_weight': 0.05,


        # General model params
        'CNN': False,
        'model': ['PeptideTransformer'],
        'random_seed': [1, 2, 3],
        'l2_reg': [0],
        'dropout': [0, 0.1, 0.3],  # <-
        'learning_rate': [0.01],  # <-
        'lr_step_size': 2,  # for learning rate decay
        'lr_gamma': [0.99],
        'early_stopping': True,
        'eval_interval': 10,
        'epochs': 100,
        'early_stopping_moving_average': True,
        'embedding_dim': [32],  # <-
        'batch_size': 100,
        'batch_size_human': [100],

        # Bag-Of-AA
        'normalize_features': True,
        'hidden_dim': [20],

        # Protein LM embeddings
        'protein_embeddings': [False],
        'protein_l1_reg': [0.0005],
        'protein_l2_reg': [0],
        'protein_dropout': [0.5],
        'protein_aa_embeddings': [False],
        'protein_model': ['tape'],

        # Pretraining
        'pretrain_model': [False],
        'pretraining_epochs': [150],
        'reuse_pretrained_model': [True],

        # Old params
        'data_weighting': False,

        # Transformer
        'layer_norm_transformer': [False],
        'half_feedforward_dim': [False],
        'mhc_presentation_feature': False,
        'pep_source_layer': [False],
        'pep_source_pos_input': [True],
        'debiasing_layer': [False],
        'attention_allele_input': False,
        'attention_mhc_class_input': False,
        'attention_pep_source_input': False,
        'attention_mhc_source': [False],
        'apply_transformer': [True],
        'output_transform': ['2-layer mlp'],  # <-
        'num_attention_layers': [1],
        'attention_heads': [8],  # <-
        'concat_positional_encoding': [True],

        # Within subgroup loss
        'default_padding': [1],
        'ce_objective_weight': [1],
        'allele_objective_weight': [0],
        'source_objective_weight': [0],

        # Debiasing
        'debiasing': False,
        'shortcut_learning_rate': [0.05],
        'shortcut_epochs': 20,
        'adversary_l2_reg': [0],
        'evaluate_adv': False,
        'adv_input': ['tcell_last_layer_pred_simple'],  #, 'tcell_last_layer_pred_label', 'tcell_last_layer_label', 'tcell_last_layer'],
        'debiasing_technique': ['adv'],
        'adversary_learning_rate': [0.01],
        'tcell_adversary_learning_rate': [0.01],
        'allele_adversary_weight': [0],
        'pep_source_adversary_weight': [0],
        'allele_tcell_adv_weight': [0],
        'source_tcell_adv_weight': [0],
        'increasing_adv_weights': [False],
        'project_gradients': [False],
        'adv_tcell_label_input': [False],
        'shortcut_skip_connection': [False],
        'adversary_hidden_dim': [32],
        'tcell_adversary_hidden_dim': [10],
        'shortcut_hidden_dim': [10],
    },


    # S5 (middle)
    # transformer trained on one peptide source
    'transformer_one_source_paper_valid_re_I': {
        'description': 'transformer_one_source_paper_I',

        # Data
        'train_eval_frac': 1,
        'dataset_folder': ['nested_cv_thesis'],
        'mhc_class': ['I'],
        'early_stopping_criterion': 'All',
        'pep_source_train_selection': ['all'],
        'one_allele_per_pep': False,
        'select_two_pep_sources': False,
        'select_pep_sources': ['Human immunodeficiency virus 1', 'Trypanosoma cruzi', 'Homo sapiens', 'unknown', 'Human betaherpesvirus 5', 'Hepacivirus C',
                               'Severe acute respiratory syndrome coronavirus 2'],
        'evaluate_pep_sources': False,
        'select_alleles': ['All'],
        'eval_human': True,
        'evaluate_alleles': False,
        'MHC_class_eval': True,
        'sample_size': 'none',
        'features': 'OneHotPeptide',
        'feature_selector': 'peptide',

        'embedding_norm': ['none'],
        'mhc_class_input': [False],
        'pep_source_input': [False],
        'allele_input': [False],
        'separate_aa_embeddings': [False],
        'separate_pos_embeddings': [False],
        'allele_pos_embeddings': [False],
        'source_pos_embeddings': [False],
        'mhc_class_pos_embeddings': [False],
        'positional_encoding': ['both'],

        'combine_sources': False,
        'human_pep_weight': [0],

        'human_domain_adapt': 'False',
        'epochs_adapt_human': [0],
        'human_source_classifier': ['2-layer mlp'],
        'human_source_hidden_dim': [10],
        'human_source_learning_rate': [0.01],
        'human_source_l2_reg': 0,
        'human_pos_weight': [1],
        'non_human_aa_embeddings': False,

        'adversary_input': ['final_activations'],
        'train_mhc_class': ['mhc_class'],

        'human_adv_domain_adapt': False,
        'source_params_only_debiasing': [True],
        'source_adaptation_weight': [0],
        'source_adaptation_tcell_weight': 0.05,


        # General model params
        'CNN': False,
        'model': ['PeptideTransformer'],
        'random_seed': [1, 2, 3],
        'l2_reg': [0],
        'dropout': [0.1, 0.2, 0.3, 0.4],
        'learning_rate': [0.01],
        'lr_step_size': [1],  # for learning rate decay
        'lr_gamma': [1.],
        'early_stopping': True,
        'early_stopping_moving_average': False,
        'eval_interval': 10,
        'epochs': 250,
        'embedding_dim': [32, 48],  # <-
        'batch_size': 500,
        'batch_size_human': [100],

        # Bag-Of-AA
        'normalize_features': True,
        'hidden_dim': [20],

        # Protein LM embeddings
        'protein_embeddings': [False],
        'protein_l1_reg': [0.0005],
        'protein_l2_reg': [0],
        'protein_dropout': [0.5],
        'protein_aa_embeddings': [False],
        'protein_model': ['tape'],

        # Pretraining
        'pretrain_model': [False],
        'pretraining_epochs': [100],
        'reuse_pretrained_model': [True],

        # Old params
        'data_weighting': False,

        # Transformer
        'layer_norm_transformer': [False],
        'half_feedforward_dim': [False],
        'mhc_presentation_feature': False,
        'pep_source_layer': [False],
        'pep_source_pos_input': [True],
        'debiasing_layer': [False],
        'attention_allele_input': False,
        'attention_mhc_class_input': False,
        'attention_pep_source_input': False,
        'attention_mhc_source': [False],
        'apply_transformer': [True],
        'output_transform': ['2-layer mlp'],
        'num_attention_layers': [1],
        'attention_heads': [16],
        'concat_positional_encoding': [True],

        # Within subgroup loss
        'default_padding': [1],
        'ce_objective_weight': [1],
        'allele_objective_weight': [0],
        'source_objective_weight': [0],

        # Debiasing
        'debiasing': False,
        'shortcut_learning_rate': [0.05],
        'shortcut_epochs': 20,
        'adversary_l2_reg': [0],
        'evaluate_adv': False,
        'adv_input': ['tcell_last_layer_pred_simple'],  #, 'tcell_last_layer_pred_label', 'tcell_last_layer_label', 'tcell_last_layer'],
        'debiasing_technique': ['adv'],
        'adversary_learning_rate': [0.01],
        'tcell_adversary_learning_rate': [0.01],
        'allele_adversary_weight': [0],
        'pep_source_adversary_weight': [0],
        'allele_tcell_adv_weight': [0],
        'source_tcell_adv_weight': [0],
        'increasing_adv_weights': [False],
        'project_gradients': [False],
        'adv_tcell_label_input': [False],
        'shortcut_skip_connection': [False],
        'adversary_hidden_dim': [32],
        'tcell_adversary_hidden_dim': [10],
        'shortcut_hidden_dim': [10],
    },
    # transformer trained on one peptide source
    'transformer_one_source_paper_valid_re_II': { # h
        'description': 'transformer_one_source_paper_II',

        # Data
        'train_eval_frac': 1,
        'dataset_folder': ['nested_cv_thesis'],
        'mhc_class': ['II'],
        'early_stopping_criterion': 'All',
        'pep_source_train_selection': ['all'],
        'one_allele_per_pep': False,
        'select_two_pep_sources': False,
        'select_pep_sources': ['Vaccinia virus', 'unknown', 'Human betaherpesvirus 6B', 'Dengue virus',
                               'Severe acute respiratory syndrome coronavirus 2', 'Phleum pratense',
                               'Human betaherpesvirus 5', 'Homo sapiens'],
        'evaluate_pep_sources': False,
        'select_alleles': ['All'],
        'eval_human': True,
        'evaluate_alleles': False,
        'MHC_class_eval': True,
        'sample_size': 'none',
        'features': 'OneHotPeptide',
        'feature_selector': 'peptide',

        'embedding_norm': ['none'],
        'mhc_class_input': [False],
        'pep_source_input': [False],
        'allele_input': [False],
        'separate_aa_embeddings': [False],
        'separate_pos_embeddings': [False],
        'allele_pos_embeddings': [False],
        'source_pos_embeddings': [False],
        'mhc_class_pos_embeddings': [False],
        'positional_encoding': ['both'],

        'combine_sources': False,
        'human_pep_weight': [0],

        'human_domain_adapt': 'False',
        'epochs_adapt_human': [0],
        'human_source_classifier': ['2-layer mlp'],
        'human_source_hidden_dim': [10],
        'human_source_learning_rate': [0.01],
        'human_source_l2_reg': 0,
        'human_pos_weight': [1],
        'non_human_aa_embeddings': False,

        'adversary_input': ['final_activations'],
        'train_mhc_class': ['mhc_class'],

        'human_adv_domain_adapt': False,
        'source_params_only_debiasing': [True],
        'source_adaptation_weight': [0],
        'source_adaptation_tcell_weight': 0.05,


        # General model params
        'CNN': False,
        'model': ['PeptideTransformer'],
        'random_seed': [1, 2, 3],
        'l2_reg': [0],
        'dropout': [0.1, 0.2, 0.3, 0.4],
        'learning_rate': [0.01],
        'lr_step_size': [1],  # for learning rate decay
        'lr_gamma': [1.],
        'early_stopping': True,
        'early_stopping_moving_average': False,
        'eval_interval': 10,
        'epochs': 250,
        'embedding_dim': [32, 48],  # <-
        'batch_size': 500,
        'batch_size_human': [100],

        # Bag-Of-AA
        'normalize_features': True,
        'hidden_dim': [20],

        # Protein LM embeddings
        'protein_embeddings': [False],
        'protein_l1_reg': [0.0005],
        'protein_l2_reg': [0],
        'protein_dropout': [0.5],
        'protein_aa_embeddings': [False],
        'protein_model': ['tape'],

        # Pretraining
        'pretrain_model': [False],
        'pretraining_epochs': [100],
        'reuse_pretrained_model': [True],

        # Old params
        'data_weighting': False,

        # Transformer
        'layer_norm_transformer': [False],
        'half_feedforward_dim': [False],
        'mhc_presentation_feature': False,
        'pep_source_layer': [False],
        'pep_source_pos_input': [True],
        'debiasing_layer': [False],
        'attention_allele_input': False,
        'attention_mhc_class_input': False,
        'attention_pep_source_input': False,
        'attention_mhc_source': [False],
        'apply_transformer': [True],
        'output_transform': ['2-layer mlp'],
        'num_attention_layers': [1],
        'attention_heads': [16],
        'concat_positional_encoding': [True],

        # Within subgroup loss
        'default_padding': [1],
        'ce_objective_weight': [1],
        'allele_objective_weight': [0],
        'source_objective_weight': [0],

        # Debiasing
        'debiasing': False,
        'shortcut_learning_rate': [0.05],
        'shortcut_epochs': 20,
        'adversary_l2_reg': [0],
        'evaluate_adv': False,
        'adv_input': ['tcell_last_layer_pred_simple'],  #, 'tcell_last_layer_pred_label', 'tcell_last_layer_label', 'tcell_last_layer'],
        'debiasing_technique': ['adv'],
        'adversary_learning_rate': [0.01],
        'tcell_adversary_learning_rate': [0.01],
        'allele_adversary_weight': [0],
        'pep_source_adversary_weight': [0],
        'allele_tcell_adv_weight': [0],
        'source_tcell_adv_weight': [0],
        'increasing_adv_weights': [False],
        'project_gradients': [False],
        'adv_tcell_label_input': [False],
        'shortcut_skip_connection': [False],
        'adversary_hidden_dim': [32],
        'tcell_adversary_hidden_dim': [10],
        'shortcut_hidden_dim': [10],
    },

    # S6
    # permutation - transformer trained on one peptide source with pretraining on all sources
    'transformer_pretrained_one_source_paper_perm_I': { # h
        'description': 'transformer_pretrained_one_source_paper_perm_I',

        # Data
        'train_eval_frac': 1,
        'dataset_folder': 'nested_cv_thesis',
        'permute_labels': 'full_init',
        'mhc_class': 'I',
        'early_stopping_criterion': 'All',
        'pep_source_train_selection': 'Homo sapiens',
        'one_allele_per_pep': False,
        'select_two_pep_sources': False,
        'select_pep_sources': 'All',
        'evaluate_pep_sources': False,
        'select_alleles': 'All',
        'eval_human': True,
        'evaluate_alleles': False,
        'MHC_class_eval': True,
        'sample_size': 'none',
        'features': 'OneHotPeptide',
        'feature_selector': 'peptide',

        'embedding_norm': 'none',
        'mhc_class_input': False,
        'pep_source_input': False,
        'allele_input': False,
        'separate_aa_embeddings': False,
        'separate_pos_embeddings': False,
        'allele_pos_embeddings': False,
        'source_pos_embeddings': False,
        'mhc_class_pos_embeddings': False,
        'positional_encoding': 'both',

        'combine_sources': False,
        'human_pep_weight': 0,

        'human_domain_adapt': 'False',
        'epochs_adapt_human': 0,
        'human_source_classifier': '2-layer mlp',
        'human_source_hidden_dim': 10,
        'human_source_learning_rate': 0.01,
        'human_source_l2_reg': 0,
        'human_pos_weight': 1,
        'non_human_aa_embeddings': False,

        'adversary_input': 'final_activations',
        'train_mhc_class': 'I+II',

        'human_adv_domain_adapt': False,
        'source_params_only_debiasing': True,
        'source_adaptation_weight': 0,
        'source_adaptation_tcell_weight': 0.05,


        # General model params
        'CNN': False,
        'model': 'PeptideTransformer',
        'random_seed': 1,
        'l2_reg': 0,
        'dropout': 0.2,
        'learning_rate': 0.01,
        'lr_step_size': 1,  # for learning rate decay
        'lr_gamma': 1.,
        'early_stopping': True,
        'early_stopping_moving_average': False,
        'eval_interval': 20,
        'epochs': 250,
        'embedding_dim': 32,  # <-
        'batch_size': 500,
        'batch_size_human': 100,

        # Bag-Of-AA
        'normalize_features': True,
        'hidden_dim': 20,

        # Protein LM embeddings
        'protein_embeddings': False,
        'protein_l1_reg': 0.0005,
        'protein_l2_reg': 0,
        'protein_dropout': 0.5,
        'protein_aa_embeddings': False,
        'protein_model': 'tape',

        # Pretraining
        'pretrain_model': True,
        'pretraining_epochs': 100,
        'reuse_pretrained_model': True,

        # Old params
        'data_weighting': False,

        # Transformer
        'layer_norm_transformer': False,
        'half_feedforward_dim': False,
        'mhc_presentation_feature': False,
        'pep_source_layer': False,
        'pep_source_pos_input': True,
        'debiasing_layer': False,
        'attention_allele_input': False,
        'attention_mhc_class_input': False,
        'attention_pep_source_input': False,
        'attention_mhc_source': False,
        'apply_transformer': True,
        'output_transform': '2-layer mlp',
        'num_attention_layers': 1,
        'attention_heads': 16,
        'concat_positional_encoding': True,

        # Within subgroup loss
        'default_padding': 1,
        'ce_objective_weight': 1,
        'allele_objective_weight': 0,
        'source_objective_weight': 0,

        # Debiasing
        'debiasing': False,
        'shortcut_learning_rate': 0.05,
        'shortcut_epochs': 20,
        'adversary_l2_reg': 0,
        'evaluate_adv': False,
        'adv_input': 'tcell_last_layer_pred_simple',  #, 'tcell_last_layer_pred_label', 'tcell_last_layer_label', 'tcell_last_layer',
        'debiasing_technique': 'adv',
        'adversary_learning_rate': 0.01,
        'tcell_adversary_learning_rate': 0.01,
        'allele_adversary_weight': 0,
        'pep_source_adversary_weight': 0,
        'allele_tcell_adv_weight': 0,
        'source_tcell_adv_weight': 0,
        'increasing_adv_weights': False,
        'project_gradients': False,
        'adv_tcell_label_input': False,
        'shortcut_skip_connection': False,
        'adversary_hidden_dim': 32,
        'tcell_adversary_hidden_dim': 10,
        'shortcut_hidden_dim': 10,
    },

    # permutation - transformer trained on one peptide source with pretraining on all sources
    'transformer_pretrained_one_source_paper_perm_II': { # h
        'description': 'transformer_pretrained_one_source_paper_perm_II',

        # Data
        'train_eval_frac': 1,
        'permute_labels': 'full_init',
        'dataset_folder': 'nested_cv_thesis',
        'mhc_class': 'II',
        'early_stopping_criterion': 'All',
        'pep_source_train_selection': 'Homo sapiens',
        'one_allele_per_pep': False,
        'select_two_pep_sources': False,
        'select_pep_sources': 'All',
        'evaluate_pep_sources': False,
        'select_alleles': 'All',
        'eval_human': True,
        'evaluate_alleles': False,
        'MHC_class_eval': True,
        'sample_size': 'none',
        'features': 'OneHotPeptide',
        'feature_selector': 'peptide',

        'embedding_norm': 'none',
        'mhc_class_input': False,
        'pep_source_input': False,
        'allele_input': False,
        'separate_aa_embeddings': False,
        'separate_pos_embeddings': False,
        'allele_pos_embeddings': False,
        'source_pos_embeddings': False,
        'mhc_class_pos_embeddings': False,
        'positional_encoding': 'both',

        'combine_sources': False,
        'human_pep_weight': 0,

        'human_domain_adapt': 'False',
        'epochs_adapt_human': 0,
        'human_source_classifier': '2-layer mlp',
        'human_source_hidden_dim': 10,
        'human_source_learning_rate': 0.01,
        'human_source_l2_reg': 0,
        'human_pos_weight': 1,
        'non_human_aa_embeddings': False,

        'adversary_input': 'final_activations',
        'train_mhc_class': 'I+II',

        'human_adv_domain_adapt': False,
        'source_params_only_debiasing': True,
        'source_adaptation_weight': 0,
        'source_adaptation_tcell_weight': 0.05,


        # General model params
        'CNN': False,
        'model': 'PeptideTransformer',
        'random_seed': 1,
        'l2_reg': 0,
        'dropout': 0.2,
        'learning_rate': 0.01,
        'lr_step_size': 1,  # for learning rate decay
        'lr_gamma': 1.,
        'early_stopping': True,
        'early_stopping_moving_average': False,
        'eval_interval': 20,
        'epochs': 250,
        'embedding_dim': 32,  # <-
        'batch_size': 500,
        'batch_size_human': 100,

        # Bag-Of-AA
        'normalize_features': True,
        'hidden_dim': 20,

        # Protein LM embeddings
        'protein_embeddings': False,
        'protein_l1_reg': 0.0005,
        'protein_l2_reg': 0,
        'protein_dropout': 0.5,
        'protein_aa_embeddings': False,
        'protein_model': 'tape',

        # Pretraining
        'pretrain_model': True,
        'pretraining_epochs': 100,
        'reuse_pretrained_model': True,

        # Old params
        'data_weighting': False,

        # Transformer
        'layer_norm_transformer': False,
        'half_feedforward_dim': False,
        'mhc_presentation_feature': False,
        'pep_source_layer': False,
        'pep_source_pos_input': True,
        'debiasing_layer': False,
        'attention_allele_input': False,
        'attention_mhc_class_input': False,
        'attention_pep_source_input': False,
        'attention_mhc_source': False,
        'apply_transformer': True,
        'output_transform': '2-layer mlp',
        'num_attention_layers': 1,
        'attention_heads': 16,
        'concat_positional_encoding': True,

        # Within subgroup loss
        'default_padding': 1,
        'ce_objective_weight': 1,
        'allele_objective_weight': 0,
        'source_objective_weight': 0,

        # Debiasing
        'debiasing': False,
        'shortcut_learning_rate': 0.05,
        'shortcut_epochs': 20,
        'adversary_l2_reg': 0,
        'evaluate_adv': False,
        'adv_input': 'tcell_last_layer_pred_simple',  #, 'tcell_last_layer_pred_label', 'tcell_last_layer_label', 'tcell_last_layer',
        'debiasing_technique': 'adv',
        'adversary_learning_rate': 0.01,
        'tcell_adversary_learning_rate': 0.01,
        'allele_adversary_weight': 0,
        'pep_source_adversary_weight': 0,
        'allele_tcell_adv_weight': 0,
        'source_tcell_adv_weight': 0,
        'increasing_adv_weights': False,
        'project_gradients': False,
        'adv_tcell_label_input': False,
        'shortcut_skip_connection': False,
        'adversary_hidden_dim': 32,
        'tcell_adversary_hidden_dim': 10,
        'shortcut_hidden_dim': 10,
    },

}


def config_generator(cv, outer_fold, inner_fold):
    keys = configurations[cv.config_name].keys()
    values = [value if type(value) is list else [value] for value in configurations[cv.config_name].values()]
    config_id = cv.start_id
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        config['inner_fold'] = inner_fold
        config['outer_fold'] = outer_fold
        config['config_id'] = config_id
        file_paths = get_file_paths(cv, config_id, outer_fold, inner_fold, params=config)
        config_id += 1
        yield {**config, **file_paths}


def get_combined_config(cv, outer_fold=None, test=False, config_id=None, params=None):
    config = configurations[cv.config_name]
    file_paths = get_file_paths(cv, config_id=config_id, outer_fold=outer_fold, test=test, params=params)
    return {**config, **file_paths}


def get_param_keys(config_name):
    config = configurations[config_name]
    param_keys = list(config.keys())
    return param_keys


def get_outer_k(config_name):
    return configurations[config_name]['outer_k']


def get_inner_k(config_name):
    return configurations[config_name]['inner_k']


def get_model_file_path(cv, outer_fold, params):
    return get_file_paths(cv, outer_fold=outer_fold, params=params)['model_file']


def get_file_paths(cv, config_id=None, outer_fold=None, inner_fold=None, test=False, params=None):
    cv_data_dir = cv.data_dir + '/' + params['dataset_folder'] if params is not None and 'dataset_folder' in params else ''
    features_dir = cv.data_dir + '/features'
    model_dir = '../saved_models/'
    vis_dir = '../visualization_data/'
    test_results_dir = '../test_results/'
    valid_results_dir = '../valid_results_cv/' + cv.experiment_name
    valid_predictions_dir = '../valid_results/' + cv.experiment_name + f'/config_{config_id}'
    tensorboard_dir = '../tensorboard'
    embedding_dim = params["embedding_dim"] if params is not None and 'embedding_dim' in params else 0
    protein_embedding_name = params["protein_model"] if params is not None and 'protein_model' in params else ''
    random_seed = params['random_seed'] if params is not None and 'random_seed' in params else None
    return {
        'tensorboard_dir':
            file_path(tensorboard_dir),
        'tensorboard_experiment_dir':
            file_path(tensorboard_dir, cv.test_experiment_name, f'config_{config_id}/cv_{outer_fold}_seed_{random_seed}_test') if test
            else file_path(tensorboard_dir, cv.experiment_name, f'config_{config_id}/cv_{outer_fold}_{inner_fold}'),
        'valid_results_dir':
            file_path(valid_results_dir),
        'valid_predictions_dir':
            file_path(valid_predictions_dir),
        'test_results_dir':
            file_path(test_results_dir, cv.test_experiment_name),
        'valid_results_file':
            file_path(valid_results_dir, f'results_{config_id}_outer_{outer_fold}_inner_{inner_fold}.pkl'),
        'valid_results_text_file':
            file_path(valid_predictions_dir, f'results_outer_{outer_fold}_inner_{inner_fold}.csv'),
        'valid_prediction_file':
            file_path(valid_predictions_dir, f'predictions_outer_{outer_fold}_inner_{inner_fold}.csv'),
        'test_results_file':
            file_path(test_results_dir, cv.test_experiment_name, f'results_outer_{outer_fold}_{random_seed}.pkl'),
        'test_results_text_file':
            file_path(test_results_dir, cv.test_experiment_name, f'results_outer_{outer_fold}_{random_seed}.csv'),
        'test_allele_results_file':
            file_path(test_results_dir, cv.test_experiment_name, f'allele_results_outer_{outer_fold}.pkl'),
        'test_prediction_file':
            file_path(test_results_dir, cv.test_experiment_name, f'predictions_outer_{outer_fold}_{random_seed}.csv'),
        'combined_test_results_file':
            file_path(test_results_dir, f'{cv.test_experiment_name}_results.csv'),
        'pseudoseq_file':
            file_path(features_dir, 'pseudosequences.txt'),
        'blosum_file':
            file_path(features_dir, 'blosum62.txt'),
        'protein_embedding_file':
            file_path(features_dir, f'peptides_{protein_embedding_name}.npz'),
        'eval_file':
            file_path(cv_data_dir, f'test_outer_{outer_fold}.csv') if test
            else file_path(cv_data_dir, f'valid_outer_{outer_fold}_inner_{inner_fold}.csv'),
        'train_file':
            file_path(cv_data_dir, f'train_outer_{outer_fold}.csv') if test
            else file_path(cv_data_dir, f'train_outer_{outer_fold}_inner_{inner_fold}.csv'),
        'saved_models_dir':
            file_path(model_dir, cv.experiment_name),
        'visualization_dir':
            file_path(vis_dir, cv.experiment_name),
        'visualization_data_file':
            file_path(vis_dir, cv.experiment_name, f'data_{config_id}'),
        'visualization_prediction_file':
            file_path(vis_dir, cv.experiment_name, f'prediction_scores_{config_id}.pkl'),
        'visualization_meta_file':
            file_path(vis_dir, cv.experiment_name, f'meta_{config_id}.tsv'),
        'model_file':
            file_path(model_dir, f'{cv.test_experiment_name}_{outer_fold}_{random_seed}.tar'),
    }


def file_path(*args):
    """
    Provides file paths relative to the location of this file
    instead of the execution directory.
    :param args: file path, relative to this files directory
    :return: absolute path to the specified file
    """
    current_dir = os.path.dirname(__file__)
    return os.path.normpath(os.path.join(current_dir, *args))
