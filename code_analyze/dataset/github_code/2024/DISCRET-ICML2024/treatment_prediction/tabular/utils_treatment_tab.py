import pandas as pd

import os,sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


from parse_args_tabular import parse_args
from tabular_data_utils.tabular_dataset import *
import synthetic_lang
from create_language import Language
from utils_treatment import load_configs, load_dataset_configs
import torch
from tab_models.tab_model import *
from data_generations.TCGA import *


def load_tab_data(args):
    if args.dataset_name == "aicc":
        all_data = load_aicc_dataset(os.path.join(args.data_folder, args.dataset_name)).dropna()
        train_df, valid_df, test_df = split_train_valid_test_df(all_data)
        id_attr = "index"
        outcome_attr = "y"
        count_outcome_attr = "count_Y"
        treatment_attr = "z"
        dose_attr=None
        args.has_dose= False
        lang = Language(train_df, id_attr, outcome_attr, count_outcome_attr, treatment_attr, dose_attr, None, precomputed=None, lang=synthetic_lang)
        normalize_y = False
        extra_info = None
        args.cont_treatment = False
        args.cat_and_cont_treatment = False
    elif args.dataset_name == "ihdp":
        id_attr = "id"
        train_df, valid_df, test_df, all_data = load_ihdp_dataset2(os.path.join(args.data_folder, args.dataset_name), args.dataset_id, args.subset_num) #, args.missing_ratio)
        outcome_attr = "y_factual"
        treatment_attr = "treatment"
        count_outcome_attr = "y_cfactual"
        # train_df, valid_df, test_df, all_data = load_ihdp_dataset()
        
        # outcome_attr = "outcome"
        
        # treatment_attr = "Treatment"
        
        # count_outcome_attr = "counter_outcome"
        
        dose_attr=None
        args.has_dose= False
        if args.subset_num is not None:
            train_df = train_df.sample(args.subset_num)

        # synthetic_lang.CAT_FEATS = ["x_" + str(k) for k in range(7, 26)]
        # data, id_attr, outcome_attr, count_outcome_attr, treatment_attr, does_attr, text_attr, precomputed, lang
        lang = Language(train_df, id_attr, outcome_attr, count_outcome_attr, treatment_attr, dose_attr, None, precomputed=None, lang=synthetic_lang)
        normalize_y = True
        extra_info = None
        args.cont_treatment = False
        args.cat_and_cont_treatment = False
    # elif args.dataset_name == "synthetic":
    #     id_attr = "id"
    #     train_df, valid_df, test_df, all_data = load_synthetic_data(os.path.join(args.data_folder, "ihdp"), args.dataset_id)
    #     outcome_attr = "y_factual"
    #     treatment_attr = "treatment"
    #     count_outcome_attr = "y_cfactual"
    #     # train_df, valid_df, test_df, all_data = load_ihdp_dataset()
        
    #     # outcome_attr = "outcome"
        
    #     # treatment_attr = "Treatment"
        
    #     # count_outcome_attr = "counter_outcome"
        
    #     dose_attr=None
    #     args.has_dose= False
    #     # synthetic_lang.CAT_FEATS = ["x_" + str(k) for k in range(7, 26)]
    #     # data, id_attr, outcome_attr, count_outcome_attr, treatment_attr, does_attr, text_attr, precomputed, lang
    #     lang = Language(train_df, id_attr, outcome_attr, count_outcome_attr, treatment_attr, dose_attr, None, precomputed=None, lang=synthetic_lang)
    #     normalize_y = True
    #     extra_info = None
    #     args.cont_treatment = False
    #     args.cat_and_cont_treatment = False
    elif args.dataset_name == "ihdp_cont":
        train_df, valid_df, test_df, all_data, t_grid = load_ihdp_cont_dataset(os.path.join(args.data_folder, args.dataset_name), args.dataset_id)    
        id_attr = "id"
        outcome_attr = "outcome"
        treatment_attr = "Treatment"
        count_outcome_attr = None
        dose_attr=None
        args.has_dose= False
        lang = Language(train_df, id_attr, outcome_attr, count_outcome_attr, treatment_attr, dose_attr, None, precomputed=None, lang=synthetic_lang)
        normalize_y = False
        extra_info = (t_grid)
        args.cont_treatment = True
        all_data = None
        args.cat_and_cont_treatment = False
    elif args.dataset_name == "news_cont":
        train_df, valid_df, test_df, all_data, t_grid = load_ihdp_cont_dataset(os.path.join(args.data_folder, args.dataset_name), args.dataset_id)    
        id_attr = "id"
        outcome_attr = "outcome"
        treatment_attr = "Treatment"
        count_outcome_attr = None
        dose_attr=None
        args.has_dose= False
        lang = Language(train_df, id_attr, outcome_attr, count_outcome_attr, treatment_attr, dose_attr, None, precomputed=None, lang=synthetic_lang)
        normalize_y = False
        extra_info = (t_grid)
        args.cont_treatment = True
        all_data = pd.concat([train_df, valid_df, test_df])
        args.cat_and_cont_treatment = False
    elif args.dataset_name == "tcga":
        dataset_params = dict()
        #         self.num_treatments = args['num_treatments']
        # self.treatment_selection_bias = args['treatment_selection_bias']
        # self.dosage_selection_bias = args['dosage_selection_bias']

        # self.validation_fraction = args['validation_fraction']
        # self.test_fraction = args['test_fraction']
        dataset_configs = load_dataset_configs(args, os.path.dirname(os.path.realpath(__file__)))
        dataset_params['num_treatments'] = args.num_treatments
        dataset_params['treatment_selection_bias'] = dataset_configs["treatment_selection_bias"]
        dataset_params['dosage_selection_bias'] = dataset_configs["dosage_selection_bias"]
        dataset_params['validation_fraction'] = dataset_configs["validation_fraction"]
        dataset_params['test_fraction'] = dataset_configs["test_fraction"]
        dataset_params['dataset_name'] = args.dataset_name
        dataset_params["data_folder"] = args.data_folder
        data_class = TCGA_Data(dataset_params)
        dataset = data_class.dataset
        train_df, valid_df, test_df = get_dataset_splits(dataset)
        
        id_attr = "id"
        outcome_attr = "y"
        treatment_attr = "t"
        dose_attr = "d"
        count_outcome_attr = None
        args.has_dose= True
        all_data = None
        lang = Language(train_df, id_attr, outcome_attr, count_outcome_attr, treatment_attr, dose_attr, None, precomputed=None, lang=synthetic_lang)
        normalize_y = False
        args.cont_treatment = False
        extra_info = dataset['metadata']['v']
        args.cat_and_cont_treatment = True
        print()
    elif args.dataset_name == "six":
        all_feat_ls = torch.load(os.path.join(args.log_folder, "all_feat_ls"))
        outcome_attr = "outcome"
        treatment_attr = "treatment"
        id_attr = "index"
        train_X = torch.load(os.path.join(args.log_folder, "flatten_train_X"))
        train_outcome = torch.load(os.path.join(args.log_folder, "flatten_train_outcome"))
        train_treatment = torch.load(os.path.join(args.log_folder, "flatten_train_treatment"))
        train_M = torch.load(os.path.join(args.log_folder, "flatten_train_M"))
        
        train_df = pd.DataFrame(torch.cat([train_X, train_outcome, train_treatment], dim=1).numpy(), columns=all_feat_ls + [outcome_attr, treatment_attr])
        train_df.reset_index(inplace=True)
        
        valid_X = torch.load(os.path.join(args.log_folder, "flatten_valid_X"))
        valid_outcome = torch.load(os.path.join(args.log_folder, "flatten_valid_outcome"))
        valid_treatment = torch.load(os.path.join(args.log_folder, "flatten_valid_treatment"))
        valid_M = torch.load(os.path.join(args.log_folder, "flatten_valid_M"))
        valid_df = pd.DataFrame(torch.cat([valid_X, valid_outcome, valid_treatment], dim=1).numpy(), columns=all_feat_ls + [outcome_attr, treatment_attr])
        valid_df.reset_index(inplace=True)
        
        if not args.is_log:
            test_X = torch.load(os.path.join(args.log_folder, "flatten_test_X"))
            test_outcome = torch.load(os.path.join(args.log_folder, "flatten_test_outcome"))
            test_treatment = torch.load(os.path.join(args.log_folder, "flatten_test_treatment"))
            test_M = torch.load(os.path.join(args.log_folder, "flatten_test_M")) 
        else:
            test_X = torch.load(os.path.join(args.log_folder, "flatten_single_test_X"))
            test_outcome = torch.load(os.path.join(args.log_folder, "flatten_single_test_outcome"))
            test_treatment = torch.load(os.path.join(args.log_folder, "flatten_single_test_treatment"))
            test_M = torch.load(os.path.join(args.log_folder, "flatten_single_test_M")) 
        
        test_df = pd.DataFrame(torch.cat([test_X, test_outcome, test_treatment], dim=1).numpy(), columns=all_feat_ls + [outcome_attr, treatment_attr])
        test_df.reset_index(inplace=True)
        dose_attr = None
        count_outcome_attr = None
        args.has_dose= False
        lang = Language(train_df, id_attr, outcome_attr, count_outcome_attr, treatment_attr, dose_attr, None, precomputed=None, lang=synthetic_lang)
        all_data = pd.concat([train_df, valid_df])
        normalize_y = False
        extra_info = None
        args.cont_treatment = False
        args.cat_and_cont_treatment = False

    elif args.dataset_name == "synthetic":
    
        id_attr = "id"
        train_df, valid_df, test_df, all_data = load_synthetic_data0(os.path.join(args.data_folder, "synthetic"), args.dataset_id)
        outcome_attr = "y_factual"
        treatment_attr = "treatment"
        count_outcome_attr = "y_cfactual"
        # train_df, valid_df, test_df, all_data = load_ihdp_dataset()
        
        # outcome_attr = "outcome"
        
        # treatment_attr = "Treatment"
        
        # count_outcome_attr = "counter_outcome"
        
        dose_attr=None
        args.has_dose= False
        synthetic_lang.CAT_FEATS = [col for col in train_df.columns if col.startswith("x_")]
        # data, id_attr, outcome_attr, count_outcome_attr, treatment_attr, does_attr, text_attr, precomputed, lang
        lang = Language(train_df, id_attr, outcome_attr, count_outcome_attr, treatment_attr, dose_attr, None, precomputed=None, lang=synthetic_lang)
        normalize_y = True
        extra_info = None
        args.cont_treatment = False
        args.cat_and_cont_treatment = False
        
    elif args.dataset_name == "synthetic2":
    
        id_attr = "id"
        train_df, valid_df, test_df, all_data = load_synthetic_data0_2(os.path.join(args.data_folder, "synthetic2"), args.dataset_id)
        outcome_attr = "y_factual"
        treatment_attr = "treatment"
        count_outcome_attr = "y_cfactual"
        # train_df, valid_df, test_df, all_data = load_ihdp_dataset()
        
        # outcome_attr = "outcome"
        
        # treatment_attr = "Treatment"
        
        # count_outcome_attr = "counter_outcome"
        
        dose_attr=None
        args.has_dose= False
        synthetic_lang.CAT_FEATS = [col for col in train_df.columns if col.startswith("x_")]
        # data, id_attr, outcome_attr, count_outcome_attr, treatment_attr, does_attr, text_attr, precomputed, lang
        lang = Language(train_df, id_attr, outcome_attr, count_outcome_attr, treatment_attr, dose_attr, None, precomputed=None, lang=synthetic_lang)
        normalize_y = True
        extra_info = None
        args.cont_treatment = False
        args.cat_and_cont_treatment = False
        
    return all_data, train_df, valid_df, test_df, lang, id_attr, outcome_attr, treatment_attr, count_outcome_attr, dose_attr, normalize_y, extra_info