import pandas as pd

import os,sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

# from miracle_local.impute import impute_train_eval_all
from parse_args_tabular import parse_args
from tabular_data_utils.tabular_dataset import *
import synthetic_lang
from create_language import Language
from utils_treatment import load_configs, transform_treatment_ids, set_lang_data
import torch
from tab_models.tab_model import *
from utils_treatment_tab import load_tab_data

if __name__ == "__main__":
    args, parser = parse_args()
    args = parser.parse_args()
    if args.seed is None and args.dataset_id is not None:
        args.seed = args.dataset_id
    if args.seed is not None:
        print("configure random seed as {}".format(args.seed))
        random_seed = args.seed
        random.seed(random_seed)

        # Set a random seed for NumPy
        np.random.seed(random_seed)

        # Set a random seed for PyTorch
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    all_data, train_df, valid_df, test_df, lang, id_attr, outcome_attr, treatment_attr, count_outcome_attr, dose_attr, normalize_y, extra_info = load_tab_data(args)
    train_dataset, valid_dataset, test_dataset, feat_range_mappings = create_dataset(args.dataset_name, all_data, train_df, valid_df, test_df, synthetic_lang, id_attr, outcome_attr, treatment_attr, synthetic_lang.DROP_FEATS, count_outcome_attr=count_outcome_attr, dose_attr=dose_attr, normalize_y=normalize_y, extra_info=extra_info)
    
    
    
    numeric_count  = len(train_dataset.num_cols)
        # numeric_count  = len(train_dataset.num_cols) if "num_feat" in lang.syntax else 0
    category_count = list(train_dataset.cat_cols) #len(lang.syntax["cat_feat"]) if "cat_feat" in lang.syntax else 0
    category_sum_count = train_dataset.cat_sum_count

    input_size = numeric_count + category_sum_count
    root_dir = os.path.dirname(os.path.realpath(__file__))
    args.root_dir = root_dir
    rl_config, model_config, backbone_model_config = load_configs(args,root_dir=root_dir)
    args.embed_size = model_config["hidden_size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lang = set_lang_data(lang, train_dataset, device=device)
    trainer = dql_algorithm(
        train_dataset, valid_dataset, test_dataset, id_attr, outcome_attr, treatment_attr, lang, args.lr, rl_config["gamma"], args.dropout_p, feat_range_mappings, args.program_max_len, rl_config["replay_memory_capacity"], rl_config, model_config, backbone_model_config, numeric_count, category_count, category_sum_count, args, topk_act=args.topk_act,
                batch_size = args.batch_size,
                modeldir = None, log_folder = args.log_folder
    )
    if args.cached_model_name is not None:
        trainer.dqn.policy_net.load_state_dict(torch.load(args.cached_model_name, map_location=trainer.device))
        trainer.dqn.target_net.load_state_dict(torch.load(args.cached_model_name, map_location=trainer.device))
    
    if args.is_log:
        # if args.dataset_name == "ihdp":
        trainer.dqn.policy_net.load_state_dict(torch.load(os.path.join(args.log_folder, "model_best"), map_location=trainer.device))
        trainer.dqn.target_net.load_state_dict(torch.load(os.path.join(args.log_folder, "model_best"), map_location=trainer.device))
    trainer.run(train_dataset, valid_dataset, test_dataset)