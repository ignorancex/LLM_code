import argparse
import os, yaml

def load_configs_nlp(args):
    if args.model_config is None:
        args.model_config = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "configs/configs.yaml")

    # yamlfile_name = os.path.join(args.model_config, "model_config.yaml")
    # elif args.model_type == "csdi":
    #     yamlfile_name = os.path.join(model_config_file_path_base, "csdi_config.yaml")
    root_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(root_dir, args.model_config), "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        rl_config = config["rl"][args.rl_algorithm]
        model_config = config["model"]
    return rl_config, model_config


def parse_args():
    parser = argparse.ArgumentParser(description="Running GRUODE on Double OU")
    # parser.add_argument('--model_name', type=str, help="Model to use", default="double_OU_gru_ode_bayes")
    parser.add_argument('--log_folder', type=str, help="Dataset CSV file", default=None)
    parser.add_argument('--explanation_type', type=str, help="posthoc explanation type", default="decision_tree", choices=["shap", "anchor", "decision_tree", "lime", "ours", "nam", "lore"])
    parser.add_argument('--dataset_name', type=str, help="dataset name", default="aicc", choices=["six","synthetic","synthetic2", "aicc", "ihdp", "tcga", "ihdp_cont", "news_cont", "sw", "tcga_str"])
    parser.add_argument('--dataset_id', type=int, help="dataset name", default=None)
    parser.add_argument('--data_folder', type=str, default="/data6/wuyinjun/causal_tabular/", help="std of the initial phi table")
    # parser.add_argument('--jitter', type=float, help="Time jitter to add (to split joint observations)", default=0)
    parser.add_argument('--lr', type=float, help="learning rate", default=2e-3)
    # parser.add_argument('--ratio', type=float, help="learning rate", default=0.5)
    parser.add_argument('--seed', type=int, help="Seed for data split generation", default=None)
    # parser.add_argument('--max_val_samples', type=int, help="Seed for data split generation", default=1)
    parser.add_argument('--batch_size', type=int, help="Seed for data split generation", default=2048)
    parser.add_argument('--epochs', type=int, help="Seed for data split generation", default=200)
    parser.add_argument('--topk_act', type=int, help="Seed for data split generation", default=1)
    parser.add_argument('--program_max_len', type=int, help="The length of the program", default=4)
    # parser.add_argument('--full_gru_ode', action="store_true", default=True)
    parser.add_argument('--is_log', action="store_true", default=False)
    parser.add_argument('--method_two', action="store_true", default=False)
    parser.add_argument('--method_three', action="store_true", default=False)
    # parser.add_argument('--featurization', type=str, choices=[one, two], default=one)
    # parser.add_argument('--no_impute',action="store_true",default = True)
    # parser.add_argument('--weighted_reward',action="store_true",default = True)
    # parser.add_argument('--demo', action = "store_true", default = False)
    parser.add_argument('--has_dose', action = "store_true", default = False)
    parser.add_argument('--tr', action = "store_true", default = False, help="Target Regularization")
    parser.add_argument('--tr_str_two', action = "store_true", default = False, help="Target Regularization")
    # parser.add_argument('--treatment_var_ids', nargs='+', type=int, help='List of integers', default=[0])
    parser.add_argument('--dropout_p', type=float, default=0, help="std of the initial phi table")
    parser.add_argument('--model_config', type=str, default=None, help="std of the initial phi table")
    parser.add_argument('--dataset_config', type=str, default="configs/dataset_configs.yaml", help="std of the initial phi table")
    parser.add_argument('--rl_algorithm', type=str, default="dqn", choices=["dqn", "ppo"], help="std of the initial phi table")
    
    parser.add_argument('--model', type=str, default="mlp", help="std of the initial phi table")
    parser.add_argument('--method', type=str, default="ours", help="std of the initial phi table")
    parser.add_argument('--reg_method', type=str, default=None, help="std of the initial phi table")
    parser.add_argument('--num_treatments', type=int, help="Seed for data split generation", default=2)
    # parser.add_argument('--missing_ratio', type=float, default=None, help="Seed for data split generation")
    parser.add_argument('--p', type=int, help="The length of the program", default=0)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument('--no_tr', action = "store_true", default = False)
    parser.add_argument('--no_hyper_adj', action = "store_true", default = False)
    parser.add_argument("--subset_num", default=None, type=int)
    parser.add_argument('--cached_model_name', type=str, default=None, help="std of the initial phi table")
    parser.add_argument("--cohort_model_path", type=str, default=None, help="std of the initial phi table")
    # cached_backbone
    parser.add_argument("--cached_backbone", type=str, default=None, help="std of the initial phi table")
    # backbone
    # fix_backbone
    parser.add_argument('--fix_backbone', action = "store_true", default = False)
    # parser.add_argument('--gpu_db', action = "store_true", default = False)
    # eval
    parser.add_argument('--eval', action = "store_true", default = False)
    parser.add_argument("--backbone", type=str, default="TransTEE", help="std of the initial phi table")
    parser.add_argument('--regression_ratio', type=float, default=0, help="std of the initial phi table")
    # parser.add_argument('--cached_model_path', type=str, default=None, help="Model to use")
    # parser.add_argument('--cached_model_suffix', type=int, default=None, help="Model to use")

    # parser.add_argument('--model', type=str, default="mlp", help="std of the initial phi table")
    # do_medical
    args = parser.parse_args()
    
    return args, parser