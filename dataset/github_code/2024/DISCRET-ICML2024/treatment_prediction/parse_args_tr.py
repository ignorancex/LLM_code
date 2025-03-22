import argparse
import os, yaml

def load_configs_tr(args):
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
    parser.add_argument('--model_name', type=str, help="Model to use", default="double_OU_gru_ode_bayes")
    parser.add_argument('--log_folder', type=str, help="Dataset CSV file", default=None)
    parser.add_argument('--dataset_name', type=str, help="dataset name", default="five")
    parser.add_argument('--jitter', type=float, help="Time jitter to add (to split joint observations)", default=0)
    parser.add_argument('--lr', type=float, help="learning rate", default=2e-3)
    parser.add_argument('--ratio', type=float, help="learning rate", default=0.1)
    parser.add_argument('--seed', type=int, help="Seed for data split generation", default=432)
    parser.add_argument('--max_val_samples', type=int, help="Seed for data split generation", default=1)
    parser.add_argument('--batch_size', type=int, help="Seed for data split generation", default=32)
    parser.add_argument('--epochs', type=int, help="Seed for data split generation", default=200)
    parser.add_argument('--topk_act', type=int, help="Seed for data split generation", default=1)
    parser.add_argument('--program_max_len', type=int, help="The length of the program", default=4)
    parser.add_argument('--full_gru_ode', action="store_true", default=True)
    parser.add_argument('--is_log', action="store_true", default=False)
    parser.add_argument('--solver', type=str, choices=["euler", "midpoint","dopri5"], default="euler")
    parser.add_argument('--no_impute',action="store_true",default = True)
    parser.add_argument('--weighted_reward',action="store_true",default = True)
    parser.add_argument('--demo', action = "store_true", default = False)
    parser.add_argument('--method_one', action = "store_true", default = False)
    parser.add_argument('--test', action = "store_true", default = False)
    parser.add_argument('--treatment_var_ids', nargs='+', type=int, help='List of integers', default=None)
    parser.add_argument('--dropout_p', type=float, default=0, help="std of the initial phi table")
    parser.add_argument('--model_config', type=str, default=None, help="std of the initial phi table")
    parser.add_argument('--rl_algorithm', type=str, default="dqn", choices=["dqn", "ppo"], help="std of the initial phi table")
    parser.add_argument('--do_medical', action = "store_true", default = False)
    parser.add_argument('--model', type=str, default="mlp", help="std of the initial phi table")
    parser.add_argument('--data_folder', type=str, default="/data6/wuyinjun/cancer_data/", help="std of the initial phi table")
    
    parser.add_argument('--cached_model_path', type=str, default=None, help="Model to use")
    parser.add_argument('--remove_std_attr', action = "store_true", default = False)
    parser.add_argument('--reward_specifity', action = "store_true", default = False)
    parser.add_argument('--cached_model_suffix', type=str, default=None, help="Model to use")
    parser.add_argument('--use_feat_bound_point_ls', action='store_true', help="std of the initial phi table")
    parser.add_argument('--posthoc', action = "store_true", default = False)
    parser.add_argument('--prefer_smaller_range', action='store_true', help='specifies what features to extract')
    parser.add_argument('--prefer_smaller_range_coeff', type=float, default=0.5, help='specifies what features to extract')

    # parser.add_argument('--model', type=str, default="mlp", help="std of the initial phi table")
    # do_medical
    args = parser.parse_args()
    return args