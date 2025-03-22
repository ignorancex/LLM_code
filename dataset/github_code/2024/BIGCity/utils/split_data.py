
import sys
sys.path.append("..")

from config import args_config
import os
import pandas as pd

if __name__ == "__main__":
    args = args_config.args
    traj_file_name = os.path.join(args.root_path, args.city, f'traj_{args.city}_11.csv')
    traj_data = pd.read_csv(traj_file_name, delimiter=';')

    if args.city == 'bj':
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1
        
        total_num = len(traj_data)

        train_num = int(train_ratio * total_num)
        val_num = int(val_ratio * total_num)
        test_num = int(test_ratio * total_num)
    
    elif args.city == 'xa' or 'cd':
        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2
        
        total_num = len(traj_data)

        train_num = int(train_ratio * total_num)
        val_num = int(val_ratio * total_num)
        test_num = int(test_ratio * total_num)

    traj_train = traj_data.head(train_num)
    traj_train_file_name = os.path.join(args.root_path, args.city, f'traj_{args.city}_train.csv')
    traj_train.to_csv(traj_train_file_name, index=False, sep=';')

    traj_val = traj_data.iloc[train_num: train_num + val_num]
    traj_val_file_name = os.path.join(args.root_path, args.city, f'traj_{args.city}_validation.csv')
    traj_val.to_csv(traj_val_file_name, index=False, sep=';')

    traj_test = traj_data.iloc[train_num + val_num: train_num + val_num + test_num]
    traj_test_file_name = os.path.join(args.root_path, args.city, f'traj_{args.city}_test.csv')
    traj_test.to_csv(traj_test_file_name, index=False, sep=';')