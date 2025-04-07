import os
from catboost import CatBoostRegressor, Pool
from tools import *

def build_reg_model(args, work_type: str):
    params = {
        'loss_function': 'RMSE', # 损失函数，取值RMSE, Logloss, MAE, CrossEntropy, Quantile, LogLinQuantile, Multiclass, MultiClassOneVsAll, MAPE, Poisson。默认Logloss。
        'eval_metric': 'MAE',  # 用于过度拟合检测和最佳模型选择的指标，取值范围同上
        'iterations': 100,  # 最大迭代次数，默认500. 别名：num_boost_round, n_estimators, num_trees
        'depth': 6,  # 树深，默认值6
        'learning_rate': 1e-2,  # 学习速率,默认0.03 别名：eta
        'use_best_model': True,
        "task_type": "GPU",
        "devices": "0",
    }
    # build_regress_model
    if work_type == "test":
        best_model_path = "ckpt_cb"
        best_model_name = args.reg_model_name
        reg_model = CatBoostRegressor(**params)
        reg_model.load_model(os.path.join(best_model_path, best_model_name))

    elif work_type == "train":
        # with GPU
        reg_model = CatBoostRegressor(**params)
    else:
        reg_model = None
    return reg_model


def val_and_save(reg_model, features_list, label_list, best_mae):
    pre_list = reg_model.predict(features_list)
    print("===Val result===")
    mae, _, _ = print_output(label_list, pre_list)
    if mae < best_mae:
        best_mae = mae
        # reg_model.save_model("ckpt_cb/best_%.4f" % mae,
        reg_model.save_model("ckpt_cb/best",
                             format="cbm",
                             export_parameters=None,
                             pool=None)
    return best_mae


def cb_reg_fuc(args, reg_model, train_data, val_data):
    # data: [n, seq_length, feature_vec]
    # label: [n, seq_length]
    train_list, train_label = train_data
    val_list, val_label = val_data
    train_list = np.array(train_list)
    train_label = np.array(train_label)
    val_list = np.array(val_list)
    val_label = np.array(val_label)

    seq_length = train_list.shape[1]
    seq_pre_list = []
    seq_val_pre_list = []
    for i in range(seq_length):
        print(val_list.shape)
        print(val_label.shape)
        print(train_list.shape)
        print(train_label.shape)
        eval_set = Pool(val_list[:, i, :], val_label[:, i])
        reg_model.fit(train_list[:, i, :], train_label[:, i],
                      eval_set=eval_set,
                      verbose=100)
        pre_list = reg_model.predict(train_list[:, i, :])
        val_pre_list = reg_model.predict(val_list[:, i, :])

        seq_pre_list.append(pre_list)  # seq_pre_list: [seq_length, n]
        seq_val_pre_list.append(val_pre_list)
    seq_pre_list = np.array(seq_pre_list).T  # seq_pre_list: [n, seq_length]
    seq_val_pre_list = np.array(seq_val_pre_list).T
    print(seq_pre_list.shape)
    print("===Train result===")
    print_output_seq(train_label, seq_pre_list)
    print("===Val result===")
    print_output_seq(val_label, seq_val_pre_list)
    # val_and_save(reg_model, val_list, val_label, 10)