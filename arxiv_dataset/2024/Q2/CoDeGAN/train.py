import os
import yaml
import json
import torch
import random
import datetime
from dataloader import get_dataloader
from test.test_acc_G import test_acc_G
from test.test_acc_E import test_acc_E
from utils.util import sample_real_data
from utils.visualization import visualize_result
from solvers.solver import get_solvers
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--config", type=str, required=True, help="Path of the config")

if __name__ == "__main__":
    # load important parameters
    args = parser.parse_args()
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    # choose train mode
    few_labels = config["few_labels"]
    pretrain = config["pretrain"]
    meta = config["meta"]
    use_ema = config["use_ema"]

    # save dir
    save_path = "./Result/{}/".format(random.randint(0, 10000))
    model_dir = {"checkpoint": "./checkpoint", "samples": "./samples"}
    for dir_ in model_dir:
        if not os.path.exists(save_path + model_dir[dir_]):
            os.makedirs(save_path + model_dir[dir_])

    # set rand seed
    torch.manual_seed(0)

    # save parameters
    with open(save_path + "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # params
    d_batch_size = config["d_batch_size"]
    critic = config["critic"]
    max_step = config["max_step"]
    test_step = config["test_step"]
    save_step = config["save_step"]
    chain_step = config["chain_step"]

    # get solver
    solver = get_solvers(config)
    dataset = get_dataloader(d_batch_size, config["dataset"])

    # load pretrain_model if needed
    load_path = config["load_path"]
    if pretrain is True:
        if few_labels is True:
            solver.G.load_state_dict(
                torch.load(load_path + "/pre_G.pth", map_location=solver.device)
            )
            solver.D.load_state_dict(
                torch.load(load_path + "/pre_D.pth", map_location=solver.device)
            )
            if use_ema:
                solver.G_EMA.load_state_dict(
                    torch.load(load_path + "/pre_G.pth", map_location=solver.device)
                )
        if hasattr(solver, "EC"):
            solver.EC.load_state_dict(
                torch.load(load_path + "/pre_EC.pth", map_location=solver.device)
            )
        if hasattr(solver, "E"):
            solver.EC.load_state_dict(
                torch.load(load_path + "/pre_E.pth", map_location=solver.device)
            )
    if hasattr(solver, "Classifier"):
        solver.Classifier.load_state_dict(
            torch.load(load_path + "/Classifier.pth", map_location=solver.device)
        )
        solver.Classifier.eval()

    print("start...")

    # load real labeled data
    if few_labels is True:
        few_labels_path = config["few_labels_path"]
        real_datasets = torch.load(few_labels_path)

    ecloss, ezloss, dloss, gloss, ema_iter = (0, 0, 0, 0, 0)
    for epoch in range(max_step):
        time_start = datetime.datetime.now()
        solver.train()

        for idx, (data, label) in enumerate(dataset):
            ema_iter += 1

            if few_labels is True:
                real_data = sample_real_data(real_datasets)
            else:
                real_data = None

            if idx % critic == 0:
                gloss = solver.optimize_parametersG()

                ecloss, ezloss = solver.optimize_parametersE(
                    real_data, ec=epoch >= chain_step - 1
                )

                if use_ema:
                    solver.ema.update(ema_iter)

            dloss = solver.optimize_parametersD(data=data)

            # adjust learning rate
            if config["adjust_lr"] and idx % 5 == 4:
                lr = solver.adjust_learning_rates()

        time_end = datetime.datetime.now()

        solver.eval()

        # save generated images and models
        visualize_result(
            epoch + 1,
            save_path + model_dir["samples"],
            solver.G,
            solver.zn_dim,
            solver.zc_dim,
            solver.device,
        )
        if epoch % save_step == save_step - 1:
            solver.save_model(save_path + model_dir["checkpoint"], epoch=epoch)

        # test disentangle performence
        if epoch % test_step == test_step - 1:
            if hasattr(solver, "Classifier"):
                test_acc_G(
                    solver.G,
                    solver.Classifier,
                    epoch + 1,
                    save_path,
                    dataset_name=config["dataset"],
                )
            elif hasattr(solver, "EC"):
                test_acc_E(
                    solver.EC, epoch + 1, save_path, dataset_name=config["dataset"]
                )
            elif hasattr(solver, "E"):
                test_acc_E(
                    solver.E, epoch + 1, save_path, dataset_name=config["dataset"]
                )

        print(
            "[{}/{}] dloss: {:.4f} gloss: {:.4f} ecloss: {:.4f} ezloss: {:.4f}".format(
                epoch + 1, max_step, dloss, gloss, ecloss, ezloss
            )
        )
        print("{:.4f} minutes...".format((time_end - time_start).seconds / 60.0))
