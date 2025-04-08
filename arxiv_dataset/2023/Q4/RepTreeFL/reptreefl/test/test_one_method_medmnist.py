import argparse
import warnings
import random
import time
from plots.plots import*
from read_files import *
import constants
from utils.utils import *
from datasets.dataset_medmnist import *
from clients.SGRepClassifier import *

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("running on GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("running on GPU MPS")
else:
    device = torch.device("cpu")
    print("running on CPU")

random.seed(constants.MANUAL_SEED)
np.random.seed(constants.MANUAL_SEED)
torch.manual_seed(constants.MANUAL_SEED)

def run(args):
    create_directories_for_saving()

    train_dataset, val_dataset, test_dataset = get_medmnist_dataset(balanced=args.balanced)
    fold_indices = generate_fold_indices_dataset(train_dataset)
    fold_configuration = generate_fold_configurations(NUMBER_FOLDS)

    metrics = ['acc', 'sens', 'spec', 'f1']
    run_name = args.run_name

    models_fedrep_depth2 = []
    for fold in range(NUMBER_FOLDS):
        print(f'Fold {fold}')

        current_fold_configuration = fold_configuration[fold]
        test_index = fold_indices[current_fold_configuration[-1]]

        test_set = Subset(train_dataset, test_index)
        test_loader = DataLoader(test_set, batch_size=constants.BATCH_SIZE, shuffle=False)
        
        models_fedrep_depth2.append([])
        for client in range(NUMBER_CLIENTS):
            model = SGRepClassifier(fold, client, args.algorithm, no_classes=constants.NUMBER_OF_CLASSES)
            model.set_generator_saving_path(f'{constants.SAVING_FOLDER_PATH}/{run_name}/weights/weight_run{fold}_client{client}_replica0.model')
            models_fedrep_depth2[fold].append(model)

            models_fedrep_depth2[fold][client].test(test_loader)

        for current_metric in metrics:
            save_metric_to_file(models_fedrep_depth2, fold, current_run_name=run_name, metric=current_metric)

    for current_metric in metrics:
        save_average_metric_to_file(models_fedrep_depth2, current_run_name=run_name, metric=current_metric)




if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('-alg', "--algorithm", default="baseline")

    # dataset
    parser.add_argument('-dataset', "--dataset", default="pneumoniamnist")
    parser.add_argument('-classes', "--no_classes", type=int, default=2)
    parser.add_argument('-dataset_size', "--desired_dataset_subset_size", type=int, default=800)
    parser.add_argument('-balanced', "--balanced", type=bool, default=False)

    # directories
    parser.add_argument('-global_run_name', "--global_run_name", default="global_run_name")
    parser.add_argument('-run_name', "--run_name", default="run_name")
    parser.add_argument('-paths', "--paths", default="local")

    args = parser.parse_args()

    constants.ALGORITHM = args.algorithm

    constants.MEDMNIST_DATASET = args.dataset
    constants.NUMBER_OF_CLASSES = args.no_classes
    constants.DESIRED_DATASET_SUBSET_SIZE = args.desired_dataset_subset_size

    constants.GLOBAL_RUN_NAME = args.global_run_name
    constants.RUN_NAME = args.run_name
    if args.paths == "local":
        constants.DATASET_FOLDER_PATH = constants.DATASET_FOLDER_PATH_LOCAL
        constants.SAVING_FOLDER_PATH = constants.SAVING_FOLDER_PATH_LOCAL
    else:
        constants.DATASET_FOLDER_PATH = constants.DATASET_FOLDER_PATH_REMOTE
        constants.SAVING_FOLDER_PATH = constants.SAVING_FOLDER_PATH_REMOTE

    print("=" * 50)
    print("Algorithm: {}".format(args.algorithm))

    print("Dataset: {}".format(args.dataset))
    print("Classes: {}".format(args.no_classes))
    print("Dataset subset size: {}".format(args.desired_dataset_subset_size))
    print("Balanced: {}".format(args.balanced))
    print("Global run name: {}".format(args.global_run_name))
    print("Run name: {}".format(args.run_name))
    print("Paths: {}".format(args.paths))
    print("Saving folder path: {}".format(constants.SAVING_FOLDER_PATH))

    print("=" * 50)

    run(args)

    print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")