
"""
Main function to transfer pretrained CXR VLMs model.
"""

import argparse
import os

import torch

from cxrvlms.modeling.model import VLMModel
from cxrvlms.transferability.data.dataloader import get_dataloader_splits
from cxrvlms.utils.metrics import evaluate, average_folds_results, save_results
from cxrvlms.modeling.misc import set_seeds
from cxrvlms.transferability.modeling.adapters import LinearProbe, ZeroShot

from local_data.constants import *
from local_data.experiments import get_experiment_setting

import warnings
warnings.filterwarnings("ignore")

set_seeds(42, use_cuda=torch.cuda.is_available())


def init_adapter(model, args):

    if args.method == "lp":
        print("Transferability by Linear Probing...", end="\n")
        adapter = LinearProbe(model, args.setting["targets"], tta=args.tta, fta=args.fta)
    elif args.method == "zero_shot":
        print("Zero-shot classification...", end="\n")
        adapter = ZeroShot(model, args.setting["targets"], tta=args.tta, fta=args.fta,
                           prompt_type=args.prompt_type)
    else:
        print("Adapter not implemented... using LP", end="\n")
        adapter = LinearProbe(args, model.vision_model)

    return adapter


def generate_experiment_id(args):
    id = "{experiment}_{model}_{method}_SHOTS_{shots}".format(
        experiment=args.setting["experiment"].replace("_", ""),
        model=args.weights_path.split("/")[-1].replace("_", "").split(".")[0],
        method=args.method, shots=args.shots_train)

    return id


def process(args):

    # Create results folder
    if not os.path.isdir('./local_data/results/transferability/'):
        os.makedirs('./local_data/results/transferability/')

    # KFold cross-validation
    args.metrics_test, args.metrics_external, args.weights = [], [[] for i in range(len(args.experiment_test))], []
    for iFold in range(args.folds):
        print("\nTransferability (fold : " + str(iFold + 1) + ")", end="\n")
        args.iFold = iFold

        # Get specific experiment settings (i.e. dataframe path, classes, tasks, ...)
        args.setting = get_experiment_setting(args.experiment)

        # Init model
        model = VLMModel(vision_type=args.architecture, from_checkpoint=args.load_weights,
                         weights_path=args.weights_path, projection=args.project_features,
                         norm_features=args.norm_features, vision_pretrained=args.init_imagenet,
                         learning_criteria=args.learning_criteria, caption=args.caption)

        # Set datasets
        args.loaders = get_dataloader_splits(args.setting["dataframe"],
                                             args.data_root_path + args.setting["base_samples_path"],
                                             args.setting["targets"], norm=args.norm,
                                             shots_train=args.shots_train, shots_val=args.shots_val,
                                             shots_test=args.shots_test, balance=args.balance,
                                             batch_size=args.batch_size, num_workers=args.num_workers, seed=iFold,
                                             size=args.size)

        # For two-head approach, select unimodal head for base classes - otherwise, use textual projection
        if args.learning_criteria == "dlilp":
            if "new" not in args.experiment:
                model.logit_scale = model.logit_scale_head2
                model.vision_model.projection_head_vision = model.vision_model.projection_head2
                model.learning_criteria = "unimodal"

        # Set adapter
        adapter = init_adapter(model, args)

        # Fit adapter
        adapter.fit(args.loaders)

        # Test model - predict and evaluate
        if args.loaders["test"] is not None:
            refs, preds = adapter.predict(args.loaders["test"])
            metrics_fold = evaluate(refs, preds)
            args.metrics_test.append(metrics_fold)

        # Store weights
        args.weights.append(adapter.model.state_dict())

        # External testing for OOD
        if args.experiment_test[0] != "":
            for i_external in range(len(args.experiment_test)):
                print("External testing: " + args.experiment_test[i_external])

                # Get setting
                setting_external = get_experiment_setting(args.experiment_test[i_external])

                # Prepare dataloaders
                loaders_external = get_dataloader_splits(setting_external["dataframe"],
                                                         args.data_root_path + setting_external["base_samples_path"],
                                                         args.setting["targets"], shots_train="0%", shots_val="0%",
                                                         shots_test="100%", balance=False, norm=args.norm,
                                                         batch_size=args.batch_size, num_workers=args.num_workers,
                                                         seed=iFold, size=args.size)
                # Test model - predict and evaluate
                refs, preds = adapter.predict(loaders_external["test"])
                metrics = evaluate(refs, preds)
                args.metrics_external[i_external].append(metrics)

    # Get metrics averaged across folds
    if args.loaders["test"] is not None:
        print("\nTransferability (cross-validation)", end="\n")
        args.metrics = average_folds_results(args.metrics_test)
    else:
        args.metrics = None

    # Save experiment metrics
    save_results(args.metrics, args.out_path, id_experiment=generate_experiment_id(args),
                 id_metrics="metrics", save_model=args.save_model, weights=args.weights)

    # Get metrics averaged across fold for external testing
    if args.experiment_test[0] != "":
        for i_external in range(len(args.experiment_test)):
            print("External testing: " + args.experiment_test[i_external])
            metrics = average_folds_results(args.metrics_external[i_external])
            # Save experiment metrics
            save_results(metrics, args.out_path, id_experiment=generate_experiment_id(args),
                         id_metrics=args.experiment_test[i_external], save_model=False)


def main():
    parser = argparse.ArgumentParser()

    # Folders, data, etc.
    parser.add_argument('--data_root_path', default=PATH_DATASETS)
    parser.add_argument('--out_path', default=PATH_RESULTS_TRANSFERABILITY, help='output path')
    parser.add_argument('--save_model', default=False, type=lambda x: (str(x).lower() == 'true'))

    # Experiment
    parser.add_argument('--experiment', default='chexpert_5x200',
                        help='chexpert_5x200 - mimic_5x200 - ssim_train - rsna_train - rsna_gloria_train'
                             'covid_train_2class - covid_train_4class'
                             'nihlt_train - nihlt_train_base - nihlt_train_new -'
                             'vindr_train - vindr_train_base - vindr_train_new')
    parser.add_argument('--experiment_test', default="",
                        help='ssim_test - rsna_test - rsna_gloria_test '
                             'covid_test_2class - covid_test_4class '
                             'nihlt_test - nihlt_test_base - nihlt_test_new -'
                             'vindr_test - vindr_test_base - vindr_test_new',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--method', default='zero_shot', help='lp - zero_shot')

    # Model base weights and architecture
    parser.add_argument('--weights_path', default='./cxrvlms/modeling/pretrained_weights/unimodal_CM.bin',
                        help='clip_CM - unicl_CM - unimodal_CM - dlilp_CM'
                             'other/medclip_weights.bin - other/cxr-clip.tar - other/gloria_weights.ckpt'
                             'other/medklip.pth - other/biovil.pt - other/KAD.pt')
    parser.add_argument('--load_weights', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--init_imagenet', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--architecture', default='resnet_v2', help='resnet_v2')
    parser.add_argument('--project_features', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--norm_features', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--learning_criteria', default="unimodal", help="unicl-clip-unimodal-dlilp")

    # Type of prompt used for inference
    parser.add_argument('--caption', default="A radiology image of [CLS]", help="A radiology image of [CLS]")
    parser.add_argument('--prompt_type', default='ensemble', help='names,description,ensemble')

    # Dataloaders: Training Validation - Testing
    parser.add_argument('--shots_train', default="16", type=lambda x: (str(x)))
    parser.add_argument('--shots_val', default="0%", type=lambda x: (str(x)))
    parser.add_argument('--shots_test', default="20%", type=lambda x: (str(x)))
    parser.add_argument('--balance', default=True, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--folds', default=5, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--size', default=224, help="224 | 512 ", type=int)
    parser.add_argument('--norm', default=False, type=lambda x: (str(x).lower() == 'true'))

    # Adapters augmentation strategies
    parser.add_argument('--fta', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--tta', default=False, type=lambda x: (str(x).lower() == 'true'))

    # Resources
    parser.add_argument('--num_workers', default=0, type=int, help='workers number for DataLoader')

    args, unknown = parser.parse_known_args()

    process(args=args)


if __name__ == "__main__":
    main()