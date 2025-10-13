import argparse
import os

import properties
import reference_scorer

parser = argparse.ArgumentParser(
    description='Reference Scorer arguments'
)
parser.add_argument(
    '--data_dir',
    default="/scratch/users/k20116188/fc_evidence_evaluation/reference_scorer_training_data",
    help='Path to training data for reference scorer'
)
parser.add_argument(
    '--training_data_file',
    default="bleurt_finetune_train_balanced.jsonl",
    help='Path to training data for reference scorer'
)
parser.add_argument(
    '--dev_data_file',
    default="bleurt_finetune_dev.jsonl",
    help='Path to dev data for reference scorer'
)
parser.add_argument(
    '--test_data_file',
    default="AVERITEC_FC_Evidence_Evaluation_Responses_no_gold_no_conf_label.xlsx",
    help='Path to test data for evaluating fine-tuned reference scorer'
)
parser.add_argument(
    '--output_dir',
    default="/scratch/users/k20116188/fc_evidence_evaluation/results/reference_scorer",
    help='Output path for reference scorer evaluation results.'
)
parser.add_argument(
    '--results_filename',
    default="results_{}.json",
    help='Output path for reference scorer evaluation results.'
)
parser.add_argument(
    '--samples_filename',
    default="prediction_{}.txt",
    help='Output path for reference scorer evaluation results.'
)
parser.add_argument(
    '--hf_model',
    default="Elron/bleurt-base-512",
    help='Path to HG bleurt model.'
)
parser.add_argument(
    '--finetuned_model',
    default="/scratch/users/k20116188/fc_evidence_evaluation/results/reference_scorer/checkpoint-54000",
    help='Path to fine-tuned model.'
)
parser.add_argument(
    '--train',
    default=False,
    action="store_true",
    help='If set, fine-tunes scorer with data specified through --training_data_path'
)
parser.add_argument(
    '--continue_train',
    default=False,
    action="store_true",
    help='If set, fine-tunes scorer with data specified through --training_data_path'
)
parser.add_argument(
    '--calc_diff',
    default=False,
    action="store_true",
    help='If set, fine-tunes scorer with data specified through --training_data_path'  # todo adjust
)
parser.add_argument(
    '--dataset',
    default="averitec_manual_eval",  # set to vitaminc if jsonl file with claim, evidence, label entries in dicts.
    choices=list(properties.Dataset),
    help='Dataset that is used for evaluation.'
)

args = parser.parse_args()
_DATA_DIR = args.data_dir
_TRAIN_DATASET_PATH = os.path.join(_DATA_DIR, args.training_data_file)
_DEV_DATASET_PATH = os.path.join(_DATA_DIR, args.dev_data_file)
_TEST_DATASET_PATH = os.path.join(_DATA_DIR, args.test_data_file)

_DATASET = properties.Dataset(args.dataset)

_OUTPUT_DIR = args.output_dir
_RESULTS_FILENAME = args.results_filename.format(args.test_data_file.split(".")[0])
_SAMPLES_FILENAME = args.samples_filename.format(args.test_data_file.split(".")[0])

print("Results filename is {}".format(_RESULTS_FILENAME))
print("Samples filename is {}".format(_SAMPLES_FILENAME))

_CALC_DIFF = args.calc_diff
_TRAIN = args.train
_CONTINUE_TRAIN = args.continue_train
if _TRAIN:
    _MODEL_PATH = args.hf_model
else:
    _MODEL_PATH = args.finetuned_model

_BATCH_SIZE = 4
_EPOCHS = 5


def main():
    reference_scorer.run_reference_scorer(train_dataset_path=_TRAIN_DATASET_PATH, dev_dataset_path=_DEV_DATASET_PATH,
                                          test_dataset_path=_TEST_DATASET_PATH, output_path=_OUTPUT_DIR,
                                          results_filename=_RESULTS_FILENAME, samples_filenames=_SAMPLES_FILENAME,
                                          _model_path=_MODEL_PATH, train=_TRAIN, continue_train=_CONTINUE_TRAIN,
                                          train_bs=_BATCH_SIZE, test_bs=_BATCH_SIZE, epoch=_EPOCHS, dataset=_DATASET,
                                          calc_diff_base_data=_CALC_DIFF)


if __name__ == '__main__':
    main()
