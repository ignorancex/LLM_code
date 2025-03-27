from util.experiment import Experiment
from util.data import Data
import traceback
import argparse
import os, sys

def start(args):
    datasets = [Data(data_dir=data_dir, train_plus_valid=args.train_plus_valid) for data_dir in args.path_dataset_folder]
    count = 0
    for d in datasets:
        folder_name = args.path_dataset_folder[count]
        experiment = Experiment(dataset=d,
                                model=args.model_name,
                                parameters=vars(args), ith_logger='_' + folder_name,
                                store_emb_dataframe=True, storage_path=folder_name.split("Triples")[0])
        count += 1
        try:
            experiment.train_and_eval()
            print()
        except RuntimeError as re:
            print(re)
            traceback.print_exc()
            print('Exit.')
            exit(1)


if __name__ == '__main__':
    base_path = os.path.dirname(os.path.realpath(__file__)).split('Embeddings')[0]
    folders = [f for f in os.listdir(base_path+"Datasets/") if not '.ip' in f]
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ConEx')
    parser.add_argument('--num_of_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--scoring_technique', default='KvsAll',
                        help="KvsAll technique or Negative Sampling. For Negative Sampling, use any positive integer as input parameter")
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=.01)
    parser.add_argument('--optim', type=str, default='RMSprop', help='Choose optimizer: Adam or RMSprop')
    parser.add_argument('--decay_rate', default=None)
    parser.add_argument('--train_plus_valid', default=False)
    parser.add_argument('--embedding_dim', type=int, default=20)
    parser.add_argument('--input_dropout', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=12.0, help='Distance parameter')
    parser.add_argument('--hidden_dropout', type=float, default=0.1)
    parser.add_argument('--feature_map_dropout', type=float, default=0.1)
    parser.add_argument('--num_of_output_channels', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument("--path_dataset_folder", nargs='+', type=str, default=[base_path+'Embeddings/'+f+"/Triples/" for f in folders])
    parser.add_argument('--num_workers', type=int, default=4, help='Number of cpus used during batching')
    start(parser.parse_args())
