# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
import argparse


if __name__ == '__main__':
       parser = argparse.ArgumentParser()
       parser.add_argument('--task', type=int, default=1, help="Task to be executed [Link Prediction on Static Graphs: 1,"
                                                               " Link Prediction on Dynamic Graphs: 2]")
       parser.add_argument('--data_path', type=str, help='Dataset path')
       parser.add_argument('--dataset_file', type=str, help="Dataset file")
       parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for SGD')
       parser.add_argument('--attention_heads', type=int, default=8, help="Number of attention heads")
       parser.add_argument('--model', type=int, default=2,
                           help="Model employed. Number depends on the task")
       parser.add_argument('--emb_dim', type=int, default=128, help="Dimension of Node embeddings")
       parser.add_argument('--ns', type=int, default=1, help="Negative samples")
       parser.add_argument('--dropout', type=float, default=0., help="Dropout")
       parser.add_argument('--features', type=bool, default=False, help="Include Node Features")
       parser.add_argument('--node_relabel', type=bool, default=False, help="Relabel Node IDs")
       parser.add_argument('--num_runs', type=int, default=1, help="Number of runs for this set of hyperparameters")

       parser.add_argument('--window', type=int, default=3, help="Window for dynamic/stream tasks")
       args = parser.parse_args()

       if args.task == 1:
              from hive.task.link_prediction.static.train import execute
              execute(args)
       elif args.task == 2:
              from hive.task.link_prediction.dynamic.train import execute
              execute(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
