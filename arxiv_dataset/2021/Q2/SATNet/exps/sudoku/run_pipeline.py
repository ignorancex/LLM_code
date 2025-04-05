import argparse
from pathlib import Path

from run import run

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='sudoku')
    parser.add_argument('--load-model', type=str)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--no-train', action='store_true')
    parser.add_argument('--leak-labels', action='store_true')
    parser.add_argument('--mode', required=False)
    parser.add_argument('--num-errors', type=int, default=0)
    parser.add_argument('--solvability', choices=['solvable', 'unsolvable', 'any'], default='any')
    parser.add_argument('--infogan-labels-dir', type=str)

    args = parser.parse_args()
    infogan_path = args.load_model


    print(f'Extracting Perm...')
    args.mode = 'train-satnet-visual-infogan'
    infogan_satnet_model = run(args, num_experiment_repetitions=1, num_epochs=2)[0]

    print(f'Generating Dataset...')
    args.mode = 'satnet-visual-infogan-generate-dataset'
    run(args, num_experiment_repetitions=1)

    print(f'Distilling...')
    args.mode = 'train-backbone-lenet-supervised'
    args.infogan_labels_dir = '.'
    args.load_model = Path(infogan_satnet_model)/'it2.pth'
    distilled_model = run(args, num_experiment_repetitions=1, num_epochs=1)[0]

    print(f'Training SATNet E2E...')
    args.mode = 'visual'
    args.load_model = Path(distilled_model)/'it1.pth'
    args.infogan_labels_dir = None
    distilled_model = run(args, num_experiment_repetitions=1, num_epochs=100)[0]



    
