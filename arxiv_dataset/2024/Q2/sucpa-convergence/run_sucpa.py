
import argparse
import numpy as np
import os
import torch
from utils import SUCPA, load_data, dataset2numclasses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='sst2')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--beta_init', type=str, default=None)
    parser.add_argument('--repetitions', type=int, default=None)
    parser.add_argument('--random_state', type=int, default=None)
    return parser.parse_args()


def main():

    # Parse arguments
    args = parse_args()
    print()

    if args.beta_init is None and args.repetitions is not None and args.random_state is not None:

        # Run experiment for each seed
        rs = np.random.RandomState(args.random_state)
        seeds = rs.randint(0, 10000, size=args.repetitions)
        for i, seed in enumerate(seeds,1):
            print(f"{i}/{len(seeds)}", end=" ")
            generator = torch.Generator().manual_seed(int(seed))
            beta_init = torch.randn(dataset2numclasses[args.dataset], generator=generator) * 4.
            run(args, beta_init)
    
    elif args.beta_init is not None and args.repetitions is None and args.random_state is None:

        # Run experiment for given beta_init
        beta_init = torch.tensor([float(x) for x in args.beta_init.split(',')])
        run(args, beta_init)
    
    else:
        raise ValueError('Invalid combination of arguments. Either specify --beta_init, or --repetitions and --random_state.')
    
    print('\nDone!\n')


def run(args, beta_init):
    num_classes = beta_init.shape[0]
    beta_init_list = [float(f"{x:.2f}") for x in beta_init]
    beta_init_str = ','.join([str(i) for i in beta_init_list])
    results_dir = f'results/dataset={args.dataset}/steps={args.steps}_beta_init={beta_init_str}'

    if not os.path.exists(results_dir):
        
        print(f'Running experiment for dataset={args.dataset}, steps={args.steps}, beta[0]={beta_init_list}...')

        # Load logits and labels
        logits, labels = load_data(args.dataset, split='test', prefix="")
        class_counts = torch.bincount(labels, minlength=num_classes)

        # Create and run SUCPA model
        model = SUCPA(num_classes=num_classes, steps=args.steps, beta_init=beta_init)
        model.fit(logits, class_counts)

        # Save results
        os.makedirs(results_dir, exist_ok=True)
        model.save_to_disk(results_dir)
        np.save(f'{results_dir}/beta_history.npy', model.beta_history.numpy())
        np.save(f'{results_dir}/jacobian_history.npy', model.jacobian_history.numpy())

    else:
        print(f"Found existing results for dataset={args.dataset}, steps={args.steps}, beta[0]={beta_init_list}. Skipping...")



if __name__ == '__main__':
    main()