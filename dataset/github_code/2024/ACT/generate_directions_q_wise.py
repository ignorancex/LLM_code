import argparse
import sys
import numpy as np
import pickle as pkl
from einops import rearrange
from utils import get_separated_activations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama_7B')
    args = parser.parse_args()
    print('Running:\n{}\n'.format(' '.join(sys.argv)))
    print(args)

    head_wise_activations = pkl.load(open(f'./activations/{args.model_name}_head_wise.pkl', 'rb'))
    labels = pkl.load(open(f'./activations/{args.model_name}_labels.pkl', 'rb'))
    num_heads = 32
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)
    separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations)
    
    # generate directions for each question
    head_wise_activation_directions = np.array([a[np.array(l) == 1].mean(axis=0) - a[np.array(l) == 0].mean(axis=0) for a, l in zip(separated_head_wise_activations, separated_labels)])
    pkl.dump(head_wise_activation_directions, open(f'./directions/{args.model_name}_directions.pkl', 'wb'))


if __name__ == '__main__':
    main()