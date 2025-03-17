
import pickle

from .data import Dataset


def load_dataset(pt_file_path):
    inputs = []
    with open(pt_file_path, 'rb') as f:
        for _ in range(5):
            inputs.append(pickle.load(f))

    dataset = Dataset(inputs)
    return dataset


def get_action_seq_len(tensor):
    # Example of `tensor`: [0, 38195, 3684, 50265, ... , 50265, 2264, 2, 1, 1, 1, ...]
    #
    # `length` does not count the 0th token, whose value is `0`,
    # as the token is only used as the input of decoder.
    #
    # Therefore, the length counts from 1th token to the last token, `2`.
    length = tensor.tolist().index(2)
    return length


def main():
    train_pt_file_path = 'baseline-processed/bart-program/train.pt'
    dataset = load_dataset(train_pt_file_path)
    action_seq_lengths = tuple(map(get_action_seq_len, dataset.target_ids))

    print('avg:', sum(action_seq_lengths) / len(action_seq_lengths))
    print('max:', max(action_seq_lengths))
    print('min:', min(action_seq_lengths))


if __name__ == '__main__':
    main()

# Example:
# $ python -m semparse_baseline.action_seq_statistics
#
#
# Expected output:
# - avg: 36.13987666355853
# - max: 162
# - min: 10
