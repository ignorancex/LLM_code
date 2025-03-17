import argparse
import sys
import numpy as np

def init_args():
    parser = argparse.ArgumentParser(
        prog='Ripser output to bars',
        description='''Takes the output from ripser and prints space separated bars, one per line.
Dimensions are separated by empty line.
The first line for each dimension is the dimension.''')

    parser.add_argument('--keep_infinite', action='store_const', const=True, default=False,
        help='Keep the infinite bars (default: do not keep the infinite bars)')

    return parser.parse_args()

def main():
    args = init_args()
    for row in sys.stdin.readlines():
        if row.startswith('persistence intervals in dim'):
            dim = int(row.strip().split()[-1][:-1])
            if dim>0:
                print()
            print(dim)
        elif row.strip().startswith('['):
            split_row = row.strip().replace('[','').replace(')','').split(',')
            bar = tuple(float(x) if x!=' ' else np.inf for x in split_row)
            if bar[1]<np.inf or args.keep_infinite:
                print(f'{bar[0]:.6f} {bar[1]:.6f}')

if __name__ == '__main__':
    main()