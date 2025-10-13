'''
    Taken from https://github.com/axel-levy/axlevy-adp-3d/blob/main/register.py
'''
from chroma.utility.api import register_key
import argparse


def main(args):
    register_key(args.key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, required=True)
    args = parser.parse_args()
    main(args)