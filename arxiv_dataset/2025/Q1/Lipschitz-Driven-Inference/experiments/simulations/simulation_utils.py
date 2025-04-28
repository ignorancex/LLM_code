import argparse

def basic_parser():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", type=int, default=250)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--m", type=int, default=100)
    parser.add_argument("--noise_std", type=float, default=1.0)
    parser.add_argument("--num_neighbors", type=int, default=1)
    parser.add_argument(
        "--bandwidth",
        type=float,
        nargs="+",
        default=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    )
    # Add a bandwidth argument
    parser.add_argument("--num_threads", type=int, default=5)
    return parser