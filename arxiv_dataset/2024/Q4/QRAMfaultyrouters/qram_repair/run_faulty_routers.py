import argparse
import sys

sys.path.append("/gpfs/gibbs/project/puri/dkw34/QRAMfaultyrouters/QRAMfaultyrouters")
sys.path.append("/Users/danielweiss/PycharmProjects/QRAMfaultyrouters")
from quantum_utils import generate_file_path

from qram_repair import MonteCarloRouterInstances

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="run monte carlo instances of faulty routers"
    )
    parser.add_argument(
        "--idx",
        default=-1,
        type=int,
        help="if -1, automatically generate a new file number for saving. Otherwise, "
        "use the passed number.",
    )
    parser.add_argument("--n", default=5, type=int, help="tree depth")
    parser.add_argument("--eps", default=0.04, type=float, help="failure rate")
    parser.add_argument(
        "--num_instances",
        default=10000,
        type=int,
        help="number of monte carlo instances",
    )
    parser.add_argument("--rng_seed", default=4674, type=int, help="rng seed")
    parser.add_argument(
        "--top_three_functioning",
        default=1,
        type=int,
        help="whether or not the top three routers should always be functioning",
    )
    parser.add_argument(
        "--run_brute_force", default=0, type=int, help="run brute force optimization"
    )
    parser.add_argument("--num_cpus", default=1, type=int, help="number of cpus")
    args = parser.parse_args()
    if args.idx == -1:
        file_str = (
            f"faulty_routers_n_{args.n}_eps_{args.eps}"
            f"_num_inst_{args.num_instances}_top_three_{args.top_three_functioning}"
        )
        filename = generate_file_path("h5py", file_str, "out")
    else:
        filename = f"out/{str(args.idx).zfill(5)}_faulty_routers.h5py"
    mcinstance = MonteCarloRouterInstances(
        args.n,
        args.eps,
        num_instances=args.num_instances,
        rng_seed=args.rng_seed,
        top_three_functioning=args.top_three_functioning,
        filepath=filename,
    )
    mcinstance.run(num_cpus=args.num_cpus, run_brute_force=args.run_brute_force)
