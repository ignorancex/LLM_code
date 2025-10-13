import dotenv

dotenv.load_dotenv()

import json
from multiprocessing import Barrier
from types import SimpleNamespace

import click
from run_api_inference import _main


@click.command()
@click.option(
    "--config_path",
    type=str,
    default="outputs/sft-vcot/gpt-4o-med-vlm-pmc_vqa/args.json",
    help="Path to the configuration file",
)
@click.option("--local_dp_rank", type=int, default=0, help="Local data parallel rank")
@click.option("--global_dp_rank", type=int, default=0, help="Global data parallel rank")
def main(**kwargs):
    config_path = kwargs.get("config_path")
    with open(config_path, "r") as f:
        args = json.load(f)
    args = SimpleNamespace(**args)

    local_dp_rank = kwargs.get("local_dp_rank")
    global_dp_rank = kwargs.get("global_dp_rank")
    barrier = Barrier(1)

    fn_kwargs = {
        "dp_size": args.dp_size,
        "local_dp_rank": local_dp_rank,
        "global_dp_rank": global_dp_rank,
        "dp_master_ip": "",
        "dp_master_port": 0,
        "tp_size": args.tp_size,
        "args": args,
        "barrier": barrier,
    }

    _main(**fn_kwargs)


if __name__ == "__main__":
    main()
