# Convenient script to manually trigger turning all pickle files into more efficient numpy storage;
# speeds up process of loading the concept embeddings significantly.

import asyncio
from dataclasses import dataclass, field

import argparse_dataclass

from bg_randomized_loce.utils.loce_storage_helpers import all_pkls_to_npz


@dataclass
class ParseConfig:
    results_dir: str = field(default='./results')
    npz_dir: str = field(default='./data/results/npz')
    force_update: bool = field(default=False, metadata=dict(type=bool, action="store_true"))

if __name__ == "__main__":
    _parser = argparse_dataclass.ArgumentParser(ParseConfig)
    config: ParseConfig = _parser.parse_args()

    ce_infos = asyncio.run(all_pkls_to_npz(results_dir=config.results_dir, npz_dir=config.npz_dir,
                                           force_update=config.force_update))
