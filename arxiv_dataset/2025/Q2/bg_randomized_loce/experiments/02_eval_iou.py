import asyncio
import logging
import os
from datetime import datetime

import argparse_dataclass

from bg_randomized_loce.utils.eval_util import IoUEvalConfig, calc_ious_global
from bg_randomized_loce.utils.logging import init_logger, log_info, get_current_git_hash, log_warn

if __name__ == '__main__':
    _parser = argparse_dataclass.ArgumentParser(IoUEvalConfig)
    config: IoUEvalConfig = _parser.parse_args()

    # some logging
    if config.output_dir or config.cache_dir:
        init_logger(os.path.join(config.output_dir if config.output_dir else config.cache_dir,
                                 "logs", f"iou_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"),
                    log_level=logging.INFO)
    log_info(f"Git Commit Hash: {get_current_git_hash()}")
    log_info(f"Starting IoU collection with config: {config}")

    # actual iou calculation
    results_df, errors = asyncio.run(calc_ious_global(config))

    # storage handling
    out_file = config.get_output_path()
    if config.output_dir is not None:
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        log_info(f"Storing results to file {out_file} ...")
        results_df.to_csv(out_file, index=False)
        log_info("DONE.")

    if config.verbose <= 1:
        for e in errors: log_warn(e)
