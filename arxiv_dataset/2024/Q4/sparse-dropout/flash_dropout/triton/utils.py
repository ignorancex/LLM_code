from typing import Any

import triton


def filter_invalid_configs(block_size_arg_names: list[str]):
    def filter_fn(
        configs: list[triton.Config], nargs: dict[str, Any]
    ) -> list[triton.Config]:
        valid_configs = []
        for config in configs:
            if all(
                nargs["BLOCK_SIZE"] % config.kwargs[key] == 0
                for key in block_size_arg_names
            ):
                valid_configs.append(config)
        return valid_configs

    return filter_fn
