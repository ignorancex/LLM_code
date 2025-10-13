import json
from datetime import datetime
from pathlib import Path

import click
import pandas as pd


def search_log_txt(search_log_txt_dir):
    if isinstance(search_log_txt_dir, str):
        search_log_txt_dir = [search_log_txt_dir]
    search_log_txt_dir = [Path(_dir) for _dir in search_log_txt_dir]

    log_txt_path_list = []
    # recursively find
    for _dir in search_log_txt_dir:
        log_txt_path_list.extend(list(_dir.rglob("log.txt")))

    if len(log_txt_path_list) == 0:
        raise FileNotFoundError(
            f"No log txt file found in the search_log_txt_dir: {search_log_txt_dir}"
        )
    return log_txt_path_list


def load_time_from_log_txt(log_txt_path):
    """
    Start time: 2025-03-22 16:34:46.117305
    End time: 2025-03-22 16:39:09.261788
    Script runtime: 00:04:23
    """
    runtime_str = "NA"

    with open(log_txt_path, "r") as f:
        for line in f.readlines():
            if line.startswith("Script runtime:"):
                runtime_str = line.lstrip("Script runtime:").strip()
                break

    return runtime_str


def gather_time(search_log_txt_dir):
    log_txt_path_list = search_log_txt(search_log_txt_dir)

    time_dict_list = []
    for log_txt_path in log_txt_path_list:
        time_dict = {
            "path": log_txt_path,
            # ignore "version_/metrics.json"
            "exp_name": log_txt_path.parents[1].name,
            "runtime": load_time_from_log_txt(log_txt_path),
        }
        time_dict_list.append(time_dict)

    df = pd.DataFrame(time_dict_list)
    df.sort_values(by="exp_name", inplace=True)
    return df


@click.command()
@click.option("--search_log_txt_dir_list", "-d", default=["outputs"], multiple=True)
@click.option("--output_dir", "-o", default="outputs")
@click.option("--output_name", "-n", default="all_exp_time")
def main(
    search_log_txt_dir_list=None,
    output_name=None,
    output_dir=None,
):
    df_list = []
    for search_log_txt_dir in search_log_txt_dir_list:
        df = gather_time(search_log_txt_dir)
        df_list.append(df)
    all_df = pd.concat(df_list, ignore_index=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formatted_date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = output_dir / f"{output_name}.{formatted_date_time}.tsv"
    all_df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
