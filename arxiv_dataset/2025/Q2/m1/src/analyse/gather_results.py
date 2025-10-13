import json
from datetime import datetime
from pathlib import Path

import click
import pandas as pd


def search_json(search_json_dir):
    if isinstance(search_json_dir, str):
        search_json_dir = [search_json_dir]
    search_json_dir = [Path(_dir) for _dir in search_json_dir]

    json_path_list = []
    # recursively find
    for _dir in search_json_dir:
        json_path_list.extend(list(_dir.rglob("metrics.json")))

    if len(json_path_list) == 0:
        raise FileNotFoundError(
            f"No json file found in the search_json_dir: {search_json_dir}"
        )
    return json_path_list


def get_clean_exp_name(exp_name):
    clean_exp_name = exp_name
    clean_exp_name = clean_exp_name.split("-")[0]

    clean_exp_name = clean_exp_name.split("_")[-1]
    try:
        clean_exp_name = int(clean_exp_name)
    except ValueError:
        clean_exp_name = exp_name

    return clean_exp_name


def load_metrics_to_df(json_path_list, use_clean_exp_name=False):
    metrics_dict_list = []

    for json_path in json_path_list:
        # ignore "version_/metrics.json"
        exp_name = json_path.parents[1].name
        if use_clean_exp_name is True:
            clean_exp_name = get_clean_exp_name(exp_name)
        else:
            clean_exp_name = exp_name

        metrics_dict = {
            "path": json_path,
            "exp_name": exp_name,
            "clean_exp_name": clean_exp_name,
        }

        with open(json_path, "r") as f:
            metrics_dict_ = json.load(f)
        metrics_dict.update({k: v["accuracy"] for k, v in metrics_dict_.items()})
        metrics_dict_list.append(metrics_dict)

    df = pd.DataFrame(metrics_dict_list)
    df.sort_values("clean_exp_name", inplace=True)
    return df


def gather_and_plot_metrics(
    search_json_dir,
    use_clean_exp_name=False,
    *args,
    **kwargs,
):
    json_path_list = search_json(search_json_dir)
    df = load_metrics_to_df(json_path_list, use_clean_exp_name)

    return df


output_name = "Thinking Budget, 1k SFT Data, 7B Model"
search_json_dir = "outputs"
search_json_dir_list = [search_json_dir]
output_dir = "outputs"


@click.command()
@click.option("--search_json_dir_list", "-d", default=["outputs"], multiple=True)
@click.option("--output_dir", "-o", default="outputs")
@click.option("--output_name", "-n", default="all_exp")
def main(
    search_json_dir_list=None,
    output_name=None,
    output_dir=None,
):
    df_list = []
    for search_json_dir in search_json_dir_list:
        df = gather_and_plot_metrics(search_json_dir)
        df_list.append(df)
    all_df = pd.concat(df_list)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formatted_date_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = output_dir / f"{output_name}.{formatted_date_time}.tsv"
    all_df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
