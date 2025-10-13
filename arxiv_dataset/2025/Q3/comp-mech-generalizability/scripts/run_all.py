# imports
import logging
import os
import sys

from pythonjsonlogger.json import JsonFormatter

# creatting logger and setting info level
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# setting json formatter
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)

# Appending system paths
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join("../src")))
sys.path.append(os.path.abspath(os.path.join("../data")))
sys.path.append(os.path.abspath(os.path.join("../plotting_scripts")))

from plotting_scripts.plot_logit_attribution_fig_3_4a import plot_logit_attribution_fig_3_4a
from plotting_scripts.plot_logit_lens_fig_2 import plot_logit_lens_fig_2
from plotting_scripts.plot_head_pattern_fig_4b import plot_head_pattern_fig_4b
from plotting_scripts.plot_ablation import plot_ablation

from model import load_model

from dataclasses import dataclass

import subprocess
from typing import Optional, Literal, Union

# Third-party library imports
from rich.console import Console
import argparse
import logging
import torch
# Local application/library specific imports

from dataset import BaseDataset  # noqa: E402
from experiment import LogitAttribution, LogitLens, Ablate, HeadPattern  # noqa: E402
from utils import display_config, display_experiments, check_dataset_and_sample, get_hf_model_name  # noqa: E402
console = Console()
# set logging level to suppress warnings
logging.basicConfig(level=logging.ERROR)

@dataclass
class Config:
    mech_fold: Literal["copyVSfact", "contextVSfact", "copyVSfact_factual"] = "copyVSfact"
    model_name: str = "gpt2"
    hf_model_name: str = "gpt2"
    device: str = "cuda"
    premise: str = "Redefine"
    prompt_type: str = None
    domain: str = None
    batch_size: int = 10
    dataset_path: str = f"../data/full_data_sampled_{model_name}.json"
    dataset_slice: Optional[int] = None
    dataset_start: Optional[int] = None
    dataset_end: Optional[int] = None
    produce_plots: bool = True
    normalize_logit: Literal["none", "softmax", "log_softmax"] = "none"
    std_dev: int = 1  # 0 False, 1 True
    total_effect: bool = False
    up_to_layer: Union[int, str] = "all"
    ablate_component:str = "all"
    flag: str = ""
    downsampled_dataset: bool = False

    @classmethod
    def from_args(cls, args):
        return cls(
            mech_fold=args.dataset,
            model_name=args.model_name,
            batch_size=args.batch,
            device=args.device,
            dataset_path= get_dataset_path(args),
            dataset_slice=args.slice,
            dataset_start=args.start,
            dataset_end=args.end,
            premise=args.premise,
            prompt_type=args.prompt_type,
            domain=args.domain,
            produce_plots=args.produce_plots,
            std_dev=1 if not args.std_dev else 0,
            total_effect=args.total_effect if args.total_effect else False,
            hf_model_name= get_hf_model_name(args.model_name),
            ablate_component=args.ablate_component,
            flag = args.flag,
            downsampled_dataset=args.downsampled_dataset
        )

def get_dataset_path(args):
    downsample_string = "_downsampled" if args.downsampled_dataset else ""
    premise_string = f"_{args.premise}" if args.premise != "Redefine" else ""
    if args.dataset == "copyVSfact":
        return f"../data/full_data_sampled_{args.model_name}_with_subjects{downsample_string}{premise_string}.json"
    if args.dataset == "copyVSfactQnA":
        return f"../data/cft_og_combined_data_sampled_{args.model_name}_with_questions{downsample_string}{premise_string}.json"
    if args.dataset == "copyVSfactDomain":
        return f"../data/full_data_sampled_{args.model_name}_with_domains{downsample_string}{premise_string}.json"
    else:
        raise ValueError("No dataset path found for folder: ", args.dataset)

@dataclass
class logit_lens_config:
    component: str = "resid_post"
    return_index: bool = True
    normalize: str = "none"


### check folder and create if not exists
def save_dataframe(folder_path, file_name, dataframe):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    dataframe.to_csv(f"{folder_path}/{file_name}.csv", index=False)

# ----- EXPERIMENTS -----

def logit_attribution(model, dataset, config, args):
    dataset_slice_name = (
        "full" if config.dataset_slice is None else config.dataset_slice
    )
    dataset_slice_name = dataset_slice_name if config.domain is None else f"{dataset_slice_name}_{config.domain}"
    dataset_slice_name = (
        dataset_slice_name if config.up_to_layer == "all" else f"{dataset_slice_name}_layer_{config.up_to_layer}"
    )
    dataset_slice_name = dataset_slice_name if not args.downsampled_dataset else f"{dataset_slice_name}_downsampled"
    dataset_slice_name = dataset_slice_name if args.premise == "Redefine" else f"{dataset_slice_name}_{args.premise}"

    print("Running logit attribution")
    attributor = LogitAttribution(dataset, model, config.batch_size // 5, config.mech_fold)
    dataframe = attributor.run(apply_ln=False, normalize_logit=config.normalize_logit, up_to_layer=config.up_to_layer)

    save_dataframe(
        f"../results/{config.mech_fold}{config.flag}/logit_attribution/{config.model_name}_{dataset_slice_name}",
        "logit_attribution_data",
        dataframe,
    )

    if config.produce_plots:
        logit_attribution_plot(config, dataset_slice_name)

def logit_attribution_plot(config, dataset_slice_name):
        plot_logit_attribution_fig_3_4a(model=config.model_name,
                                        model_folder=f'{config.model_name}_{dataset_slice_name}',
                                        experiment=config.mech_fold,
                                        domain=config.domain,
                                        downsampled=config.downsampled_dataset)


def logit_lens(model, dataset, config, args):
    data_slice_name = "full" if config.dataset_slice is None else config.dataset_slice
    data_slice_name = data_slice_name if config.domain is None else f"{data_slice_name}_{config.domain}"
    data_slice_name = data_slice_name if not args.downsampled_dataset else f"{data_slice_name}_downsampled"
    data_slice_name = data_slice_name if args.premise == "Redefine" else f"{data_slice_name}_{args.premise}"

    logit_lens_cnfg = logit_lens_config()
    print("Running logit lens")
    logit_lens = LogitLens(dataset, model, config.batch_size, config.mech_fold)
    dataframe = logit_lens.run(
        logit_lens_cnfg.component,
        logit_lens_cnfg.return_index,
        normalize_logit=config.normalize_logit,
    )
    save_dataframe(
        f"../results/{config.mech_fold}{config.flag}/logit_lens/{config.model_name}_{data_slice_name}",
        "logit_lens_data",
        dataframe,
    )

    if config.produce_plots:
        logit_lens_plot(config, data_slice_name)

def logit_lens_plot(config, data_slice_name):
        plot_logit_lens_fig_2(model=config.model_name,
                              model_folder=f'{config.model_name}_{data_slice_name}',
                              experiment=config.mech_fold,
                              domain=config.domain,
                              downsampled=config.downsampled_dataset)
        

def ablate(model, dataset, config, args):
    data_slice_name = "full" if config.dataset_slice is None else config.dataset_slice
    data_slice_name = data_slice_name if config.domain is None else f"{data_slice_name}_{config.domain}"
    start_slice_name = "" if config.dataset_start is None else f"{config.dataset_start}_"
    data_slice_name = f"{start_slice_name}{data_slice_name}_total_effect" if config.total_effect else data_slice_name
    data_slice_name = data_slice_name if not args.downsampled_dataset else f"{data_slice_name}_downsampled"
    data_slice_name = data_slice_name if args.premise == "Redefine" else f"{data_slice_name}_{args.premise}"

    LOAD_FROM_PT = None
    ablator = Ablate(dataset, model, config.batch_size, config.mech_fold)
    if args.ablate_component == "all":
        dataframe, tuple_results = ablator.run_all(normalize_logit=config.normalize_logit, total_effect=args.total_effect, load_from_pt=LOAD_FROM_PT)
        save_dataframe(
            f"../results/{config.mech_fold}{config.flag}/ablation/{config.model_name}_{data_slice_name}",
            "ablation_data",
            dataframe,
        )
        torch.save(tuple_results, f"../results/{config.mech_fold}{config.flag}/ablation/{config.model_name}_{data_slice_name}/ablation_data.pt")
    else:
        dataframe, tuple_results = ablator.run(args.ablate_component, normalize_logit=config.normalize_logit, total_effect=args.total_effect, load_from_pt=LOAD_FROM_PT)
        save_dataframe(
            f"../results/{config.mech_fold}{config.flag}/ablation/{config.model_name}_{data_slice_name}",
            f"ablation_data_{args.ablate_component}",
            dataframe,
        )
        torch.save(dataframe, f"../results/{config.mech_fold}{config.flag}/ablation/{config.model_name}_{data_slice_name}/ablation_data_{args.ablate_component}.pt")

    if config.produce_plots:
        ablate_plot(config, data_slice_name)

def ablate_plot(config, data_slice_name):
    data_slice_name = f"{data_slice_name}_total_effect" if config.total_effect else data_slice_name
    print("plotting from source: ",  f"../results/{config.mech_fold}/ablation/{config.model_name}_{data_slice_name}")
    subprocess.run(
        [
            "Rscript",
            "../src_figure/ablation.R",
            f"../results/{config.mech_fold}{config.flag}/ablation/{config.model_name}_{data_slice_name}",
            f"{config.std_dev}",
        ]
    )

def pattern(model, dataset, config, args):
    data_slice_name = "full" if config.dataset_slice is None else config.dataset_slice
    data_slice_name = data_slice_name if config.domain is None else f"{data_slice_name}_{config.domain}"
    data_slice_name = data_slice_name if not args.downsampled_dataset else f"{data_slice_name}_downsampled"
    data_slice_name = data_slice_name if args.premise == "Redefine" else f"{data_slice_name}_{args.premise}"
    
    print("Running head pattern")
    pattern = HeadPattern(dataset, model, config.batch_size, config.mech_fold)
    dataframe = pattern.run()
    save_dataframe(
        f"../results/{config.mech_fold}{config.flag}/head_pattern/{config.model_name}_{data_slice_name}",
        "head_pattern_data",
        dataframe,
    )

    if config.produce_plots:
        pattern_plot(config, data_slice_name)

def pattern_plot(config, data_slice_name):
    plot_head_pattern_fig_4b(
        model=config.model_name,
        experiment=config.mech_fold,
        model_folder=f'{config.model_name}_{data_slice_name}',
        domain=config.domain,
        downsampled=config.downsampled_dataset
    )

def main(args):
    config = Config().from_args(args)
    console.print(display_config(config))
    # create experiment folder
    if not os.path.exists(f"../results/{config.mech_fold}"):
        os.makedirs(f"../results/{config.mech_fold}")
    # create experiment folder
    if args.only_plot:
        data_slice_name = "full" if config.dataset_slice is None else config.dataset_slice
        data_slice_name = data_slice_name if not args.downsampled_dataset else f"{data_slice_name}_downsampled"
        plots = []

        if args.logit_attribution:
            plots.append(logit_attribution_plot)
        if args.logit_lens:
            plots.append(logit_lens_plot)
        if args.ablate:
            plots.append(ablate_plot)
        if args.pattern:
            plots.append(pattern_plot)
        if args.all:
            plots = [logit_attribution_plot, logit_lens_plot, ablate_plot, pattern_plot]

        for plot in plots:
            try:
                plot(config, data_slice_name)
            except FileNotFoundError:
                print(f"No {plot.__name__} data found")
        return

    check_dataset_and_sample(config.dataset_path)
    # load model
    model = load_model(config)
    # load the dataset
    dataset = BaseDataset(path=config.dataset_path,
                          experiment=config.mech_fold,
                          model=model,
                          start=args.start, end=args.end,
                          prompt_type=args.prompt_type,
                          domain=args.domain,
                          premise=config.premise,
                          no_subject=False)

    experiments = []
    if args.logit_attribution:
        experiments.append(logit_attribution)
    if args.logit_lens:
        experiments.append(logit_lens)
    if args.ablate:
        experiments.append(ablate)
    if args.pattern:
        experiments.append(pattern)
    if args.all:
        experiments = [logit_attribution, logit_lens, pattern, ablate]

    status = ["Pending" for _ in experiments]


    for i, experiment in enumerate(experiments):
        try:
            status[i] = "Running"
            table = display_experiments(experiments, status)
            console.print(table)
            # print(dataset.full_data)
            experiment(model, dataset, config, args)
            status[i] = "Done"
        except Exception as e:
            status[i] = "Failed"
            logger.error(f"Experiment - {experiment.__name__} Failed - {e}", exc_info=True)

if __name__ == "__main__":
    config_defaults = Config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=config_defaults.model_name)
    parser.add_argument("--slice", type=int, default=config_defaults.dataset_slice)
    parser.add_argument("--start", type=int, default=config_defaults.dataset_start)
    parser.add_argument("--end", type=int, default=config_defaults.dataset_end)
    parser.add_argument("--prompt_type", type=str, default=config_defaults.prompt_type)
    parser.add_argument("--premise", type=str, default=config_defaults.premise)
    parser.add_argument("--domain", type=str, default=config_defaults.domain)
    parser.add_argument("--no-plot", dest="produce_plots", action="store_false", default=True)
    parser.add_argument("--batch", type=int, default=config_defaults.batch_size)
    parser.add_argument("--only-plot", action="store_true")
    parser.add_argument("--std-dev", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--logit-attribution", action="store_true")
    parser.add_argument("--logit_lens", action="store_true")
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--total-effect", action="store_true")
    parser.add_argument("--pattern", action="store_true", default=False)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--ablate-component", type=str, default="all")
    parser.add_argument("--dataset", type=str, default="copyVSfact")
    parser.add_argument("--flag", type=str, default="")
    parser.add_argument("--downsampled-dataset", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
