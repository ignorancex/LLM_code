# imports
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join("../src")))
sys.path.append(os.path.abspath(os.path.join("../data")))
sys.path.append(os.path.abspath(os.path.join("../plotting_scripts")))

from dataset import BaseDataset, DOMAINS
from experiment import Ablator
from model import ModelFactory
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from run_all import get_dataset_path

# Appending system paths
sys.path.append(os.path.abspath(os.path.join("..")))
sys.path.append(os.path.abspath(os.path.join("../src")))
sys.path.append(os.path.abspath(os.path.join("../data")))
sys.path.append(os.path.abspath(os.path.join("../plotting_scripts")))


def plot_results(ablation_result,
                 axis_title_size=20,
                 axis_text_size=14,
                 title=True):

    FACTUAL_COLOR = "#005CAB"
    COUNTERFACTUAL_COLOR = "#E31B23"

    # Plot settings
    plt.figure(figsize=(12, 8))
    bar_width = 0.4
    x = range(len(ablation_result["domain"]))

    # Plotting bars
    mem_bars = plt.bar(x, ablation_result["mem_win"], width=bar_width,
            label="mem_win", color=FACTUAL_COLOR, edgecolor="black")
    cp_bars = plt.bar([i + bar_width for i in x], ablation_result["cp_win"], width=bar_width, 
            label="cp_win", color=COUNTERFACTUAL_COLOR, edgecolor="black")

    # Add value labels above bars
    for bars in [mem_bars, cp_bars]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')

    plt.grid(axis='y', linestyle='-', alpha=0.7, zorder=0)
    plt.gca().set_axisbelow(True)

    # Adding labels and title
    # plt.xlabel(f"Premise: {args.premise}", fontsize=axis_title_size)
    plt.ylabel("Wins", fontsize=axis_text_size)
    if title:
        plt.title(f"Ablation comparison - {ablation_layer_heads} with multiplier {multiplier}\nPremise: {args.premise}",
              fontsize=axis_title_size)
    plt.xticks([i + bar_width/2 for i in x], ablation_result["domain"],
               rotation=45, ha='right')
    plt.xticks(fontsize=axis_text_size)
    plt.yticks(fontsize=axis_text_size)
    plt.legend()
    plt.tick_params(left=False, bottom=False)

    # Adjust bottom margin to prevent label cutoff
    plt.tight_layout(pad=2.0, rect=[0, 0.1, 1, 1])

    # Show plot
    # plt.show()


def run_ablator(model, dataset, batch_size, multiplier,
                experiment, prompt_type, position, ablation_layer_heads, start, end):
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)

    print(f"Ablating '{ablation_layer_heads}' at position '{position}'")
    results = []

    if experiment == "copyVSfactDomain":
        for domain in DOMAINS:
            if domain in excluded_domains:
                print(f"Skipping {domain}")
                continue

            # experiment setup
            ds = BaseDataset(path=dataset,
                            model=model,
                            experiment=experiment,
                            domain=domain,
                            prompt_type=prompt_type,
                            no_subject=False,
                            start=start,
                            end=end,
                            premise=args.premise)
            ablator = Ablator(model=model, dataset=ds, experiment=experiment, batch_size=batch_size)
            ablator.set_heads(heads=ablation_layer_heads, value=multiplier, position=position)

            # run the ablator
            result_df = ablator.run()
            result_df["domain"] = domain
            result_df["ablation_layer_heads"] = str(ablation_layer_heads)

            # shift columns
            columns = ["domain", "ablation_layer_heads"] + [col for col in result_df.columns if
                                                            col not in ["domain", "ablation_layer_heads"]]
            result_df = result_df[columns]

            results.append(result_df)

        ablation_result = pd.concat(results)
    else:
        # experiment setup
        results = []

        ds = BaseDataset(path=dataset,
                         model=model,
                         experiment=experiment,
                         domain=None,
                         prompt_type=prompt_type,
                         no_subject=False,
                         start=start,
                         end=end,
                         premise=args.premise)
        
        # Baseline without ablation
        base_ablator = Ablator(model=model, dataset=ds, experiment=experiment, batch_size=batch_size)
        base_ablator.set_heads(heads=[], value=multiplier, position=position)
        
        # Run the baseline
        base_ablation_result = base_ablator.run()
        base_ablation_result["ablation_layer_heads"] = "[]"
        base_ablation_result["domain"] = "Baseline"

        base_columns = ["ablation_layer_heads"] + [col for col in base_ablation_result.columns if
                                              col not in ["ablation_layer_heads"]]
        base_ablation_result = base_ablation_result[base_columns]
        results.append(base_ablation_result)

        # Run with ablation
        ablator = Ablator(model=model, dataset=ds, experiment=experiment, batch_size=batch_size)
        ablator.set_heads(heads=ablation_layer_heads, value=multiplier, position=position)
        
        # run the ablator
        ablation_result = ablator.run()
        ablation_result["ablation_layer_heads"] = str(ablation_layer_heads)
        ablation_result["domain"] = "Ablated"

        # shift columns
        columns = ["ablation_layer_heads"] + [col for col in ablation_result.columns if
                                              col not in ["ablation_layer_heads"]]
        ablation_result = ablation_result[columns]
        results.append(ablation_result)

    ablation_result = pd.concat(results)
    ablation_result.to_csv(f"{SAVE_FOLDER}/ablation_{position}_{ablation_layer_heads}_{multiplier}.csv",
                                        index=False)

    return ablation_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run domain ablation experiments.")
    parser.add_argument("--dataset", type=str, default="copyVSfactDomain", help="Path to the dataset.")
    parser.add_argument("--start", type=int, default=None, help="Start index of the dataset.")
    parser.add_argument("--end", type=int, default=None, help="End index of the dataset.")
    parser.add_argument("--experiment", type=str, default="copyVSfactDomain", help="Name of the experiment.")
    parser.add_argument("--downsampled_dataset", action="store_true", default=True, help="Whether to use the downnsampled dataset or not.")
    parser.add_argument("--use_mquake", action="store_true", default=False, help="Whether to use the mquake dataset or not.")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name.")
    parser.add_argument("--position", type=str, default="attribute", help="Position setting for the ablator.")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for the ablator.")
    parser.add_argument("--multiplier", type=int, default=5, help="Multiplier value for ablation.")
    parser.add_argument("--prompt_type", type=str, default=None, help="Type of prompt used.")
    parser.add_argument("--ablation_layer_heads", type=str, default="[(10, 7), (11, 10)]",
                        help="Heads to ablate in the format '[(layer, head), ...]'.")
    parser.add_argument("--only_plot", action="store_true", default=False, help="Whether to only plot the results.")
    parser.add_argument("--premise", type=str, default="Redefine", help="Premise to use for the ablator.")

    args = parser.parse_args()

    premise_path = "" if args.premise == "Redefine" else f"_{args.premise}"

    if args.use_mquake:
        SAVE_FOLDER = f"../results/{args.dataset}{premise_path}/attention_modification/{args.model_name}_mquake"
    elif args.downsampled_dataset:
        SAVE_FOLDER = f"../results/{args.dataset}{premise_path}/attention_modification/{args.model_name}_full_downsampled"
    else:
        SAVE_FOLDER = f"../results/{args.dataset}{premise_path}/attention_modification/{args.model_name}_full"

    # Parse ablation_layer_heads
    ablation_layer_heads = eval(args.ablation_layer_heads)
    multiplier = args.multiplier

    if args.only_plot:
        print(f"Plotting results for {args.position} with {args.ablation_layer_heads} and multiplier {args.multiplier}")
        # load the results
        results_path = f"{SAVE_FOLDER}/ablation_{args.position}_{args.ablation_layer_heads}_{args.multiplier}.csv"
        if not os.path.exists(results_path):
            print(f"Results file does not exist: {results_path}")
            exit()
        ablation_result = pd.read_csv(results_path)
        print(f"Ablation result: {ablation_result}")
        plot_results(ablation_result)
        plot_filename = f"{SAVE_FOLDER}/ablation_{args.position}_{args.ablation_layer_heads}_{args.multiplier}.pdf"
        plt.savefig(plot_filename, bbox_inches="tight")
        exit()

    excluded_domains = []

    if args.use_mquake:
        dataset_path = f"../data/full_data_sampled_mquake_{args.model_name}_with_subjects_downsampled.json"
    else:
        # Add premise argument with default value
        args.premise = getattr(args, 'premise', 'Redefine')  # Default to 'Redefine' if not present
        dataset_path = get_dataset_path(args)
    print(f"Dataset path: {dataset_path}")

    if args.dataset == "copyVSfactDomain":
        # no data present for these domains
        with open(dataset_path, "r") as f:
            data = json.load(f)
        data_domains = [row["domain"] for row in data]

        excluded_domains = list(set(DOMAINS) - set(data_domains))
    print(f"Excluded Domains: {excluded_domains}")

    # Load model
    model = ModelFactory.create(args.model_name)

    # Run ablator
    ablation_result = run_ablator(
        model=model,
        dataset=dataset_path,
        batch_size=args.batch_size,
        multiplier=args.multiplier,
        experiment=args.experiment,
        prompt_type=args.prompt_type,
        position=args.position,
        ablation_layer_heads=ablation_layer_heads,
        start=args.start,
        end=args.end
    )

    plot_results(ablation_result)
    plot_filename = f"{SAVE_FOLDER}/ablation_{args.position}_{args.ablation_layer_heads}_{args.multiplier}.pdf"
    plt.savefig(plot_filename, bbox_inches="tight")
