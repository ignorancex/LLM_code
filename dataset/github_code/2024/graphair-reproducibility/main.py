import time

import torch
import wandb

from models.fairgraph.dataset import NBA, POKEC, Citeseer, Cora, PubMed
from models.fairgraph.method import run
from utils import (
    get_best_params,
    parse_arguments,
    plot_homophily,
    plot_spearman,
    set_seed,
)


class FairGNN:
    def __init__(self, model, args, seed, device) -> None:
        self.model = model
        self.args = args
        self.seed = seed
        self.device = device
        self.dataset = self.get_dataset(self.args.dataset)
        self.hyperparameters = self.process_hparams()

    def get_dataset(self, dataset_name: str) -> None:
        dataset_classes = {
            "pokec": POKEC,
            "nba": NBA,
            "citeseer": Citeseer,
            "cora": Cora,
            "pubmed": PubMed,
        }

        # POKEC dataset has two samples, Z and N. choose the correct sample
        if "pokec" in dataset_name:
            return dataset_classes["pokec"](dataset_sample=dataset_name)
        else:
            return dataset_classes[dataset_name]()

    def process_hparams(self) -> dict[str, list]:
        hparams = {}
        
        # only pass relevant args to wandb sweep config
        ignored_args = ['dataset', 'model', 'use_best_params', 'homophily']
        for arg, value in vars(self.args).items():
            if arg not in ignored_args:
                if not isinstance(value, list):
                    value = [value]
                hparams[arg] = {'values': list(value)}

        # if use_best_params flag is enabled, get best params from json file if they exist
        if args.use_best_params:
            best_hparams = get_best_params(args.dataset)
            if best_hparams:
                for param, value in best_hparams[self.model]['hyperparameters'].items():
                    if not isinstance(value, list):
                        value = [value]
                    print(f'Updating hyperparameter {param} with value {value}')
                    hparams[param] = {'values': list(value)}

        return hparams

    def run_gnn(self, config=None):
        with wandb.init(
            config=config,
            name=f'{self.model}-{self.args.dataset}-{time.strftime("%Y%m%d%H%M")}',
        ) as logger:
            config = wandb.config
            # set seed for reproducibility
            set_seed(self.seed)

            # initialize train architecture runner
            run_fairgraph = run()
            print("running with hparams:")
            for param, value in config.items():
                print(f'{param}: {value}')

            # train the model and perform evaluation
            results = run_fairgraph.run(
                model=model,
                dataset=self.dataset,
                device = device,
                config=config,
                logger=logger,
            )

            if self.model == "graphair":
                # plot homophily and spearman correlation
                if args.homophily:
                    plot_homophily(self.dataset, results)
                    plot_spearman(self.dataset, results)


if __name__ == "__main__":
    # set seed value
    seed = 20

    # Parse command line arguments
    args = parse_arguments()

    # select model
    model = args.model

    # Set device, but Graphair requires GPU regardless
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("running training with the following args:\n")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    runner = FairGNN(model, args, seed, device)

    config = {"method": "grid", "metric": {"name": "results/accuracy", "goal": "maximize"}}

    config["parameters"] = runner.hyperparameters
    wandb_sweeper = wandb.sweep(config, project="graphair-reproducibility")
    wandb.agent(wandb_sweeper, function=runner.run_gnn)
