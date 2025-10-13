import os
import numpy as np
import csv
import time
import wandb
import torch
import random
import torch.nn as nn
import hydra
import pandas as pd

from pathlib import Path
from functools import partial
from omegaconf import DictConfig, OmegaConf

from source.models import ModelFactory, LogisticRegression, LogisticRegressionMixing
from source.dataset import ExtractedFeaturesDataset
from source.dataset_utils import collate_features
from source.utils import (
    initialize_wandb,
    train,
    test,
    compute_time,
    log_on_step,
    OptimizerFactory,
    SchedulerFactory,
    seed_torch,
    next_candidates,
    calculate_stats,
)

@hydra.main(version_base="1.2.0", config_path="config/training", config_name="global")
def main(cfg: DictConfig):

    for strategy in cfg.strategies:
        print("SEED: {}, DATASET: {}, LEVEL: {}, BATCH_SIZE: {}, STRATEGY: {}, INITIALIZATION: {}".format(cfg.seed, cfg.dataset_name, cfg.level, cfg.batch_size, strategy, cfg.model_name))
        
        # set 16 CPU cores for computation
        os.environ["NUMEXPR_MAX_THREADS"] = "16"
        # seed everything (for reproducibility)
        seed_torch(seed=cfg.seed)

        weighted_status = "weighted" if cfg.weighted else "not_weighted"
        root_dir = Path(cfg.level, "seed_{}".format(cfg.seed), "bs_{}".format(cfg.batch_size), "initial_{}_gen_{}_budget_{}".format(cfg.AL.initial_pool, cfg.AL.number_of_generations, cfg.AL.WSI_budget), cfg.model_name, weighted_status)

        output_dir = Path(cfg.output_dir, cfg.dataset_name)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_root_dir = Path(output_dir, "checkpoints", root_dir, strategy)
        result_root_dir = Path(output_dir, "results", root_dir, strategy)
        script_root_dir = Path(output_dir, "scripts", root_dir)

        checkpoint_root_dir.mkdir(parents=True, exist_ok=True)
        result_root_dir.mkdir(parents=True, exist_ok=True)
        script_root_dir.mkdir(parents=True, exist_ok=True)

        # set up wandb
        if cfg.wandb.enable:
            key = os.environ.get("WANDB_API_KEY")
            config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            _ = initialize_wandb(
                cfg.wandb.project,
                cfg.wandb.username,
                cfg.wandb.exp_name,
                dir=cfg.wandb.dir,
                config=config,
                key=key,
            )
            wandb.define_metric("epoch", summary="max")

        features_dir = Path(cfg.output_dir, "pretrain", "features", "hipt", cfg.level, "slide")
    
        data_dir = Path(cfg.data_dir, cfg.dataset_name)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # in order to be added later
        pool = pd.read_csv(Path(data_dir, f"pool.csv")).values.tolist()
        test_df = pd.read_csv(Path(data_dir, "test.csv"))

        test_dataset = tune_dataset = ExtractedFeaturesDataset(
            test_df, features_dir, cfg.label_name, cfg.label_mapping, cfg.model.slide_pos_embed
        )

        used_data = []
        train_data = []
        test_aucs, test_accs, test_precisions, test_recalls = [], [], [], []

        start_time = time.time()
        print(f"Training on {cfg.AL.number_of_generations} generations")
        for g in range(cfg.AL.number_of_generations):

            checkpoint_dir = Path(checkpoint_root_dir, f"gen_{g}")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            result_dir = Path(result_root_dir, f"gen_{g}")
            result_dir.mkdir(parents=True, exist_ok=True)

            print(f"Generation: {g}. Size of WSI pool: {len(pool)}")
            if g == 0:
                print(f"Random sampling in 1st generation")
                candidates = random.sample(pool, cfg.AL.initial_pool)
                train_df = pd.DataFrame(candidates, columns=['slide_id', 'label'])
                train_data += candidates

            else:
                pool_df = pd.DataFrame(pool, columns=['slide_id', 'label'])
                pool_dataset = ExtractedFeaturesDataset(
                    pool_df, features_dir, cfg.label_name, cfg.label_mapping, cfg.model.slide_pos_embed
                )
                train_df = pd.DataFrame(train_data, columns=['slide_id', 'label'])
                train_dataset = ExtractedFeaturesDataset(
                    train_df, features_dir, cfg.label_name, cfg.label_mapping, cfg.model.slide_pos_embed
                )
                train_df, candidates = next_candidates(
                    model=model,
                    pool_dataset=pool_dataset,
                    train_dataset=train_dataset,
                    collate_fn=partial(collate_features, label_type=label_type),
                    batch_size=cfg.batch_size,
                    strategy_name=strategy,
                    WSI_budget=cfg.AL.WSI_budget,
                )
                train_data += candidates

            print(f"Size of train WSI: {len(train_data)}")
            used_data += [x[0] for x in train_data]
            train_df.to_csv(Path(result_dir, "used_data.csv"), index=False)
            pool = [(i, l) for (i, l) in pool if i not in used_data]

            train_df = pd.DataFrame(train_data, columns=['slide_id', 'label'])
            train_dataset = ExtractedFeaturesDataset(
                train_df, features_dir, cfg.label_name, cfg.label_mapping, cfg.model.slide_pos_embed,
            )
                
            print("Backbone Initialization")
            backbone = ModelFactory(cfg.level, cfg.model).get_model()
            backbone.relocate()

            backbone_checkpoint_path = Path(cfg.output_dir, "pretrain", "checkpoints", cfg.level, f"{cfg.model_name}.pth")
            print(backbone_checkpoint_path)
    
            if os.path.exists(backbone_checkpoint_path):
                print("Load Pretrained Backbone")
                checkpoint = torch.load(backbone_checkpoint_path)
                backbone.load_state_dict(checkpoint['net'])
            else:
                print("Random Initialization")

            if getattr(cfg.model, 'mixing', None):
                model = LogisticRegressionMixing(backbone, cfg.num_classes)
            else:
                model = LogisticRegression(backbone, cfg.num_classes)
            model.relocate()
            print(model)
            
            model_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = OptimizerFactory(
                cfg.optim.name, model_params, lr=cfg.optim.lr, weight_decay=cfg.optim.wd
            ).get_optimizer()
            scheduler = SchedulerFactory(optimizer, cfg.optim.lr_scheduler).get_scheduler()

            if cfg.loss == "ce":
                criterion = nn.CrossEntropyLoss()
                label_type = "int"
            elif cfg.loss == "mse":
                criterion = nn.MSELoss()
                label_type = "float"
            elif cfg.loss == "ordinal":
                criterion = nn.MSELoss()
                label_type = "float"

            gen_start_time = time.time()

            if cfg.wandb.enable:
                wandb.define_metric(f"gen_{g}/train/epoch", summary="max")

            best_tune_acc = 0.0
            tune_acc_list = []
            tune_auc_list = []
            tune_precision_list = []
            tune_recall_list = []

            for epoch in range(1, cfg.nepochs + 1):
                epoch_start_time = time.time()
        
                train_results = train(
                    epoch,
                    model,
                    train_dataset,
                    optimizer,
                    criterion,
                    batch_size=cfg.batch_size,
                    weighted_sampling=cfg.weighted,
                    gradient_clipping=cfg.gradient_clipping,
                    mixing=getattr(cfg.model, 'mixing', None),
                    result_dir=result_dir,
                )

                if cfg.wandb.enable:
                    log_on_step(
                        f"generation_{g}/train",
                        train_results,
                        step=f"generation_{g}/train/epoch",
                        to_log=cfg.wandb.to_log,
                    )

                train_dataset.df.to_csv(Path(result_dir, f"train_{epoch}.csv", index=False))

                # tune test dataset to find the highest accuracy
                if cfg.tuning.use and epoch % cfg.tuning.tune_every == 0:

                    tune_results = test(
                        model,
                        tune_dataset,
                        batch_size=cfg.batch_size,
                        result_dir=result_dir,
                    )

                    tune_acc = tune_results["accuracy"]
                    tune_auc = tune_results["auc"]
                    tune_precision = tune_results["precision"]
                    tune_recall = tune_results["recall"]

                    tune_acc_list.append(tune_acc)
                    tune_auc_list.append(tune_auc)
                    tune_precision_list.append(tune_precision)
                    tune_recall_list.append(tune_recall)


                    if tune_acc > best_tune_acc:
                        best_tune_acc = tune_acc
                        print(f"Highest ACC: {tune_acc}, AUC: {tune_auc}, Precision: {tune_precision}, Recall: {tune_recall}")
                        print("Saving..")
                        state = {
                            "net": model.state_dict(),
                            "accuracy": tune_acc,
                            "auc": tune_auc,
                            "precision": tune_precision,
                            "recall": tune_recall,
                            "epoch": epoch,
                        }
                    
                        torch.save(state, Path(checkpoint_dir, "best_model.pth"))

                    if cfg.wandb.enable:
                        log_on_step(
                            f"gen_{g}/tune",
                            tune_results,
                            step=f"gen_{g}/train/epoch",
                            to_log=cfg.wandb.to_log,
                        )

                    tune_dataset.df.to_csv(
                        Path(result_dir, f"tune_{epoch}.csv"), index=False
                    )

                if cfg.wandb.enable:
                    wandb.define_metric(
                        f"gen_{g}/train/lr", step_metric=f"gen_{g}/train/epoch"
                    )
                if scheduler:
                    lr = scheduler.get_last_lr()
                    if cfg.wandb.enable:
                        wandb.log({f"gen_{g}/train/lr": lr})
                    scheduler.step()
                else:
                    if cfg.wandb.enable:
                        wandb.log({f"gen_{g}/train/lr": cfg.optim.lr})
                
                epoch_end_time = time.time()
                epoch_mins, epoch_secs = compute_time(epoch_start_time, epoch_end_time)
                print(
                    f"End of epoch {epoch+1} / {cfg.nepochs} \t Time Taken: {epoch_mins}m {epoch_secs}s"
                )

            
            gen_end_time = time.time()
            gen_mins, gen_secs = compute_time(gen_start_time, gen_end_time)
            print(f"Total time taken for generation {g}: {gen_mins}m {gen_secs}s")

            # save tune loss, tune acc, tune auc
            np.savetxt(Path(result_dir, "tune_acc.txt"), tune_acc_list, fmt='%.2f')
            np.savetxt(Path(result_dir, "tune_auc.txt"), tune_auc_list, fmt='%.2f')
            np.savetxt(Path(result_dir, "tune_precision.txt"), tune_precision_list, fmt='%.2f')
            np.savetxt(Path(result_dir, "tune_recall.txt"), tune_recall_list, fmt='%.2f')

            best_model_fp = Path(checkpoint_dir, "best_model.pth")

            if cfg.wandb.enable:
                wandb.save(str(best_model_fp))
            best_model_sd = torch.load(best_model_fp)
            model.load_state_dict(best_model_sd["net"])

            test_results = test(
                    model, 
                    test_dataset,
                    batch_size=cfg.batch_size,
                    result_dir=result_dir
                )
            test_dataset.df.to_csv(Path(result_dir, f"test.csv"), index=False)

            with open(Path(result_dir, "test_results.csv"), "w+") as test_file:
                test_writer = csv.DictWriter(test_file, fieldnames=test_results.keys())
                test_writer.writeheader()
                test_writer.writerow(test_results)
            
            for (r_test, v_test) in test_results.items():
                if r_test in ["auc", "accuracy", "precision", "recall"]:
                    metric_values = {
                        "auc": test_aucs,
                        "accuracy": test_accs,
                        "precision": test_precisions,
                        "recall": test_recalls,
                    }
                    metric_values[r_test].append(v_test)
                    v_test = round(v_test, 3)
                    print(f"Best test {r_test} on generation {g}: {v_test}")

                if r_test in cfg.wandb.to_log:
                    if cfg.wandb.enable:
                        wandb.log({f"gen_{g}/test/{r_test}": v_test})


        mean_test_auc, std_test_auc = calculate_stats(metric_values["auc"])
        mean_test_acc, std_test_acc = calculate_stats(metric_values["accuracy"])
        mean_test_precision, std_test_precision = calculate_stats(metric_values["precision"])
        mean_test_recall, std_test_recall = calculate_stats(metric_values["recall"])
        
        if cfg.wandb.enable:
            wandb.log({
                "test/auc_mean": mean_test_auc,
                "test/auc_std": std_test_auc,
                "test/acc_mean": mean_test_acc,
                "test/acc_std": std_test_acc
            })

        test_aucs, test_accs, test_precisions, test_recalls = ', '.join(map(str, test_aucs)), ', '.join(map(str, test_accs)), ', '.join(map(str, test_precisions)), ', '.join(map(str, test_recalls))

        with open(os.path.join(script_root_dir, "log.txt"), "a") as writer:

            writer.write("SEED: {}, DATASET: {}, LEVEL: {}, BATCH SIZE: {}, STRATEGY: {}\n".format(cfg.seed, cfg.dataset_name, cfg.level, cfg.batch_size, strategy))
            writer.write("INITIAL POOL: {}, GENERATIONS: {}, BUDGET: {}, MODEL NAME: {}, WEIGHTED: {}\n\n".format(cfg.AL.initial_pool, cfg.AL.number_of_generations, cfg.AL.WSI_budget, cfg.model_name, cfg.weighted))
            writer.write("Test AUC across generations: {}\n".format(test_aucs))
            writer.write("Test AUC mean: {:.3f} +/- {:.3f}\n\n".format(mean_test_auc, std_test_auc))
            writer.write("Test ACC across generations: {}\n".format(test_accs))
            writer.write("Test ACC mean: {:.3f} +/- {:.3f}\n\n".format(mean_test_acc, std_test_acc))
            writer.write("Test Precision across generations: {}\n".format(test_precisions))
            writer.write("Test Precision mean: {:.3f} +/- {:.3f}\n\n".format(mean_test_precision, std_test_precision))
            writer.write("Test Recall across generations: {}\n".format(test_recalls))
            writer.write("Test Recall mean: {:.3f} +/- {:.3f}\n\n".format(mean_test_recall, std_test_recall))
            writer.write("============================================================\n")
            
        end_time = time.time()
        mins, secs = compute_time(start_time, end_time)
        print(f"Total time taken for ({cfg.AL.number_of_generations} generations): {mins}m {secs}s")


if __name__ == "__main__":
    main()
