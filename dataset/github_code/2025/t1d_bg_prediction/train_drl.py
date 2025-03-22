import os
import argparse
import torch
import d3rlpy
from d3rlpy.algos import DDPGConfig, TD3Config, SACConfig, NormalNoise
from d3rlpy.models import VectorEncoderFactory
from d3rlpy.dataset import create_fifo_replay_buffer
from d3rlpy.preprocessing import (
    MinMaxActionScaler,
    MinMaxObservationScaler,
    MinMaxRewardScaler,
)
from utils_data import *
from utils_drl import *


class CFG:
    l_bound = 20.0
    u_bound = 420.0
    subjects = [29, 30, 31, 36, 37, 38, 39, 42, 45, 46, 47, 49, 50, 51, 52, 53, 54]
    train_split = [30, 31, 36, 38, 39, 46, 47, 49, 50, 53, 54]
    test_split = [29, 37, 42, 45, 51, 52]
    horizons = list(range(0, -60, -5))
    output_csv_header = ["date", "subject_id", "bgClass", "target", "y_pred"]


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="DiaTrend", type=str)
parser.add_argument("--output_path", type=str, default="outputs")
parser.add_argument("--stack_features_path", type=str, default="stack_features")
parser.add_argument(
    "--exp_name",
    type=str,
    choices=["ddpg", "td3", "sac"],
    help="Model name. Choose from [ddpg, td3, sac]",
    required=True,
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--observation_size", type=int, default=6)
parser.add_argument("--action_size", type=int, default=6)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--activation", type=str, default="tanh")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-4)

if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    if not os.path.exists(args.stack_features_path):
        os.mkdir(args.stack_features_path)

    train_set, test_set, X_cols, y_cols = get_data(
        args.data_path,
        subjects=CFG.subjects,
        horizons=CFG.horizons,
        train_split=CFG.train_split,
        test_split=CFG.test_split,
        scale=False,
    )

    # Get RL environment
    env = DiatrendEnv(
        train_set, X_cols + y_cols, args.observation_size, args.action_size, args.seed
    )
    d3rlpy.envs.seed_env(env, args.seed)
    _ = env.reset()

    # Train model
    n_steps_per_epoch = int(len(train_set) * args.observation_size / args.action_size)
    n_steps = int(args.epochs * n_steps_per_epoch)
    print(f"Training for {n_steps} steps and {n_steps_per_epoch} steps per epoch")

    actor_encoder = VectorEncoderFactory(
        hidden_units=[args.hidden_size] * args.num_layers,
        activation=args.activation,
        dropout_rate=args.dropout,
    )
    critic_encoder = VectorEncoderFactory(
        hidden_units=[args.hidden_size] * args.num_layers,
        activation=args.activation,
        dropout_rate=args.dropout,
    )
    action_scaler = MinMaxActionScaler(minimum=CFG.l_bound, maximum=CFG.u_bound)
    observation_scaler = MinMaxObservationScaler(
        minimum=CFG.l_bound, maximum=CFG.u_bound
    )
    reward_scaler = MinMaxRewardScaler(minimum=0.0, maximum=1.0)

    model_args = {
        "observation_scaler": observation_scaler,
        "action_scaler": action_scaler,
        "reward_scaler": reward_scaler,
        "actor_learning_rate": args.lr,
        "critic_learning_rate": args.lr,
        "actor_encoder_factory": actor_encoder,
        "critic_encoder_factory": critic_encoder,
        "batch_size": args.batch_size,
    }

    if args.exp_name == "ddpg":
        model = DDPGConfig
    elif args.exp_name == "td3":
        model = TD3Config
    elif args.exp_name == "sac":
        model = SACConfig
        model_args.update({"temp_learning_rate": args.lr})

    model = model(**model_args).create(args.device)
    model.build_with_env(env)
    buffer = create_fifo_replay_buffer(limit=n_steps_per_epoch, env=env)
    explorer = NormalNoise(std=0.1)
    model.fit_online(
        env,
        buffer,
        explorer,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        experiment_name=args.exp_name,
    )

    # Evaluate
    input_cols = X_cols + y_cols
    pred_cols = []
    for i in range(0, args.observation_size, args.action_size):
        cols = input_cols[i : args.observation_size] + pred_cols
        preds = model.predict(test_set[cols].values)
        for j in range(args.action_size):
            test_set[f"pred{i + j}"] = preds[:, j]
            pred_cols.append(f"pred{i + j}")

    test_set = test_set.rename(columns={y_cols[-1]: "target", pred_cols[-1]: "y_pred"})
    test_set = test_set[CFG.output_csv_header]

    print_results(test_set)

    test_set.to_csv(f"{args.output_path}/{args.exp_name}_output.csv", index=False)

    # Generate outputs for stacking purposes
    pred_cols = []
    for i in range(0, args.observation_size, args.action_size):
        cols = input_cols[i : args.observation_size] + pred_cols
        preds = model.predict(train_set[cols].values)
        for j in range(args.action_size):
            train_set[f"pred{i + j}"] = preds[:, j]
            pred_cols.append(f"pred{i + j}")

    train_set = train_set.rename(
        columns={y_cols[-1]: "target", pred_cols[-1]: "y_pred"}
    )
    train_set = train_set[CFG.output_csv_header]

    train_set.to_csv(
        f"{args.stack_features_path}/{args.exp_name}_output.csv", index=False
    )
