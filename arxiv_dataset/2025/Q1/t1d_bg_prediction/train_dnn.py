import os
import argparse
import torch
from torch import optim
from utils_data import *
from utils_dnn import *


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
    choices=["mlp", "lstm", "gru"],
    help="Model name. Choose from [mlp, lstm, gru]",
    required=True,
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--activation", type=str, default="tanh")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--lr_patience", type=int, default=3)
parser.add_argument("--patience", type=int, default=5)

if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(args.seed)

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
    )
    train_loader, test_loader = get_dataloader(
        train_set, test_set, X_cols, y_cols, args.exp_name, args.batch_size
    )

    # Train model
    input_size = len(X_cols) if args.exp_name == "mlp" else 1

    if args.exp_name == "mlp":
        model = MLPModel(
            input_size,
            [args.hidden_size] * args.num_layers,
            args.dropout,
            args.activation,
        )
    else:
        model = RNNModel(
            args.exp_name, input_size, args.hidden_size, args.num_layers, args.dropout
        )

    get_params_count(model)

    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.lr_patience, min_lr=args.min_lr
    )
    early_stopper = EarlyStopper(patience=args.patience)

    model = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        early_stopper,
        train_loader,
        test_loader,
        args.device,
        args.epochs,
        args.output_path,
        args.exp_name,
    )

    print("\n")

    # Evaluate
    test_set["y_pred"] = predict_in_batches(
        model, test_set[X_cols], args.device, args.exp_name
    )
    test_set = test_set.rename(columns={y_cols[-1]: "target"})
    test_set = rescale_data(test_set, ["target", "y_pred"])
    test_set = test_set[CFG.output_csv_header]

    print_results(test_set)

    test_set.to_csv(f"{args.output_path}/{args.exp_name}_output.csv", index=False)

    # Generate outputs for stacking purposes
    train_set["y_pred"] = predict_in_batches(
        model, train_set[X_cols], args.device, args.exp_name
    )
    train_set = train_set.rename(columns={y_cols[-1]: "target"})
    train_set = rescale_data(train_set, ["target", "y_pred"])
    train_set = train_set[CFG.output_csv_header]

    train_set.to_csv(
        f"{args.stack_features_path}/{args.exp_name}_output.csv", index=False
    )
