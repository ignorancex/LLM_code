import argparse, os, sys, json

parser = argparse.ArgumentParser(
    prog="Agentic-Predictor", description="Configurations for training and testing predictors"
)
parser.add_argument("--data_path", type=str, required=True, help="Path to data dir")
parser.add_argument(
    "--label_ratio",
    type=float,
    default=1.0,
    help="Ratio of labeled samples from the training dataset",
)
parser.add_argument("--llm", type=str, default="ST")
parser.add_argument("--input_dim", type=int, default=384, help="Input dimension")
parser.add_argument(
    "--base_conv",
    type=str,
    default="GATv2Conv",
    choices=["GCNConv", "GINEConv", "GATConv", "GATv2Conv", "TransformerConv"],
    help="Base convolution layer",
)
parser.add_argument(
    "--model_type",
    type=str,
    default="MVP",
    choices=["GNN", "OFA", "MLP", "MVP", "FlowerFormer", "CAP", "NARFormerV2", "PINAT"],
    help="Base modeling method",
)
parser.add_argument(
    "--hidden_dim", type=int, default=512, help="Hidden layer dimension"
)
parser.add_argument("--arch", type=str, default="concat", help="architecture of GNN")
parser.add_argument(
    "--encoding",
    type=str,
    default="multi-view",
    choices=[
        "graph",
        "multi-graph",
        "code",
        "text",
        "code,multi-graph",
        "code,text",
        "text,multi-graph",
        "multi-view",
    ],
    help="Encoding method of workflows",
)
parser.add_argument("--n_layers", type=int, default=3, help="Number of GNN layers")
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument(
    "--weight_decay", type=float, default=5e-4, help="Weight decay for optimizer"
)
parser.add_argument("--n_mlplayers", type=int, default=2, help="Number of MLP layers")
parser.add_argument(
    "--mode",
    type=str,
    default="train-test",
    choices=["train-test", "train", "test", "pretrain"],
    help="Mode of runner",
)
parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
parser.add_argument("--n_repeats", default=3, type=int)
args = parser.parse_args()

# define GPU location
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np

from agent_predictor.evaluate_mvgnn import main as run_test
from agent_predictor.train_mvgnn import main as run_train
from utils import print_message

args.encoding = args.encoding.split(",")
args.encoding = args.encoding if len(args.encoding) > 1 else args.encoding[0]
args.cross_system = None

result_file = "results.jsonl"

# run encoding process (optional) -> training process -> testing process (based on the given mode)
acc, uti = [], []
for i in range(args.n_repeats):
    # Random seed for reproducibility
    args.seed = 2**i
    print_message("log", f"RANDOM SEED: {args.seed}\r\n{args}")
    model = run_train(args)

    if "test" in args.mode:
        test_acc, utility = run_test(args)
        result = {
            "round": i,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "n_layers": args.n_layers,
            "model_type": args.model_type,
            "views": args.encoding,
            "hidden_dim": args.hidden_dim,
            "base_conv": args.base_conv,
            "domain": args.domain,
            "accuracy": test_acc * 100,
            "dropout": args.dropout,
            "utility": utility * 100,
            "label_ratio": args.label_ratio,
        }
        with open(result_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        acc.append(test_acc * 100)
        uti.append(utility * 100)

    print_message(
        "log",
        f"B{args.batch_size}: {args.model_type}-D{args.hidden_dim}\t{args.base_conv}\t{args.domain}\tACC:\t{np.mean(acc):.2f}±{np.std(acc):.2f}",
    )
    print_message(
        "log",
        f"B{args.batch_size}: {args.model_type}-D{args.hidden_dim}\t{args.base_conv}\t{args.domain}\tUTI:\t{np.mean(uti):.2f}±{np.std(uti):.2f}",
    )
