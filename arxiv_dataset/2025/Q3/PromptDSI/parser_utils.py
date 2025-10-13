def general_subparser(parser):
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--lr", default=5e-3, type=float)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument(
        "--model_encoder", default="incdsi", choices=["incdsi", "l2p", "flexprompt"]
    )
    parser.add_argument(
        "--model_name", default="bert-base-uncased", choices=["bert-base-uncased"]
    )
    parser.add_argument("--seed", default=42, type=int, required=False)
    parser.add_argument("--val_only", action="store_true")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--continue", action="store_true")
    parser.add_argument(
        "--continue_checkpoint",
        default=None,
        type=str,
        help="checkpoint to continue training",
    )
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument(
        "--original_model", default=None, type=str, help="path to saved model"
    )
    parser.add_argument(
        "--load_weight",
        action="store_true",
        default=True,
        help="if true, load the incdsi checkpoint, else train from scratch",
    )
    parser.add_argument(
        "--load_weight_original",
        action="store_true",
        default=True,
        help="if true, load the incdsi checkpoint, else load the pretrained bert checkpoint",
    )
    parser.add_argument(
        "--load_weight_incdsi",
        action="store_true",
        default=False,
        help="load weight from IncDSI for val/test purpose only, the checkpoint path is hardcoded in get_model()",
    )
    parser.add_argument(
        "--freeze",
        action="store_true",
        default=True,
        help="for freezing the parameters of the base model",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        help="where the train/test/val data is located",
    )
    parser.add_argument(
        "--f", "-f", help="a dummy argument to fool ipython", default="1"
    )


def log_subparser(parser):
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--logging_step", default=50, type=int)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("wandb_tags", nargs="*", type=str)


def cl_subparser(parser):
    parser.add_argument("--num_tasks", type=int, default=2)
    parser.add_argument(
        "--train_mask",
        default=True,
        type=bool,
        help="if using the class mask at training",
    )
    parser.add_argument("--nb_classes", type=int, default=19748, help="num new docs")
    parser.add_argument(
        "--reinit_optimizer",
        type=bool,
        default=True,
        help="reinit optimizer (default: True)",
    )
    parser.add_argument(
        "--one_epoch_task1",
        type=bool,
        default=True,
        help="Train only 1 epoch for the first task (due to initialization)",
    )
    parser.add_argument("--filter_num", type=int, default=-1, help="num new docs")


def opt_subparser(parser):
    parser.add_argument(
        "--opt",
        default="adam",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adam")',
    )
    parser.add_argument(
        "--opt_eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt_betas",
        default=(0.9, 0.999),
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: (0.9, 0.999), use opt default)",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=1.0,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="M", help="SGD momentum"
    )
    parser.add_argument("--weight_decay", type=float, default=0.0)


def lr_sched_subparser(parser):
    parser.add_argument(
        "--sched",
        default="constant",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "constant")',
    )
    parser.add_argument(
        "--unscale_lr",
        type=bool,
        default=True,
        help="scaling lr by batch size (default: True)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--total_steps",
        type=int,
    )
    parser.add_argument(
        "--lr_noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr_noise_pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr_noise_std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std_dev (default: 1.0)",
    )
    parser.add_argument(
        "--warmup_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e_5)",
    )
    parser.add_argument(
        "--decay_epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown_epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience_epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10)",
    )
    parser.add_argument(
        "--decay_rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )


def prompt_subparser(parser):
    parser.add_argument("--pool_size", default=10, type=int)
    parser.add_argument("--prompt_length", default=5, type=int)
    parser.add_argument("--top_k", default=5, type=int)
    parser.add_argument("--initializer", default="uniform", type=str)
    parser.add_argument("--prompt_key", default=True, type=bool)
    parser.add_argument("--prompt_key_init", default="uniform", type=str)
    parser.add_argument("--use_prompt_mask", default=False, type=bool)
    parser.add_argument("--batchwise_prompt", default=True, type=bool)
    parser.add_argument("--embedding_key", default="cls", type=str)
    parser.add_argument("--pull_constraint", default=True)
    parser.add_argument("--pull_constraint_coeff", default=0.1, type=float)
    parser.add_argument("--prompt_pool", action="store_true", default=True)
    parser.add_argument("--shared_prompt_pool", action="store_false", default=False)
    parser.add_argument("--shared_prompt_key", action="store_false", default=False)
    parser.add_argument("--diversify_prompt_freq", action="store_false", default=False)
    parser.add_argument("--freeze_topk", action="store_false", default=False)

    # Adding dual prompt parameters here
    parser.add_argument(
        "--use_g_prompt", default=True, type=bool, help="if using G-Prompt"
    )
    parser.add_argument(
        "--g_prompt_length", default=5, type=int, help="length of G-Prompt"
    )
    parser.add_argument(
        "--g_prompt_layer_idx",
        default=[0, 1],
        type=int,
        nargs="+",
        help="the layer index of the G-Prompt",
    )
    parser.add_argument(
        "--prefix_g_prompt",
        default=True,
        type=bool,
        help="if using the prefix tune for G-Prompt",
    )

    # E-Prompt parameters
    parser.add_argument(
        "--use_e_prompt", default=True, type=bool, help="if using the E-Prompt"
    )
    parser.add_argument(
        "--e_prompt_layer_idx",
        default=[2, 3, 4],
        type=int,
        nargs="+",
        help="the layer index of the E-Prompt",
    )
    parser.add_argument(
        "--prompt_allocation",
        default=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        type=int,
        nargs="+",
        help="the layer index of the E-Prompt",
    )
    parser.add_argument(
        "--prefix_e_prompt",
        default=True,
        type=bool,
        help="if using the prefix tune for E-Prompt",
    )

    parser.add_argument("--same_key_value", default=False, type=bool)
    # My own parameters
    parser.add_argument("--key_refresh", default=False, type=bool)
    parser.add_argument(
        "--key_refresh_strat", default="mean", type=str
    )  # another option is "copy", only be used if key_refresh is true
    parser.add_argument("--fasttext", default=False, type=bool)
