from configargparse import Parser


def add_input_args(parser):

    parser.add_argument(
        "--base_model",
        type=str,
        default="tabpfn",
        help="Primary classifier model to use for predictions",
        choices=["tabpfn", "tabicl", "gbm", "tabdpt"],
        )

    parser.add_argument(
        "--base_model_uncertainty",
        type=str,
        default="tabpfn",
        help="Model to use for uncertainty estimation in reduction methods (lr: Logistic Regression, tabpfn: TabPFN Classifier)",
        choices=["lr", "tabpfn", "tabicl"],
    )

    parser.add_argument(
        "--selection_method",
        type=str,
        default="vanilla",
        choices=["uncertain", "group_balanced", "correlation_remover", "vanilla", "class_group_balanced"],
        help="Method for selecting demonstration examples: 'uncertain' selects examples with uncertain predictions, 'certain' selects examples with confident predictions, 'balanced' ensures equal representation across groups, 'correlation_remover' reduces correlation with sensitive attributes, 'vanilla' selects examples randomly",
    )

    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="german_age",
        help="Dataset to use for experiments. Available options include: adult_income (ACS income prediction), celeba_attract (CelebA attractiveness), diabetes_race (Diabetes readmission), and others",
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Custom name for the experiment (used for logging and saving results)",
    )

    parser.add_argument(
        "--cp_alpha",
        type=float,
        default=0.05,
        help="Significance level for conformal prediction (controls the coverage in uncertainty estimation)",
    )

    parser.add_argument(
        "--train_size",
        type=int,
        default=1000,
        help="Number of training examples to use (subsample size from the full dataset) -1 means all the training data is used if possible",
    )

    parser.add_argument(
        "--cr_sanitization_strategy",
        type=int,
        default=1,
        help="Sanitization strategy of correlation remover. `1` apply transform to both training and testing; `2` apply transform only to the training set.",
        choices=[1, 2],
    )

    return parser