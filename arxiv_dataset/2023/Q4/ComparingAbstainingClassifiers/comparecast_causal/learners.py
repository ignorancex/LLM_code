"""
Base classifiers & regressors for nuisance function estimation

Mostly a repackaging of certain sklearn & mlens modules.
"""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import log_loss, mean_squared_error

from mlens.ensemble import SuperLearner


def get_learner(task="classification", model="superlearner", **kwargs):
    """Get a base classifier or regressor.

    :param task: classification (default) or regression.
    :param model: linear, knn, rf, svm, sl (default), or mlp.
    :return: either a `sklearn` estimator or a `mlens` super learner instance.
    """
    task, model = task.lower(), model.lower()
    assert model in [
        "linear", "knn", "rf", "svm", "sl", "mlp",
        "logistic", "randomforest", "kernelsvm", "superlearner",  # alternative names
    ], (
        f"supported model options are 'linear', 'knn', 'rf', 'svm', 'sl', and 'mlp'; "
        f"got {model}"
    )

    if task in ["c", "clf", "classification"]:
        if model in ["linear", "logistic"]:
            return make_linear_classifier(**kwargs)
        elif model in ["knn"]:
            return make_knn_classifier(**kwargs)
        elif model in ["rf", "randomforest"]:
            return make_rf_classifier(**kwargs)
        elif model in ["svm", "kernelsvm"]:
            return make_svm_classifier(**kwargs)
        elif model in ["sl", "superlearner"]:
            return make_sl_classifier(**kwargs)
        elif model == "mlp":
            return make_mlp_classifier(**kwargs)
    elif task in ["r", "reg", "regression"]:
        if model in ["linear", "logistic"]:
            return make_linear_regressor(**kwargs)
        elif model in ["knn"]:
            return make_knn_regressor(**kwargs)
        elif model in ["rf", "randomforest"]:
            return make_rf_regressor(**kwargs)
        elif model in ["svm", "kernelsvm"]:
            return make_svm_regressor(**kwargs)
        elif model in ["sl", "superlearner"]:
            return make_sl_regressor(**kwargs)
        elif model == "mlp":
            return make_mlp_regressor(**kwargs)
    else:
        raise ValueError("input task must be 'classification' or 'regression'; "
                         f"got {task}")


def make_linear_classifier(scaler=StandardScaler(), **kwargs):
    """L2-regularized logistic regression."""
    steps = [scaler, LogisticRegression(**kwargs)] if scaler is not None else [LogisticRegression(**kwargs)]
    return make_pipeline(*steps)


def make_linear_regressor(scaler=StandardScaler(), **kwargs):
    """L2-regularized linear regression."""
    steps = [scaler, Ridge(**kwargs)] if scaler is not None else [Ridge(**kwargs)]
    return make_pipeline(*steps)


def make_knn_classifier(scaler=StandardScaler(), **kwargs):
    """kNN classification."""
    steps = [scaler, KNeighborsClassifier(**kwargs)] if scaler is not None else [KNeighborsClassifier(**kwargs)]
    return make_pipeline(*steps)


def make_knn_regressor(scaler=StandardScaler(), **kwargs):
    """kNN regression."""
    steps = [scaler, KNeighborsRegressor(**kwargs)] if scaler is not None else [KNeighborsRegressor(**kwargs)]
    return make_pipeline(*steps)


def make_rf_classifier(**kwargs):
    """Random forest classification."""
    return RandomForestClassifier(**kwargs)


def make_rf_regressor(**kwargs):
    """Random forest regression."""
    return RandomForestRegressor(**kwargs)


def make_svm_classifier(**kwargs):
    """Support vector classification with the Gaussian RBF kernel and
    probability estimation via Platt scaling."""
    return SVC(probability=True, **kwargs)


def make_svm_regressor(**kwargs):
    """Support vector regression with the Gaussian RBF kernel."""
    return SVR(**kwargs)


def make_sl_classifier(base_learners=None, scorer=log_loss, **kwargs):
    """A SuperLearner classifier with logistic regression, kNN, random forests."""
    sl_clf = SuperLearner(scorer=scorer, **kwargs)
    base_learners = ["Logistic", "kNN", "SVM", "RandomForest"] if base_learners is None else base_learners
    base_learners = [get_learner("classification", base_learner) for base_learner in base_learners]
    # estimators = [
    #     LogisticRegression(),  # L2 penalty
    #     KNeighborsClassifier(),
    #     RandomForestClassifier(),
    #     SVC(probability=True),
    # ]
    sl_clf.add(base_learners, proba=True)
    sl_clf.add_meta(LogisticRegression(), proba=True)
    return sl_clf


def make_sl_regressor(base_learners=None, scorer=mean_squared_error, **kwargs):
    """A SuperLearner regressor with ridge regression, kNN, SVM, and random forests."""
    sl_reg = SuperLearner(scorer=scorer, **kwargs)
    base_learners = ["Logistic", "kNN", "SVM", "RandomForest"] if base_learners is None else base_learners
    base_learners = [get_learner("regression", base_learner) for base_learner in base_learners]
    sl_reg.add(base_learners)
    sl_reg.add_meta(LinearRegression())
    return sl_reg


def make_mlp_classifier(
        config="wide",
        scaler=StandardScaler(),
        solver="adam",
        l2_weight_decay=1e-4,
        lr_schedule="invscaling",
        lr_init=0.001,
        batch_size="auto",
        max_iter=10000,
        **kwargs,
):
    """A feedforward neural network classifier. (Warning: this is a baseline scikit-learn version.)"""
    hidden_layers = {
        "wide": (128, ),
        "deep2": (64, 32),
        "deep3": (32, 32, 16),
    }[config]
    mlp = MLPClassifier(solver=solver, hidden_layer_sizes=hidden_layers,
                        alpha=l2_weight_decay, learning_rate=lr_schedule, learning_rate_init=lr_init,
                        batch_size=batch_size, max_iter=max_iter, **kwargs)
    steps = [scaler, mlp] if scaler is not None else [mlp]
    return make_pipeline(*steps)


def make_mlp_regressor(
        config="wide",
        scaler=StandardScaler(),
        solver="adam",
        l2_weight_decay=1e-4,
        lr_schedule="invscaling",
        lr_init=0.001,
        batch_size="auto",
        max_iter=10000,
        **kwargs,
):
    """A feedforward neural network regressor. (Warning: this is a baseline scikit-learn version.)"""
    hidden_layers = {
        "wide": (128, ),
        "deep2": (64, 32),
        "deep3": (32, 32, 16),
    }[config]
    mlp = MLPRegressor(solver=solver, hidden_layer_sizes=hidden_layers,
                       alpha=l2_weight_decay, learning_rate=lr_schedule, learning_rate_init=lr_init,
                       batch_size=batch_size, max_iter=max_iter, **kwargs)
    steps = [scaler, mlp] if scaler is not None else [mlp]
    return make_pipeline(*steps)
