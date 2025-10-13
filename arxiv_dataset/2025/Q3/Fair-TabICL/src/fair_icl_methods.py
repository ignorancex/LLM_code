from fairlearn.preprocessing import CorrelationRemover
from sklearn.model_selection import train_test_split
from mapie.classification import MapieClassifier
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from tabpfn import TabPFNClassifier


def correlation_remover(X, s, alpha=1):
    """
    Apply correlation removal to reduce the correlation between features and sensitive attributes.

    Args:
        X (numpy.ndarray): Feature matrix
        s (numpy.ndarray): Sensitive attribute vector
        cp_alpha (float, optional): Parameter controlling the strength of correlation removal. Defaults to 1.

    Returns:
        tuple: Transformed feature matrix and unchanged sensitive attribute vector
    """

    features_with_sensitive = np.hstack((np.expand_dims(s, axis=1), X))
    cr = CorrelationRemover(sensitive_feature_ids=[0], alpha=alpha)

    assert np.all(
        features_with_sensitive[:, 0] == s
    ), f"The sensitive attribute column is wrong"

    cr.fit(features_with_sensitive)

    transformed_features = cr.transform(features_with_sensitive)

    return transformed_features, cr

def class_group_balanced_sampling(X_train, y_train, s_train, train_size=-1):
    """
    Sample data to create a balanced dataset with equal representation from each group.

    Args:
        X_train (numpy.ndarray): Feature matrix
        y_train (numpy.ndarray): Target variable vector
        s_train (numpy.ndarray): Sensitive attribute vector
        train_size (int, optional): Desired size of the balanced dataset. If -1, uses maximum possible balanced size.
            Defaults to -1.

    Returns:
        numpy.ndarray: Indices of the balanced sample
    """
    # Get indices for each subgroup (y,s)
    subgroup_00_indices = np.where((y_train == 0) & (s_train == 0))[0]
    subgroup_01_indices = np.where((y_train == 0) & (s_train == 1))[0]
    subgroup_10_indices = np.where((y_train == 1) & (s_train == 0))[0]
    subgroup_11_indices = np.where((y_train == 1) & (s_train == 1))[0]

    target_size = train_size
    if train_size == -1:
        target_size = len(X_train)

    # Find size of smallest subgroup
    min_subgroup_size = min(
        len(subgroup_00_indices),
        len(subgroup_01_indices),
        len(subgroup_10_indices),
        len(subgroup_11_indices)
    )
    samples_per_subgroup = min(min_subgroup_size, target_size // 4)

    # Randomly sample equal number from each subgroup
    subgroup_00_samples = np.random.choice(
        subgroup_00_indices, size=samples_per_subgroup, replace=False
    )
    subgroup_01_samples = np.random.choice(
        subgroup_01_indices, size=samples_per_subgroup, replace=False
    )
    subgroup_10_samples = np.random.choice(
        subgroup_10_indices, size=samples_per_subgroup, replace=False
    )
    subgroup_11_samples = np.random.choice(
        subgroup_11_indices, size=samples_per_subgroup, replace=False
    )

    # Combine indices
    balanced_indices = np.concatenate([
        subgroup_00_samples,
        subgroup_01_samples,
        subgroup_10_samples,
        subgroup_11_samples
    ])
    return balanced_indices

def group_balanced_sampling(X_train, s_train, train_size=-1):
    """
    Sample data to create a balanced dataset with equal representation from each group.

    Args:
        X_train (numpy.ndarray): Feature matrix
        s_train (numpy.ndarray): Sensitive attribute vector
        train_size (int, optional): Desired size of the balanced dataset. If -1, uses maximum possible balanced size.
            Defaults to -1.

    Returns:
        numpy.ndarray: Indices of the balanced sample
    """
    # Get indices for each group
    group_0_indices = np.where(s_train == 0)[0]
    group_1_indices = np.where(s_train == 1)[0]
    target_size = train_size
    if train_size == -1:
        target_size = len(X_train)

    # Find size of smaller group
    min_group_size = min(len(group_0_indices), len(group_1_indices))
    samples_per_group = min(min_group_size, target_size // 2)

    # Randomly sample equal number from each group
    group_0_samples = np.random.choice(
        group_0_indices, size=samples_per_group, replace=False
    )
    group_1_samples = np.random.choice(
        group_1_indices, size=samples_per_group, replace=False
    )

    # Combine indices
    balanced_indices = np.concatenate([group_0_samples, group_1_samples])

    return balanced_indices


 

def uncertainty_based_sample_selection(
    base_classifier,
    X_train,
    data_with_sensitive_attrib,
    cp_alpha=0.05,
):
    """
    Identify certain and uncertain predictions using conformal prediction.

    Args:
        X_train (numpy.ndarray): Feature matrix for training data
        data_with_sensitive_attrib (tuple): Tuple containing (X, s) where X is features, and s is sensitive attributes
        cp_alpha (float, optional): Significance level for conformal prediction. Defaults to 0.05.
        base_model_uncertainty (str, optional): Base model to use for uncertainty estimation.
                                               Defaults to "tabpfn".

    Returns:
        tuple: Indices of certain and uncertain predictions
    """

    def get_certainty_set(prediction_scores):
        """
        Separate predictions into certain and uncertain sets based on prediction set size.

        Args:
            prediction_scores (numpy.ndarray): Prediction set scores

        Returns:
            tuple: Indices of certain and uncertain predictions
        """
        alpha_index = 0
        uncertain_indices = []
        certain_indices = []
        for i in range(len(prediction_scores)):
            prediction_set_size = prediction_scores[i].sum(axis=0) #[alpha_index]
            if prediction_set_size == 1:
                certain_indices.append(i)
            else:
                uncertain_indices.append(i)
        return certain_indices, uncertain_indices

    X_train_aux, s_train_aux = data_with_sensitive_attrib

    (
        X_train_split,
        X_validation,
        s_train_split,
        s_validation,
    ) = train_test_split(X_train_aux, s_train_aux, test_size=0.3)

    X_calib, X_conformal, s_calib, s_conformal = train_test_split(
        X_validation, s_validation, test_size=0.5
    )

    if isinstance(base_classifier, TabPFNClassifier) and len(X_train_split) >= 10000:
        # sub-sample the maximum of 10000 samples that TabPFN can handle
        print(
            "=============== SUBSAMPLING TO 10000 SAMPLES FOR TabPFN==================== "
        )
        sample_indices = np.random.choice(len(X_train_split), size=10000, replace=False)
        X_train_split = X_train_split[sample_indices]
        s_train_split = s_train_split[sample_indices]

    base_classifier = base_classifier.fit(X_train_split, s_train_split)

    calibrated_classifier = CalibratedClassifierCV(
        estimator=base_classifier, method="sigmoid", cv="prefit"
    )

    calibrated_classifier.fit(X_calib, s_calib)

    conformal_classifier = MapieClassifier(
        estimator=calibrated_classifier, method="lac", cv="prefit"
    )
    conformal_classifier.fit(X_conformal, s_conformal)
 
    _, prediction_set_scores = conformal_classifier.predict(X_train, alpha=cp_alpha)

    certain_indices, uncertain_indices = get_certainty_set(prediction_set_scores)

    return certain_indices, uncertain_indices
