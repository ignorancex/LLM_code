import numpy as np
from scipy.sparse import issparse
from sklearn.datasets import (
    load_iris,
    load_breast_cancer,
    load_wine,
    load_digits,
    fetch_california_housing,
    fetch_olivetti_faces,
    fetch_lfw_people,
    fetch_covtype,
)
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize


# -----------------------------------------------------------------------------
# Utility: dataset loader with padding and sample reduction
# -----------------------------------------------------------------------------
def safe_l2_normalize(X, eps=1e-8):
    """Safe row-wise L2 normalization with zero protection"""
    if issparse(X):
        norms = np.sqrt(np.array(X.power(2).sum(axis=1)))
        norms = np.maximum(norms, eps)
        return X.multiply(1 / norms)
    else:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)

    return X / norms


def load_dataset(name, max_qubits=13, max_samples=1000, random_state=42):
    """Load dataset with feature padding and sample size control

    Args:
        name: Dataset name
        max_qubits: Maximum number of qubits for feature dimension
        max_samples: Maximum number of samples to return
        random_state: Random seed for reproducibility

    Returns:
        X: Normalized and padded features
        y: Integer labels
        n_qubits: Number of qubits needed for features
        n_classes: Number of unique classes
    """
    # Handle special datasets first
    if name == "diabetes":  # Binary classification version
        from sklearn.datasets import fetch_openml

        X, y = fetch_openml("diabetes", version=7, return_X_y=True, as_frame=False)
        y = np.where(y == "No", 0, 1).astype(int)

    elif name == "california_housing":
        X, y = fetch_california_housing(return_X_y=True)
        # Bin into 5 classes for classification
        binner = KBinsDiscretizer(
            n_bins=5, encode="ordinal", strategy="quantile", random_state=random_state
        )
        y = binner.fit_transform(y.reshape(-1, 1)).ravel().astype(int)

    elif name == "20newsgroups":
        from sklearn.datasets import fetch_20newsgroups

        # First get labels to filter
        newsgroups = fetch_20newsgroups(
            subset="all",
            remove=("headers", "footers", "quotes"),
            shuffle=True,
            random_state=random_state,
        )
        y = newsgroups.target

        # Remove classes with < 5 samples for 5-fold CV
        unique, counts = np.unique(y, return_counts=True)
        keep_classes = unique[counts >= 5]
        keep_indices = np.where(np.isin(y, keep_classes))[0]

        # Now load with filtered indices
        newsgroups = fetch_20newsgroups(
            subset="all",
            remove=("headers", "footers", "quotes"),
            shuffle=True,
            random_state=random_state,
        )

        # Apply filtering
        X_text = [newsgroups.data[i] for i in keep_indices]
        y = newsgroups.target[keep_indices]

        # Reduce sample size if needed
        if max_samples and len(X_text) > max_samples:
            _, X_text, _, y = train_test_split(
                X_text, y, train_size=max_samples, stratify=y, random_state=random_state
            )

        # Vectorize text with limited features
        vectorizer = TfidfVectorizer(max_features=1024, stop_words="english")
        X = vectorizer.fit_transform(X_text)

    elif name == "ionosphere":
        from sklearn.datasets import fetch_openml

        X, y = fetch_openml("ionosphere", version=1, return_X_y=True, as_frame=False)
        y = np.where(y == "b", 0, 1).astype(int)

    elif name == "fashion-mnist":  # 9 qubits
        from sklearn.datasets import fetch_openml

        X, y = fetch_openml("Fashion-Mnist", version=1, return_X_y=True, as_frame=False)
        y = y.astype(int)

    elif name == "mnist":  # 9 qubits
        from sklearn.datasets import fetch_openml

        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        y = y.astype(int)

    elif name == "covtype":
        X, y = fetch_covtype(return_X_y=True)
        y = y - 1  # Convert to 0-6 range
        y = y.astype(int)

    elif name == "lfw_people":
        from sklearn.datasets import fetch_lfw_people

        # First get labels to filter
        lfw = fetch_lfw_people(
            min_faces_per_person=5, resize=0.4
        )  # Ensure min 5 samples per class
        X, y = lfw.data, lfw.target

    elif name == "random3":
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=1000, n_features=8, random_state=random_state
        )
        y = y.astype(int)

    elif name == "random5":
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=1000, n_features=32, random_state=random_state
        )
        y = y.astype(int)

    elif name == "random10":
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=1000, n_features=1024, random_state=random_state
        )
        y = y.astype(int)

    else:  # Standard sklearn datasets
        loader_map = {
            "iris": load_iris,
            "breast_cancer": load_breast_cancer,
            "wine": load_wine,
            "digits": load_digits,
            "olivetti_faces": fetch_olivetti_faces,
        }
        if name in loader_map:
            data = loader_map[name](return_X_y=True)
            X, y = data[0], data[1]
            y = y.astype(int)
        else:
            raise ValueError(f"Unknown dataset: {name}")

    # For non-text datasets, reduce sample size
    if name != "20newsgroups" and max_samples and X.shape[0] > max_samples:
        unique_classes, class_counts = np.unique(y, return_counts=True)
        n_classes = len(unique_classes)

        # Check if stratification is possible
        if n_classes > 1 and np.min(class_counts) >= 2:
            X, _, y, _ = train_test_split(
                X, y, train_size=max_samples, stratify=y, random_state=random_state
            )
        else:
            indices = np.random.choice(X.shape[0], max_samples, replace=False)
            X, y = X[indices], y[indices]

    # Feature normalization with safety
    X = safe_l2_normalize(X)
    # # Convert sparse to dense after normalization
    if issparse(X):
        X = X.toarray()

    # Feature dimension adjustment
    n_qubits = min(int(np.ceil(np.log2(X.shape[1]))), max_qubits)

    target_dim = 2**n_qubits

    if X.shape[1] < target_dim:
        print(f"dimension {X.shape[1]}, padding with zeros for {target_dim}")
        # Pad with zeros
        pad = np.zeros((X.shape[0], target_dim - X.shape[1]))
        X = np.hstack([X, pad])
    else:
        # Truncate to target dimension
        X = X[:, 0:target_dim]

    return X, y, n_qubits, len(np.unique(y))
