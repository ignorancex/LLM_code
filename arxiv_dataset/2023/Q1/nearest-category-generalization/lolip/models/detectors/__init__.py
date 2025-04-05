from .drocc_trainer import DROCCTrainer
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from .extended_iso_forest import ExtendedIsoForest


def get_detector(name, n_jobs=12):
    if name == "IsolationForest":
        return IsolationForest(n_jobs=n_jobs, random_state=0)
    elif name == "IsolationForestv2":
        return IsolationForest(n_estimators=200, n_jobs=n_jobs, random_state=0)
    elif name == "ExtIsoForest":
        return ExtendedIsoForest(ntrees=200, sample_size=256)
    elif name == "LocalOutlierFactor":
        return LocalOutlierFactor(n_jobs=n_jobs, novelty=True)
    elif name == "linearOneClassSVM":
        return OneClassSVM('linear', max_iter=100)
    else:
        raise ValueError(f"[get_detector] no such name: {name}")
