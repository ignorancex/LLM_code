from sklearn.covariance import EmpiricalCovariance
import numpy as np
import torch


def mahalanobis(training_features, training_y, eval_features, num_class):
    training_features = training_features.cpu().detach().numpy()
    training_y = np.array(training_y)
    print("Fitting Mahalanobis model")
    train_means = []
    train_feat_centered = []
    for i in range(num_class):
        assert np.sum(training_y == i) != 0
        fs = training_features[training_y == i, :]
        _m = fs.mean(axis=0)
        train_means.append(_m)
        train_feat_centered.extend(fs - _m)
    print("Obtaining Covariance matrix")
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(np.array(train_feat_centered).astype(np.float64))
    print("Covariance Matrix Obtained")

    mean = torch.from_numpy(np.array(train_means)).cuda().float()
    prec = torch.from_numpy(ec.precision_).cuda().float()
    result = []
    eval_features = eval_features.cuda()
    for i in range(0, eval_features.shape[0]):
        f = eval_features[i, :]
        result.append(
            (((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item()
        )
    return -np.array(result)
