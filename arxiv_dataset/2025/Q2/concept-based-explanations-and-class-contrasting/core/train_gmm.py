from sklearn.mixture import GaussianMixture
import torch
import numpy as np

from core.separate_features import load_activation_vecs




def train_gmm(features=[1, 32]):
    all_data = []
    for feature in features:
        data = load_activation_vecs(feature)

        all_data.append(data)

    training_data = torch.cat(all_data, dim=0).numpy()
    return train_gmm_from_data(training_data)


def train_gmm_from_data(training_data, n_components=6):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    #print("fit model...")
    gmm.fit(training_data)

    density_score = gmm.score_samples(training_data)
    print("mean density score: {}".format(np.mean(density_score)))
    print("max density score: {}".format(np.max(density_score)))

    return gmm


# Calculate likelihood for a new sample
def gmm_density_score(x, gmm):
    # Convert tensor to numpy for GMM scoring
    x_np = x.numpy()
    # Get the log likelihood
    log_likelihood = gmm.score_samples(x_np)  # Higher values mean the sample is closer to training data
    return torch.tensor(log_likelihood)




if __name__ == "__main__":

    # Assume `train_features` is a tensor of features from the training data

    activation_vecs_bcc = load_activation_vecs(1).cpu()
    activation_vecs_trichoblastoma = load_activation_vecs(32).cpu()

    activation_vecs_histiozytoma = load_activation_vecs(5).cpu()


    train_features = torch.cat([activation_vecs_bcc, activation_vecs_trichoblastoma], dim=0)


    # Convert to numpy for GMM fitting
    train_features_np = train_features.numpy()

    # Fit a Gaussian Mixture Model
    gmm = GaussianMixture(n_components=5, covariance_type='full')
    print("fit model...")
    gmm.fit(train_features_np)
    print("finished fit!")

    # Calculate likelihood for a new sample
    def gmm_density_score_(x):
        # Convert tensor to numpy for GMM scoring
        x_np = x.numpy()
        # Get the log likelihood
        log_likelihood = gmm.score_samples(x_np)  # Higher values mean the sample is closer to training data
        return torch.tensor(log_likelihood)

    # Example input tensor
    new_data_point = torch.randn(5, 256)  # Batch of 5 new samples
    density_scores = gmm_density_score_(new_data_point)
    print(torch.mean(density_scores))

    density_scores = gmm_density_score_(activation_vecs_bcc)
    print(torch.mean(density_scores))

    density_scores = gmm_density_score_(activation_vecs_histiozytoma)
    print(torch.mean(density_scores))


