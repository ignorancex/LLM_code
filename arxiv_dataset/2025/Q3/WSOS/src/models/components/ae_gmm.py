import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture

from src.models.components import AE


class AEGMM(nn.Module):
    """This module is only used for testing and not traning"""

    def __init__(self, ae_model: AE, gmm_model: GaussianMixture):
        super(AEGMM, self).__init__()
        self.ae = ae_model
        self.gmm = gmm_model

    def forward(self, x):
        # Pass input through the AE
        encoded_features = self.ae.encoder(x)

        # Use the GMM for clustering or other tasks
        gmm_output = torch.tensor(self.gmm.predict(encoded_features.cpu().numpy())).to(x)  # Modify as needed

        return encoded_features, gmm_output

    def eval(self):
        super().eval()
        self.ae.eval()
