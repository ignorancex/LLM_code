import faiss
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader

from ExpToolKit.models import BaseAE

from .const import RETRIEVER_CONST
from .utils import get_x


class Retriever:
    def __init__(
        self, retriever_type: str, model: BaseAE | None = None, **kwargs
    ) -> None:
        """
        __init__ function of Retriever class.

        Parameters
        ----------
        retriever_type : str
            The type of retriever to be used.
            Available options are:
                RETRIEVER_CONST.NEARSET_NEIGHBOR_RETRIEVER
                RETRIEVER_CONST.MAXIMUM_INNER_PRODUCT_RETRIEVER
        model : BaseAE | None
            The model to be used for feature extraction.
            If None, the raw data is used.
        **kwargs
            Additional keyword arguments for the retriever.
        """
        self.model = model
        if retriever_type == RETRIEVER_CONST.NEARSET_NEIGHBOR_RETRIEVER:
            self.retriever = faiss.IndexFlatL2
        elif retriever_type == RETRIEVER_CONST.MAXIMUM_INNER_PRODUCT_RETRIEVER:
            self.retriever = faiss.IndexFlatIP
        self.kwargs = kwargs

    def evaluate(
        self, dataloader: DataLoader, top_K: int, retrieval_N: int, label_num: int = 100
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Evaluate the retriever.

        Parameters
        ----------
        dataloader : DataLoader
            The DataLoader of the dataset to be evaluated.
        top_K : int
            The number of nearest neighbors to be retrieved.
        retrieval_N : int
            The number of nearest neighbors in the latent space to be retrieved.
        label_num : int
            The number of labels to be considered.

        Returns
        -------
        recall : float
            The average recall of the retriever.
        distance_matrix_x : np.ndarray
            The distance matrix of the data points in the original space.
        distance_matrix_x_latent : np.ndarray
            The distance matrix of the data points in the latent space.
        """
        x, x_latent = get_x(self.model, dataloader, label_num)

        index_x = self.retriever(x.shape[1])
        index_x_latent = self.retriever(x_latent.shape[1])

        index_x.add(x)
        index_x_latent.add(x_latent)

        recall_list = []
        for i in range(x.shape[0]):
            D, I = index_x.search(x[i : i + 1], top_K)
            D_latent, I_latent = index_x_latent.search(x_latent[i : i + 1], retrieval_N)
            intersect = np.intersect1d(I[0], I_latent[0])
            recall_list.append(intersect.shape[0] / top_K)
        recall = np.mean(recall_list)

        distance_matrix_x = np.zeros((x.shape[0], x.shape[0]))
        distance_matrix_x_latent = np.zeros((x_latent.shape[0], x_latent.shape[0]))
        for i in range(x.shape[0]):
            for j in range(i + 1, x.shape[0]):
                distance_matrix_x[i, j] = np.linalg.norm(x[i] - x[j])
                distance_matrix_x[j, i] = distance_matrix_x[i, j]
                distance_matrix_x_latent[i, j] = np.linalg.norm(
                    x_latent[i] - x_latent[j]
                )
                distance_matrix_x_latent[j, i] = distance_matrix_x_latent[i, j]

        return recall, distance_matrix_x, distance_matrix_x_latent
