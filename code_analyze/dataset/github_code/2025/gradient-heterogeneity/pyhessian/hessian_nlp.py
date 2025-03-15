# *
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
# *


import numpy as np
import torch

from .utils import (
    get_params_grad,
    group_product,
    hessian_vector_product,
    normalization,
    orthnormal,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class hessian_nlp:
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(
        self, model, criterion, data=None, dataloader=None, cuda=True, param_name=None
    ):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """

        # make sure we either pass a single batch or a dataloader
        assert (data is not None and dataloader is None) or (
            data is None and dataloader is not None
        )

        self.model = model.eval()  # make model is in evaluation model
        self.criterion = criterion

        if data is not None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        if cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.targets = self.data["labels"]
            if self.device == "cuda":
                self.inputs = {k: v.cuda() for k, v in self.data.items()}
                self.targets = self.targets.cuda()

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.

            outputs = self.model(**self.data)
            logits = outputs["logits"].to(device)
            loss = self.criterion(logits, self.targets)
            loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model
        self.param_name = param_name
        params, gradsH = get_params_grad(self.model, self.param_name)
        self.params = params
        self.gradsH = gradsH  # gradient used for Hessian computation

    def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1, get_maximum=False):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        get_maximum: if True, return the maximum eigenvalue
        """

        assert top_n >= 1 or get_maximum

        device = self.device

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        while computed_dim < top_n:
            eigenvalue = None
            v = [
                torch.randn(p.size()).to(device) for p in self.params
            ]  # generate random vector
            v = normalization(v)  # normalize the vector

            for i in range(maxIter):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if self.full_dataset:
                    raise NotImplementedError
                else:
                    Hv = hessian_vector_product(self.gradsH, self.params, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue is None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if (
                        abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6)
                        < tol
                    ):
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            if get_maximum:
                if eigenvalue > 0:
                    return eigenvalue, v
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1
        if get_maximum:
            maximum_index = np.argmax(eigenvalues)
            return eigenvalues[maximum_index], eigenvectors[maximum_index]
        return eigenvalues, eigenvectors
