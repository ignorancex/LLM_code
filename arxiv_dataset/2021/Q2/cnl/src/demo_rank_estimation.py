p_dim = 100
dim = 50
min_singular = 0.3
true_rank = 10
import numpy as np
a_true = np.random.uniform(min_singular, 1, dim)
for i in range(dim-true_rank):
    a_true[i] = 0
from matrix_util import *
D = rectangular_diag(a_true, p_dim=p_dim, dim=dim)
from random_matrices import *
U = haar_unitary(p_dim)
V = haar_unitary(dim)
A_true = U @ D @ V ; #random rotation
sigma_true = 0.1
X = A_true + sigma_true*Ginibre(p_dim, dim) ### sample matrix
from psvd import *
rank, a, sigma = rank_estimation(X) ### estimated rank and parameters
print("estimaed_rank =", rank, "true_rank=", true_rank) ### compare with the true_rank !
