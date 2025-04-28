import datetime
import itertools

import cupyx.scipy.sparse as cpsparse
import nvtx

# import eliot
import scipy.sparse as spsparse

# from filprofiler.api import profile
from qcheff.iswt import NPAD
from qcheff.operators import create, destroy, number, qcheffOperator

# eliot.to_file(open("iswt_check.log", "w"))


n_op = 100000
with nvtx.annotate("Create Operator", color="orange"):
    base_mat = destroy(n_op) + create(n_op) + number(n_op)

with nvtx.annotate("Create qcheffOperator", color="pink"):
    test_op = qcheffOperator(cpsparse.csr_matrix(spsparse.csr_array(base_mat)))
# test_op = qcheffOperator(spsparse.csr_array(base_mat))

with nvtx.annotate("Initialize NPAD", color="blue"):
    testSWT = NPAD(test_op, copy=False)


with nvtx.annotate("Getting Largest Couplings", color="yellow"):
    cpls = list(itertools.islice(testSWT.largest_couplings(20), 20))

with nvtx.annotate("NPAD Givens Rotation", color="lightblue"):
    givens_rot_mat = testSWT.givens_rotation_matrix(0, 1)

with nvtx.annotate("Unitary Transformation with saved Matrix", color="purple"):
    testSWT.unitary_transformation(givens_rot_mat)

with nvtx.annotate("Unitary Transformation without saved Matrix", color="green"):
    testSWT.unitary_transformation(U=testSWT.givens_rotation_matrix(0, 1))

with nvtx.annotate("Eliminate Coupling", color="green"):
    testSWT.eliminate_coupling(0, 1)

with nvtx.annotate("Eliminate 10 Couplings", color="red"):
    testSWT.eliminate_couplings(
        (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
        )
    )
