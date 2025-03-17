"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"A Preconditioned Interior Point Method for Support Vector Machines Using an
ANOVA-Decomposition and NFFT-Based Matrix–Vector Products"
by T. Wagner, John W. Pearson, M. Stoll (2023)

Init file for nfftsvmipm module.
""" 

from .data_SVMipm import susy, higgs, cod_rna
from .data_preprocessing import under_sample, z_score_normalization, data_preprocess
from .hessian_ipm import hessian_ipm_pd
from .kernel_matvec import kernel_matvec
from .precond import pivoted_chol_rp, SMW_prec, get_diag, get_row
from .svm_ipm import evaluate_F_Newton, svm_ipm_pd_line_search
from .svm_predict_fastsum import svm_predict_fastsum
