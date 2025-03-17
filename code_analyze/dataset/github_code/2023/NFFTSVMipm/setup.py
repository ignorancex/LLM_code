"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"A Preconditioned Interior Point Method for Support Vector Machines Using an
ANOVA-Decomposition and NFFT-Based Matrix–Vector Products"
by T. Wagner, John W. Pearson, M. Stoll (2023)

Setup file for the NFFTSVMipm project.
""" 

from distutils.core import setup

# run setup
setup(name = 'nfftsvmipm',
    version = '1.0',
    description = 'NFFT-Accelerated Preconditioned Interior Point Method for Support Vector Machines',
    author = 'Theresa Wagner',
    author_email = 'theresa.wagner@math.tu-chemnitz.de',
    url = 'https://github.com/wagnertheresa/NFFTSVMipm',
    packages = ['nfftsvmipm'],
    py_modules = ['nfftsvmipm.class_NFFTSVMipm', 'nfftsvmipm.func_prec_timings'])