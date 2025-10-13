from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Ensure Cython is available before building
try:
    import Cython
except ImportError:
    raise RuntimeError("Cython is required but not installed. Run `pip install cython` first.")

extensions = [
    Extension("vat_clustering.vat_cython", ["vat_clustering/vat_cython.pyx"], include_dirs=[np.get_include()])
]

setup(
    name="vat_clustering",
    version="1.1.0",
    packages=["vat_clustering"],
    ext_modules=cythonize(extensions),
    install_requires=["numpy", "scipy", "matplotlib", "numba", "cython"],
    setup_requires=["cython"],  # Ensures Cython is available during setup
    description="VAT Algorithm with Python-C binding using Cython.",
    author="MSR AVINASH",
    license="Apache-2.0",
)
