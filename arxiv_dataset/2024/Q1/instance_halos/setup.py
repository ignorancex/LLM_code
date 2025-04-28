from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "instance_halos.cython_funcs",
        sources=["instance_halos/cython_funcs.pyx"],
        include_dirs=[np.get_include()]  # Include NumPy's header files
    )
]

setup(
    name="instance_halos",
    author="Daniel Lopez",
    author_email="daniellopezcano13@gmail.com",
    version="0.1.0",
    ext_modules = cythonize(extensions),
    packages=["instance_halos"],
)