#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from Cython.Build import cythonize

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python >= 3.6 is required for fairseq.")


if sys.platform == "darwin":
    extra_compile_args = ["-stdlib=libc++", "-O3"]
else:
    extra_compile_args = ["-std=c++11", "-O3"]


class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy

        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs


extensions = [
    NumpyExtension(
        "data_utils_fast",
        sources=["data_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    install_requires=['numpy', 'cython'],
    python_requires='>=3.6',
    zip_safe=False
)
