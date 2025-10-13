from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension
from distutils.command.build import build
from setuptools.command.install import install
from setuptools.command.develop import develop
import os
import subprocess
import platform
# pylint: skip-file


setup(name='extensions',
    py_modules=['extensions'],
    install_requires=[
        'torch',
    ]
)
