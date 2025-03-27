#!/usr/bin/env python3

import os

from setuptools import find_packages, setup

# Assign root dir location for later use
root_dir = os.path.dirname(__file__)


def read_deps_from_file(filname):
    """Read in requirements file and return items as list of strings"""
    with open(os.path.join(root_dir, filname), "r") as fh:
        return [line.strip() for line in fh.readlines() if not line.startswith("#")]


# Read in the requirements from the requirements.txt file
install_requires = read_deps_from_file("requirements.txt")

setup(
    name="scalarize",
    description="Scalarize",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.8",
)
