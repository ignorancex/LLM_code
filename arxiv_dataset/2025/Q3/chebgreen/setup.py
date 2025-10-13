#!/usr/bin/env python

import os
import sys

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
sys.path.append(here)

# get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

install_requires = [
    "scipy==1.14.1",
    "numpy==2.1.0",
    "matplotlib==3.9.2",
    "torch==2.4.0",
]

setup(
    # metadata
    name="chebgreen",
    description="Python library for learning and interpolating Green's function for 1-Dimensional problems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hsharsh/chebgreen",
    author="Harshwardhan Praveen",
    author_email="[praveenharsh01@gmail.com]",
    version="0.1.1",
    python_requires=">=3.12",
    install_requires=install_requires,
    packages=find_packages(exclude=["datasets"]),
    package_data={
        # include MATLAB scripts in the package
        "chebgreen": [
            "scripts/examples/*.m",
            "scripts/*.m",
            "settings.ini",
        ]
    },
    zip_safe=False,
)
