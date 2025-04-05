#!/usr/bin/env python
# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="faradaydreams",
    version="0.0.3",
    author="Luke Pratley",
    author_email="luke.pratley@gmail.com",
    description="Faraday Rotation Recovery using Convex Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Luke-Pratley/Faraday-Dreams",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=['numpy', 'scipy', 'PyWavelets', 'optimusprimal', 'pynufft'])
