#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

__name__ = 'CorrMatrix_ABC'

release_info = {}
infopath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                           __name__, 'info.py'))
with open(infopath) as open_file:
    exec(open_file.read(), release_info)

setup(
    name = __name__,
    author = release_info['__author__'],
    author_email = release_info['__email__'],
    version = release_info['__version__'],
    url = 'https://github.com/emilleishida/CorrMatrix_ABC',
    packages = find_packages(),
    install_requires = release_info['__requires__'],
    description = release_info['__description__'],
    scripts = ['scripts/cov_matrix_definition/cm_likelihood.py']
)
