#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# simplicial_test -- a python module to realize simplicial degree-size sequences
#
# Copyright (C) 2020-2021 Tzu-Chi Yen <tzuchi.yen@colorado.edu>
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
""" Simplicial-test realizes a simplicial complex from a prescribed degree-size sequence (when feasible).

Simplicial-test provides a deterministic, backtracking-based search algorithm for the simplicial complex realization problem.

This is the software repository behind the paper:

- Construction of Simplicial Complexes with Prescribed Degree-Size Sequences, Tzu-Chi Yen (2021).

  Read it on arXiv: https://arxiv.org/abs/2106.00185

👈 Please check the Project Links on the left for source code, documentation, discussion threads, etc.

All simplicial-test sdists and wheels distributed on PyPI are licensed under LGPL-3.0-or-later.
"""

from os.path import realpath, dirname, join
from setuptools import setup, find_packages
from simplicial_test import __version__

DOCLINES = (__doc__ or '').split("\n")

PROJECT_ROOT = dirname(realpath(__file__))

REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

install_reqs.append('setuptools')

setup(
    name='simplicial-test',
    version=__version__,
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    author='Tzu-Chi Yen',
    author_email='tzuchi.yen@colorado.edu',
    license='LGPLv3',
    package_dir={"": "."},
    packages=find_packages(where="."),
    package_data={
        'simplicial_test': [
            'COPYING',
            'COPYING.LESS',
            'README.rst',
            'requirements.txt',
        ]},
    include_package_data=True,
    install_requires=install_reqs,
    extras_require={
        'test': ['pytest'],
        'docs': ['sphinx']
    },
    url='https://github.com/junipertcy/simplicial-test',
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    test_suite='pytest',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Natural Language :: English',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
    ],
    keywords=[
        'simplicial-test',
        'simplicial complex',
        'degree-size sequence',
        'simplex',
        'topological data analysis',
        'topology',
        'recursive algorithm',
        'fixed-parameter tractable',
        'realizability',
    ],
    project_urls={
        'Bug Tracker': 'https://github.com/junipertcy/simplicial-test/issues',
        'Forum': 'https://github.com/junipertcy/simplicial-test/discussions',
        'Source Code': 'https://github.com/junipertcy/simplicial-test',
        'Documentation': 'https://docs.netscied.tw/simplicial-test/index.html',
        'Benchmark': 'https://docs.netscied.tw/simplicial-test/dataset/benchmark.html'
    },
)
