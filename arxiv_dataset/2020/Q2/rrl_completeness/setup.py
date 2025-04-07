import os
import sys
import re

try:
  from setuptools import setup
except ImportError:
  from distutils.core import setup


setup(
    name='rrl_completeness',
    version='1.0.0',
    author='C. Mateu',
    author_email='cmateu@fisica.edu.uy',
    packages=['completeness_utils'],
    package_data={'data':['*.csv'],
                  'maps':['*.csv','*.png'],
                  'examples':['map_computing_examples.ipynb'],
                  'examples':['map_plotting_examples.ipynb']},
#    scripts=['bin/'],
    url='https://github.com/cmateu/rrl_completeness',
    license='LICENSE',
    description='Utils to plot and compute completeness maps for RRL catalogues',
    long_description=open('README.md').read(),
    install_requires=[
      "numpy",
      "scipy",
      "astropy",
      "healpy"
    ],
)

