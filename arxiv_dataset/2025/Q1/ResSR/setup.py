from setuptools import setup, find_packages, Extension
import numpy as np
import os

NAME = "ResSR"
VERSION = "0.1"
DESCR = "ResSR is an efficient and modular residual-based method for super-resolving the lower-resolution bands of a multispectral image."
REQUIRES = ['numpy']
LICENSE = "BSD-3-Clause"

AUTHOR = 'ResSR development team'
EMAIL = "sullivanhe@ornl.gov"
PACKAGE_DIR = "ResSR"

setup(install_requires=REQUIRES,
      zip_safe=False,
      name=NAME,
      version=VERSION,
      description=DESCR,
      author=AUTHOR,
      author_email=EMAIL,
      license=LICENSE,
      packages=find_packages(include=['ResSR']),
      )

