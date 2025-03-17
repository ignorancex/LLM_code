#!/usr/bin/env python

from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

extensions = Extension('clara.pylpsolve', ['clara/pylpsolve.pyx'],
                  libraries=['lpsolve55'],
                  library_dirs=['libs/lp_solve_dev', 'libs/lp_solve_5.5/lpsolve55/bin/ux64'], 
                  include_dirs=[
                        'libs/lp_solve_dev',
                        'libs/lp_solve_5.5/lpsolve55/bin/ux64'
                  ])

setup(name='clara',
      version='1.0',
      packages=['clara'],
      ext_modules = cythonize(extensions),
      install_requires=['pycparser', 'zss'],
      scripts=['bin/clara']
     )