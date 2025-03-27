"""Setup script for the Triplet structure C++ extensions.

See https://docs.python.org/3/extending/building.html
"""
from distutils.core import setup, Extension
from glob import glob
import pybind11

TC_CPP_MODULE = Extension("ts_cpp",
                          include_dirs=[pybind11.get_include()],
                          extra_compile_args=["-O3", "-std=c++11"],
                          sources=glob("*.cc"))

setup(name="ts_cpp",
      version="1.0",
      description="Optimized triplet structure extension",
      author="Matthew A. Sotoudeh",
      author_email="masotoudeh@ucdavis.edu",
      ext_modules=[TC_CPP_MODULE])
