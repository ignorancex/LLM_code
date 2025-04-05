#from distutils.core import setup
#from distutils.extension import Extension
from Cython.Distutils import build_ext
#from Cython.Build import cythonize
from setuptools import setup, Extension

import numpy as np

dirname = "cy_fde"

o_sourcefiles = ['spn.c', 'test_all.c', 'matrix_util.c', 'test_matrix_util.c']
sourcefiles = ["cy_fde/cy_spn.pyx"]
for s in o_sourcefiles:
    s = "{}/{}".format(dirname, s)
    sourcefiles.append(s)



#BLAS_PATH =  "/opt/OpenBLAS/include/"

ext_modules = [
Extension("cy_fde_spn",
sourcefiles,
include_dirs=[np.get_include()])]



setup(
    packages=["cy_fde_pack"],
    ext_modules = ext_modules,
    cmdclass = {'build_ext': build_ext},
)
