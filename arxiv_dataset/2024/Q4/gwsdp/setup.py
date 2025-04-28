from distutils.core import Extension, setup

import numpy
from Cython.Build import cythonize

extensions = [
    Extension(name="gwsdp.fast_cost_tensor", sources=["gwsdp/fast_cost_tensor.pyx"],
              include_dirs=[numpy.get_include()],
              extra_compile_args=['-g0'], # weird libelf error, need to do this to fix, check here https://github.com/cython/cython/issues/2865
              )
]
setup(
    name='gwsdp',
    ext_modules=cythonize(extensions)
)
