from setuptools import setup, Extension
from Cython.Build import cythonize
from glob import glob
import os

# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
import distutils.sysconfig
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

#includes=['']
#libs=[path for path in os.environ['LD_LIBRARY_PATH'].split(':') if path]
#-------------HqEvo Module------------------
fileLBT = [	'cython/HqEvo.pyx', 
			'src/matrix_elements.cpp', 
			'src/utility.cpp', 
			'src/Xsection.cpp',
			'src/sample_methods.cpp',
			'src/rates.cpp']
fileLGV = [	'cython/HqLGV.pyx', 
			'src/Langevin.cpp',
			'src/matrix_elements.cpp',
			'src/qhat_matrix_elements.cpp', 
			'src/utility.cpp', 
			'src/qhat_Xsection.cpp',
			'src/qhat.cpp',
			'src/TLorentz.cpp']
modules = [
        Extension('HqEvo', 
        		 sources=fileLBT, 
        		 language="c++", 
#        		 include_dirs=includes,
        		 #library_dirs=libs,
        		 extra_compile_args=["-std=c++11", '-march=native', '-fPIC'],
        		 libraries=["m", "gsl", "gslcblas", "boost_filesystem", "hdf5", "hdf5_cpp"]),
        Extension('HqLGV', 
        		 sources=fileLGV, 
        		 language="c++", 
#        		 include_dirs=includes,
        		 #library_dirs=libs,
        		 extra_compile_args=["-std=c++11", '-march=native', '-fPIC'],
        		 libraries=["m", "gsl", "gslcblas", "boost_filesystem", "hdf5", "hdf5_cpp"]),
]


setup(
        ext_modules=cythonize(modules),
       # include_dirs=includes
)
