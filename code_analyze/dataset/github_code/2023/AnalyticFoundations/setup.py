from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(name='gridutil',
      ext_modules=[CppExtension('gridutil', ['gridutil.cpp'])],
      cmdclass={'build_ext': BuildExtension})

setup(name='ifceutil',
      ext_modules=[CppExtension('ifceutil', ['ifceutil.cpp'])],
      cmdclass={'build_ext': BuildExtension})