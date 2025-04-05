from setuptools import setup


setup(name='ISCFB',
      install_requires=['libsvmdata>=0.2', 'numpy>=1.12', 'numba',
                        'scipy>=0.18.0', 'matplotlib>=2.0.0'],
      packages=['ISCFB'],
      )
