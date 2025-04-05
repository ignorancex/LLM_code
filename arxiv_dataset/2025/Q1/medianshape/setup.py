from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst')) as f:
    long_description = f.read()

setup(name="medianshape",
    version='1.0',
    description='A simple linear program to solve Simplicial Median Shape Problem',
    long_description=long_description,
    keywords='Simplicial Median shape, flat norm, meshing',
    author='Altansuren Tumurbaatar',
    author_email='atumurbaatar@math.wsu.edu',
    url = 'https://bitbucket.org/altaa22/medianshape',
    license = 'GNU GPL',
    include_package_data=True,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'cplex'
        'cvxopt',
        'PyDistMesh',
        'matplotlib',
    ]

)
