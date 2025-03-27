from setuptools import setup, find_packages

setup(
    name='cctoolkit',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'camb', # Uncomment if CAMB is necessary
    ],
    description='A package for Cluster Cosmology calculations.',
    author='Tiago Castro',
    author_email= 'tiagobscastro@gmail.com',
    url='https://github.com/TiagoBsCastro/CCToolkit',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
