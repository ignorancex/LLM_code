from setuptools import setup, find_packages

setup(
    name='SNOB',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'astropy>=6.0.1',
        'matplotlib>=3.8.4',
        'numpy>=1.24.2',
        'pandas>=2.0.1',
        'pyvo>=1.5.1',
        'scipy>=1.14.1',
        'seaborn>=0.13.2',
        'setuptools>=65.5.0',
        'openpyxl>=3.1.5'
    ],
    # Additional metadata about your package
    author='Viktor Mikalsen',
    author_email='viktor.mikalsen@gmail.com',
    description='A brief description of your project',
    # Any other metadata
)
