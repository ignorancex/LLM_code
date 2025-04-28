from setuptools import setup, find_packages

setup(
    name='psps_py',
    version='0.1.0',
    author='Jiacheng Miao',
    author_email='jiacheng.miao@wisc.edu',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'statsmodels'
    ],
    description='A Python package for PoSt-Prediction Sumstats-based inference.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://arxiv.org/abs/2405.20039',  # Optional
)