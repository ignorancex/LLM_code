from pkg_resources import parse_version
from setuptools import setup, find_packages
import os
assert parse_version(setuptools.__version__)>=parse_version('36.2')

with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='DeepRUOT',                       
    version='0.1.0',                       
    author='Zhenyi Zhang',                      
    author_email='zhenyizhang@stu.pku.edu.cn', 
    description='DeepRUOT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/zhenyiizhang/DeepRUOT',  
    packages=find_packages(),              
    include_package_data=True,
    install_requires=[
        "torch>=1.11.0",
        "matplotlib",
        "numpy<2",
        "torchdiffeq",
        "torchsde",
        "scipy",
        "scikit-learn",
        "pot",
        "phate",
        "pyyaml",
        "tqdm",
        "seaborn>=0.12.2",
        "pandas",
        "ipywidgets",
        "scanpy",
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',      
        'Intended Audience :: Developers',
        'Natural Language :: Chinese',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  
    entry_points={
    },
)