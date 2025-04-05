from setuptools import setup, find_packages

import acsconv


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

setup(
    name='ACSConv',
    version=acsconv.__version__,
    url='https://github.com/M3DV/ACSConv',
    license='Apache-2.0 License',
    author='Jiancheng Yang and Xiaoyang Huang',
    author_email='jekyll4168@sjtu.edu.cn',
    python_requires=">=3.6.0",
    description='[IEEE JBHI] Reinventing 2D Convolutions for 3D Images',
    long_description=readme(),
    install_requires=[
        'fire', 
        'numpy',
        'matplotlib', 
        'pandas', 
        'tqdm', 
        'scikit-image', 
        'scikit-learn', 
        'scipy', 
        'tensorboardx',
        'torch',
        'torchvision'
        ],
    long_description_content_type="text/markdown",
    packages=find_packages(),
    zip_safe=True,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only"
    ]
)