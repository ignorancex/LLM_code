import os
from setuptools import setup, find_packages

# Read long description safely
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read() if os.path.exists("README.md") else ""

setup(
    name="aptx_activation",
    version="0.0.5",
    author="Ravin Kumar",
    author_email="mr.ravin_kumar@hotmail.com",
    description="A PyTorch implementation of the APTx activation function.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mr-ravin/aptx_activation",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research"
    ],
    keywords=[
        "APTx", "activation function", "deep learning", "pytorch", "neural networks",
        "machine learning", "AI", "ML", "DL", "torch", "Pytorch", "MISH", "SWISH", "ReLU"
    ],
    license="MIT",
    include_package_data=True
)
