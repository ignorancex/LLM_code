"""
Copyright (c) 2024-present Naver Cloud Corp.
This source code is based on code from the Segment Anything Model (SAM)
(https://github.com/facebookresearch/segment-anything).

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from setuptools import find_packages, setup

setup(
    name="zim_anything",
    version="0.1",
    install_requires=["onnx", "onnxruntime-gpu"],
    packages=find_packages(exclude="notebooks"),
    extras_require={
        "all": ["matplotlib", "pycocotools", "opencv-python"],
        "dev": ["flake8", "isort", "black", "mypy"],
    },
)