"""
Setup script
"""

from setuptools import find_packages, setup

with open("requirements.txt", encoding="utf-8") as f:
    basic_requirements = [line.rstrip() for line in f]

with open("requirements-torch.txt", encoding="utf-8") as f:
    torch_requirements = [line.rstrip() for line in f]

with open("requirements-engagement.txt", encoding="utf-8") as f:
    engagement_requirements = [line.rstrip() for line in f]

setup(
    name="emotiefflib",
    version="1.0",
    license="Apache-2.0",
    author="Andrey Savchenko, Egor Churaev",
    author_email="andrey.v.savchenko@gmail.com, egor.churaev@gmail.com",
    packages=find_packages("."),
    download_url="https://github.com/av-savchenko/EmotiEffLib/archive/v1.0.tar.gz",
    url="https://github.com/av-savchenko/EmotiEffLib",
    description="EmotiEffLib Python Library for Facial Emotion and Engagement Recognition",
    keywords=[
        "face expression recognition",
        "emotion analysis",
        "facial expressions",
        "engagement detection",
    ],
    install_requires=basic_requirements,
    extras_require={
        "torch": torch_requirements,
        "engagement": engagement_requirements,
        "all": torch_requirements + engagement_requirements,
    },
)
