from setuptools import setup, find_packages

setup(
    name="SOAR",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # List your project dependencies here
        # For example: "requests>=2.25.1"
    ],
    author="Julien Pourcel",
    author_email="j.pourcel31830@gmail.com",
    description="SOAR",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/flowersteam/SOAR",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)