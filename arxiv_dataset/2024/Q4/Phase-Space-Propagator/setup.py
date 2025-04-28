from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='psprop',
    version='0.1.1',
    description='Partially coherent wave propagation through phase space',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Jake Rogers',
    author_email='jakejohnrogers@gmail.com',
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy'
    ],
)