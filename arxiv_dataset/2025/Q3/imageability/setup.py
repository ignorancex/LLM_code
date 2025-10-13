from setuptools import find_namespace_packages
from setuptools import setup

setup(
    name='imageability',
    version='0.1',
    url='https://github.com/Artificial-Memory-Lab/imageability.git',
    author='artificial-memory-lab',
    description=('Exploring embedding geometry and imageability.'),
    install_requires=[
        'numpy~=2.1.3',
        'matplotlib~=3.9.2',
        'scipy~=1.14.1',
        'tqdm~=4.67.0',
        'scikit-learn~=1.5.2',
        'sentence-transformers~=3.3.1',
        'faiss-cpu~=1.9.0',
        'nltk~=3.9.1'
    ],
    packages=find_namespace_packages(include=['imageability', 'imageability.*']),
    python_requires='>=3.9',
)
