# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    name="treqa",
    version="0.0.1",
    author="Patrick Fernandes, Sweta Agrawal, Emmanouil Zaranis",
    author_email="pfernand@cs.cmu.edu",
    url="https://github.com/deep-spin/treqa/",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.7",
    setup_requires=[],
    install_requires=[
        'vllm==v0.6.2',
        'litellm',
        'sacrebleu',
        'spacy',
        'Levenshtein',
        'bert-score',
        'keybert',
    ],
    entry_points={
        'console_scripts': [
            'treqa-generate=treqa.generate_qa:main',
            'treqa-evaluate=treqa.evaluate:main',
        ],
    },
)