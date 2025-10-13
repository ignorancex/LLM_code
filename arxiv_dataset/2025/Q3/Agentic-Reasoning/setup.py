from setuptools import setup, find_packages

setup(
    name="agentic_research",
    version="0.1",
    packages=find_packages() + ['scripts', 'scripts.tools'],
    install_requires=[
        'litellm',
        'dspy',
        'numpy',
    ]
) 