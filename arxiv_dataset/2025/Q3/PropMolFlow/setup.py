from setuptools import setup, find_packages

setup(
    name="propmolflow",
    version="0.1.0",
    author="Jirui Jin, Cheng Zeng",
    author_email="jiruijin@ufl.edu, c.zeng@ufl.edu",
    description="Flow matching for 3D de novo molecule generation with certain property.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Liu-Group-UF/PropMolFlow.git",  
    packages=find_packages(),
    install_requires=[
        "wandb",
        "useful_rdkit_utils",
        "py3Dmol",
        "ase",
    ],
    python_requires=">=3.11",
)
