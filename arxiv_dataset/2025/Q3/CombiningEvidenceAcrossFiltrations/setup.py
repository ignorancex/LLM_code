from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='ecombine',
    version='0.0.0',
    packages=['ecombine', 'ecombine.data'],
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.20',
        'scipy',
        'pandas>=1.0',
        'seaborn>=0.11',
        'tqdm',
        'openpyxl',
        'confseq>=0.0.9',
        'comparecast==1.0.0',
    ],
    url='https://github.com/yjchoe/CombiningEvidenceAcrossFiltrations',
    license='MIT',
    author='Yo Joong Choe, Aaditya Ramdas',
    author_email='yjchoe@uchicago.edu',
    description='Combining Evidence Across Filtrations',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
