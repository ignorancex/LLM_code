from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='comparecast_causal',
    version='0.0.0',
    packages=['comparecast_causal', 'comparecast_causal.data_utils'],
    python_requires='>=3.7.1',
    install_requires=[
        'numpy>=1.22',
        'scipy',
        'pandas>=1.0',
        'seaborn>=0.12',
        'tqdm',
        'jupyter',
        'scikit-learn>=1.0',
        'statsmodels>=0.13.1',
        'mlens>=0.2.3',
        'comparecast',
    ],
    url='https://github.com/yjchoe/ComparingAbstainingClassifiers',
    license='MIT',
    author='Yo Joong Choe, Aditya Gangrade, Aaditya Ramdas',
    author_email='yjchoe@uchicago.edu',
    description='Counterfactually Comparing Abstaining Classifiers',
    long_description=long_description,
    long_description_content_type="text/markdown",
)
