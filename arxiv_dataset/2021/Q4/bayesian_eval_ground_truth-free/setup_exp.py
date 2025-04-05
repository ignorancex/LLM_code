from setuptools import setup

long_description = ''
with open('experiment/index.md', 'r') as f:
    long_description = f.read()

install_requires = ''
with open('requirements.txt', 'r') as f:
    install_requires = f.read()

setup(
    name='experiment',
    version='0.1.0',
    author='',
    author_email='',
    packages=[
        'experiment',
        #'experiment.crowd_layer',
        'experiment.research',
        'experiment.research.bnn',
        'experiment.research.measure',
        'experiment.research.pred',
        'experiment.research.sjd',
        'experiment.visuals',
    ],
    #scripts
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='',
    install_requires=install_requires, # requires psych_metric
    python_requires='>=3.6',
)
