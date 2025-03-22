from setuptools import setup

setup(
    description='Light scattering properties of complex particles at visible ane near infrared',
    author='Ryo Tazaki',
    author_email='ryo.tazaki1205@gmail.com',
    url='https://github.com/rtazaki1205/AggScatVIR',
    project_urls={'Documentation': 'https://rtazaki1205.github.io/AggScatVIR/'},
    name='aggscatpy',
    version='1.0.0',
    license='MIT',
    packages=['aggscatpy'],
    install_requires=[
        'numpy','matplotlib'
    ],
)
