from setuptools import setup, find_packages

setup(
    name='watermark_stealing',
    version='0.1.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        # List dependencies here
    ],
    entry_points={
        'console_scripts': [
            'watermark_stealing=watermark_stealing.main:main'  # adjust as needed
        ]
    }
)
