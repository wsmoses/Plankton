import os
from setuptools import setup, find_packages
import itertools
import glob
import os

with open("README.md", "r") as fp:
    long_description = fp.read()

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
setup(
    name='plankton',
    version='0.1',
    url='https://github.com/wsmoses/Plankton',
    author='Multiple',
    author_email='multiple@authors.info',
    description='Plankton',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6, <3.10',
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_data={'': (['*.cpp'])},
    install_requires=[],
    extras_require={
        'testing': [
            'coverage', 'pytest', 'yapf', 'pytest-cov', 'pytest-xdist',
            'torchvision'
        ],
    })
