import os

from setuptools import find_packages, setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext

import numpy as np

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "ancomP",
    version = "0.0.1",
    author = "Jamie Morton",
    author_email = "jamietmorton@gmail.com",
    description = ("Statistical methods for analyzing microbiomes"),
    packages=find_packages(),
    long_description=read('README.md'),
    license='MIT',
    include_dirs=[np.get_include()],
    install_requires=[
          'pandas',
          'numpy >= 1.7',
          'scipy >= 0.13.0'
          'biom-format',
          'scikit-bio >= 0.4.0'
      ],
)
