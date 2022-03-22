from setuptools import setup
from Cython.Build import cythonize
import matplotlib.pyplot as plt

import numpy

setup(
    ext_modules = cythonize('cyth.pyx'),
    include_dirs=[numpy.get_include()]
)
