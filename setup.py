from setuptools import setup, find_packages
import numpy

setup(
    name="process_petsys", packages=find_packages(), include_dirs=[numpy.get_include()]
)
