from setuptools import setup, find_packages
import numpy

setup(name="event_petsys", packages=find_packages(), include_dirs=[numpy.get_include()])
