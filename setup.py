from setuptools import setup, find_packages

def get_numpy_include():
    try:
        import numpy
        return [numpy.get_include()]
    except Exception:
        # numpy may not be available at import time (build isolation);
        # build frontend will install it from pyproject.toml when needed.
        return []

setup(
    name="process_petsys",
    packages=find_packages(),
    include_dirs=get_numpy_include(),
)