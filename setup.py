from setuptools import setup
import os

ON_RTD = os.environ.get('READTHEDOCS', None) == 'True'

if ON_RTD:
	include_package_data = False
else:
	include_package_data = True

try:
	from dypy import __version__, __author__
except RuntimeError:  # added so it can be installed with setup.py develop when signed in as an admin that's not signed into Portal in ArcGIS Pro
	__version__ = "0.0.1b"
	__author__ = "nickrsan"

setup(name="dypy",
	version=__version__,
	description="A general purpose backward dynamic programming solver",
	long_description="""Forthcoming
	""",
	packages=['dypy',],
	requires=['numpy', 'pandas', 'six'],
	author=__author__,
	author_email="nrsantos@ucdavis.edu",
	url='https://github.com/ucd-cws/dypy',
	include_package_data=include_package_data,
)
