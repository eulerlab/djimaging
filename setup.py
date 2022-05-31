from setuptools import setup, find_packages

__version__ = '0.0.0'
exec(open('djimaging/version.py').read())  # Read version from version file

setup(
    name='djimaging',
    version=__version__,
    packages=find_packages(include=['djimaging']),
    url='',
    license='',
    author='Jonathan Oesterle',
    author_email='jonathan.oesterle@uni-tuebingen.de',
    description='DataJoint for 2P imaging data'
)
