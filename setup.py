from setuptools import setup, find_packages

exec(open('djimaging/version.py').read())

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
