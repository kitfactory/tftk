import os, sys
from setuptools import setup, find_packages

def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements

setup(
    name='tftk',
    version='0.0.1',
    description='TensorFlow simple utility',
    long_description='READEME.md',
    author='Naruhide KITADA',
    author_email='kitfactory@gmail.com',
    install_requires=read_requirements(),
    url='https://github.com/kitfactory/tftk',
    license='LICENSE',
    packages=find_packages(exclude=('tests', 'docs'))
)
