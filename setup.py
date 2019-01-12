
from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='scikit-roughsets',
    version='1.0',
    description='Feature reduction using rough set theory',
    long_description=long_description,
    url='http://www.github.com/paudan/scikit-roughsets',
    author='Paulius Danenas',
    author_email='danpaulius@gmail.com',
    license='MIT',
    keywords='machine_learning',
    packages=['scikit_roughsets'],
    package_dir={'scikit_roughsets': 'scikit_roughsets'},
    install_requires=['numpy', 'scikit-learn'],
)