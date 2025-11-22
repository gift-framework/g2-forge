# -*- coding: utf-8 -*-
"""
g2-forge: Universal Neural Construction of G2 Holonomy Metrics
"""

from setuptools import setup, find_packages
import os

# Read version from __init__.py
def get_version():
    init_file = os.path.join('g2forge', '__init__.py')
    with open(init_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip("'\"")
    return '0.1.0-dev'

# Read README
def get_long_description():
    readme_file = 'README.md'
    if os.path.exists(readme_file):
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

setup(
    name='g2forge',
    version=get_version(),
    author='Brieuc de La Fourniere',
    author_email='brieuc@bdelaf.com',
    description='Universal Neural Construction of G2 Holonomy Metrics',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/gift-framework/g2-forge',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'jupyter>=1.0.0',
            'matplotlib>=3.7.0',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/gift-framework/g2-forge/issues',
        'Source': 'https://github.com/gift-framework/g2-forge',
    },
)
