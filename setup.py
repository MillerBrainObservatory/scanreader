#!/usr/bin/env python3
from setuptools import setup

long_description = "Python TIFF Stack Reader for ScanImage recordings (including multiROI)."

setup(
    name='scanreader',
    version='0.4.12',
    description="Reader for ScanImage 5 scans (including slow stacks and multiROI).",
    long_description=long_description,
    license='MIT',
    url='https://github.com/MillerBrainObservtory/scanreader',
    keywords='ScanImage scanreader multiROI 2019 tiff',
    packages=['scanreader'],
    install_requires=['numpy>=1.12.0', 'tifffile>=2019.2.22'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English'
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
