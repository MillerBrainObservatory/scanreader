from setuptools import find_packages, setup

#!/usr/bin/env python3
# twine upload dist/rbo-lbm-x.x.x.tar.gz
# twine upload dist/rbo-lbm.x.x.tar.gz -r test
# pip install --index-url https://test.pypi.org/simple/ --upgrade rbo-lbm

import setuptools

install_deps = [
        "tifffile",
        "numpy>=1.24.3",
        "scipy>=1.9.0",
        "dask",
        "zarr"
        ]

extras_require = {
    "docs": [
        "sphinx>=6.1.3",
        "docutils>=0.19",
        "nbsphinx",
        "numpydoc",
        "sphinx-autodoc2",
        "sphinx_gallery",
        "sphinx-togglebutton",
        "sphinx-copybutton",
        "sphinx_book_theme",
        "pydata_sphinx_theme",
        "sphinx_design",
        "sphinxcontrib-images",
        "sphinxcontrib-video",
        "myst_nb",
    ],
    "notebook": [
        "jupyterlab",
    ]
}

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="scanreader",
    version="0.4.12",
    description="Reader for ScanImage 5 scans (including slow stacks and multiROI).",
    long_description=long_description,
    author="Flynn OConnell",
    author_email="foconnell@rockefeller.edu",
    license="MIT",
    url="https://github.com/MillerBrainObservtory/scanreader",
    keywords="ScanImage scanreader multiROI 2019 tiff",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English"
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        ],
    packages=find_packages(),
    install_requires=install_deps,
    extras_require=extras_require,
    include_package_data=True,
    entry_points = {
        "console_scripts": [
            "scanreader = scanreader.__main__:main",
            ]
        },
    )
