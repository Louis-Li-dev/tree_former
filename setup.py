"""
setup.py for HMST-v2
====================

Install via:
    pip install git+https://github.com/<your-org>/hmst.git

Or in editable mode (development):
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name             = "hmst",
    version          = "0.1.0",
    description      = (
        "HMST-v2: Hierarchical Masked Spatial Transformer for "
        "urban population-flow forecasting"
    ),
    author           = "HMST Research Team",
    python_requires  = ">=3.10",
    packages         = find_packages(include=["hmst", "hmst.*"]),
    install_requires = [
        "torch>=2.0",
        "numpy>=1.24",
        "pandas>=1.5",
        "scikit-learn>=1.2",
        "pyarrow>=12.0",          # for reading .parquet data files
        "tabulate>=0.9",           # for to_markdown() in results tables
    ],
    extras_require = {
        "dev": [
            "notebook>=7.0",
            "matplotlib>=3.7",
        ],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
