import pathlib
import sys

import pkg_resources
from setuptools import setup

__version__ = "0.0.5"

# Check Python version
if sys.version_info < (3, 7):
    sys.exit("Hypper requires Python 3.7 or later.")

# Setup build
setup(
    name="hypper",
    packages=[
        "hypper",
        "hypper.classification",
        "hypper.data",
        "hypper.feature_selection",
        "hypper.plotting",
        "hypper.undersampling",
    ],
    version=__version__,
    author="Szymon Janowski, Paweł Misiorek",
    author_email="szy.janowski@gmail.com, pawel.misiorek@put.poznan.pl",
    url="https://github.com/hypper-team/hypper",
    description="Hypergraph-based data mining tool for binary classification.",
    keywords="hypergraphs machine-learning undersampling feature-selection classification",
    project_urls={
        "Bug Tracker": "https://github.com/hypper-team/hypper/issues",
        "Documentation": "https://hypper-team.github.io/hypper.html",
    },
    install_requires=[
        "pandas>=1.3.5",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.2",
        "bidict>=0.22.0",
        "psutil>=5.9.0",
        "hypernetx>=1.2.3",
        "requests>=2.23.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    long_description=(pathlib.Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    package_data={"hypper": ["logger.conf"]},
    extras_require={
        "documentation": ["decorator>=5.1.1", "pdoc"],
        "develop": ["pytest", "black"],
        "benchmarking": [
            "tqdm",
            "pyyaml",
            "imbalanced-learn",
            "xgboost",
            "catboost",
            "lightgbm",
        ],
        "all": [
            "decorator>=5.1.1",
            "pytest",
            "pdoc",
            "black",
            "tqdm",
            "pyyaml",
            "imbalanced-learn",
            "xgboost",
            "catboost",
            "lightgbm",
        ],
    },
)
