from setuptools import setup
from pathlib import Path
import sys
import subprocess

__version__ = "0.0.4"

# Check Python version
if sys.version_info < (3, 7):
    sys.exit("Hypper requires Python 3.7 or later.")

# Setup build
setup(
    name="hypper",
    packages=[
        "hypper"
    ],
    version=__version__,
    author="Szymon Janowski, Paweł Misiorek",
    author_email="szy.janowski@gmail.com, pawel.misiorek@put.poznan.pl",
    url="https://github.com/sleter/hypper",
    description="Hypergraph-based data mining tool for binary classification.",
    project_urls={
        "Bug Tracker": "https://github.com/sleter/hypper/issues",
        "Documentation": "https://hypper-team.github.io/hypper.html"
    },
    install_requires=[
        "pandas>=1.4.1",
        "numpy>=1.22.3",
        "scikit-learn>=1.0.2",
        "bidict>=0.21.4",
        "psutil>=5.9.0",
        "hypernetx>=1.2.3"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type='text/markdown',
    extras_require={
        "testing": ["pytest>=7.0.1"],
        "documentation": ["decorator>=5.1.1", "pdoc>=10.0.3"],
        "all": [
            "pytest>=7.0.1",
            "decorator>=5.1.1",
            "pdoc>=10.0.3"

        ],
    },
)