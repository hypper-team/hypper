import pathlib
import sys
from pathlib import Path

import pkg_resources
from setuptools import setup

__version__ = "0.0.2"

# Check Python version
if sys.version_info < (3, 7):
    sys.exit("Hypper requires Python 3.7 or later.")

# Parse requirements
with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

# Setup build
setup(
    name="hypper",
    packages=["hypper"],
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
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    extras_require={
        "testing": ["pytest"],
        "documentation": ["decorator>=5.1.1", "pdoc"],
        "all": ["decorator>=5.1.1", "pytest", "pdoc", "autopep8"],
    },
)
