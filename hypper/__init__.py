"""
.. include:: ../README.md
"""
__docformat__ = "restructuredtext"

import logging
import logging.config
from pathlib import Path

logging.config.fileConfig(
    fname=Path(__file__).parent / "logger.conf",
    disable_existing_loggers=False,
)

from .classification import *
from .data import *
from .feature_selection import *
from .plotting import *
from .undersampling import *

__all__ = [
    "base",
    "hypergraph",
    "feature_selection",
    "undersampling",
    "classification",
    "data",
    "plotting",
]
