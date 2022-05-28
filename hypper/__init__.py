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

from hypper.classification import *
from hypper.data import *
from hypper.feature_selection import *
from hypper.plotting import *
from hypper.undersampling import *

__all__ = [
    "hypper.classification",
    "hypper.data",
    "hypper.feature_selection",
    "hypper.plotting",
    "hypper.undersampling",
]
