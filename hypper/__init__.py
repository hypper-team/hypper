"""
.. include:: ../README.md
"""
__docformat__ = "restructuredtext"

import logging
import logging.config
from pathlib import Path

logging.config.fileConfig(
    fname=Path(__file__).parent / "logging" / "logger.conf",
    disable_existing_loggers=False,
)
