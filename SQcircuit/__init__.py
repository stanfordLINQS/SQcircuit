"""
Importing all the modules.
"""

import logging
import sys
sq_logger = logging.getLogger(__name__)

def get_logger() -> logging.Logger:
    """Get the SQcircuit-wide parent logger."""
    return sq_logger

def log_to_stdout(level: int =logging.INFO) -> logging.Logger:
    """Set the SQcircuit package to log to stdout.

    Parameters
    ----------
        level:
            The minimum level of logs to log.

    Returns
    ----------
        The logger.
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    handler.setFormatter(formatter)
    sq_logger.addHandler(handler)
    sq_logger.setLevel(level)

    return sq_logger

import qutip
qutip.settings.core['auto_tidyup'] = True
qutip.settings.core['auto_tidyup_atol'] = 1e-12 # Make consistent with QuTip 4.7.x

from SQcircuit.elements import *
from SQcircuit.circuit import *
from SQcircuit.sweep import *
from SQcircuit.storage import *
from SQcircuit.units import *
from SQcircuit.systems import *

from SQcircuit.settings import (
    set_engine,
    get_engine,
)
from SQcircuit.torch_extensions import set_max_eigenvector_grad


#TODO: change to importlib meta
try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    version_tuple = (0, 0, "unknown")
