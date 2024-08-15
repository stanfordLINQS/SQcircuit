"""
Importing all the modules.
"""

import logging
sq_logger = logging.getLogger(__name__)

def get_logger() -> logging.Logger:
    """Get the SQcircuit-wide parent logger."""
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
