"""logs.py contains the error and warning messages for SQcircuit"""

import warnings

from SQcircuit.settings import get_optim_mode
import SQcircuit.functions as sqf
import numpy as np

UNIT_ERROR = "The input unit is not correct. Look at the API documentation " \
             "for the correct input format."

OPTIM_ERROR = "Please turn on the optimization mode of the SQcircuit."


def raise_unit_error():
    raise TypeError(UNIT_ERROR)


def raise_optim_error_if_needed():

    if get_optim_mode() is False:
        raise ValueError(OPTIM_ERROR)