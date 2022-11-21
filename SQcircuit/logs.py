"""logs.py contains the error and warning messages for SQcircuit"""

import warnings

from SQcircuit.settings import get_optim_mode

UNIT_ERROR = "The input unit is not correct. Look at the API documentation " \
             "for the correct input format."

OPTIM_ERROR = "Please turn on the optimization mode of the SQcircuit."


def raise_unit_error():
    raise TypeError(UNIT_ERROR)


def raise_optim_error_if_needed():

    if get_optim_mode() is False:
        raise ValueError(OPTIM_ERROR)


def raise_negative_value_error(baseline_value, element_value):
    raise ValueError(f"Attempting to set user-provided element value to {element_value}. This is lower than the baseline value of {baseline_value}, so SQcircuit will terminate.")


def raise_negative_value_warning(baseline_value, element_value):
    raise ValueError(f"Attempting to set element value to {element_value} during optimization. This is less than the baseline value of {baseline_value}, so SQcircuit will automatically set this element's value to the baseline.")