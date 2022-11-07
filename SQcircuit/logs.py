"""logs.py contains the error and warning messages for SQcircuit"""

from SQcircuit.settings import get_optim_mode

UNIT_ERROR = "The input unit is not correct. Look at the API documentation " \
             "for the correct input format."

OPTIM_ERROR = "Please turn on the optimization mode of the SQcircuit."


def raise_unit_error():
    raise TypeError(UNIT_ERROR)


def raise_optim_error_if_needed():

    if get_optim_mode() is False:
        raise ValueError(OPTIM_ERROR)


def raise_negative_value_error(element, value, unit):
    raise ValueError(f"Attempting to set element {element} value to {value} {unit}. \
    Negative values are not supported, so this will be set to the absolute value instead.")