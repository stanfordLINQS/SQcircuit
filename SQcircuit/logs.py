"""logs.py contains the error and warning messages for SQcircuit"""

import warnings

from SQcircuit.settings import get_optim_mode
import SQcircuit.functions as sqf

UNIT_ERROR = "The input unit is not correct. Look at the API documentation " \
             "for the correct input format."

OPTIM_ERROR = "Please turn on the optimization mode of the SQcircuit."


def raise_unit_error():
    raise TypeError(UNIT_ERROR)


def raise_optim_error_if_needed():

    if get_optim_mode() is False:
        raise ValueError(OPTIM_ERROR)


def raise_value_out_of_bounds_error(element_type, cutoff_value, element_value):
    if cutoff_value < element_value:
        raise ValueError(f"Setting {element_value} for element of type {element_type} is greater than the maximum allowed value of {cutoff_value}.")
    if cutoff_value > element_value:
        raise ValueError(f"Setting {element_value} for element of type {element_type} is lower than the minimum allowed value of {cutoff_value}")


def raise_value_out_of_bounds_warning(element_type, cutoff_value, element_value):
    if cutoff_value < element_value:
        overshoot_ratio = sqf.round((element_value - cutoff_value) / cutoff_value, 3)
        warnings.warn(f"Setting {element_value} for element of type {element_type} is {overshoot_ratio} times above the maximum allowed value of {cutoff_value}.")
    if cutoff_value > element_value:
        undershoot_ratio = sqf.round((cutoff_value - element_value) / cutoff_value, 3)
        warnings.warn(f"Setting {element_value} for element of type {element_type} is {undershoot_ratio} times lower than the maximum allowed value of {cutoff_value}.")