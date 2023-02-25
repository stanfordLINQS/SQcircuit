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


def raise_value_out_of_bounds_error(element_type, cutoff_value, element_value):
    if cutoff_value < element_value:
        raise ValueError(f"Attempting to set value for element of type {element_type} to {element_value}. This is greater than the maximum value of {cutoff_value}, so SQcircuit will terminate.")
    if cutoff_value > element_value:
        raise ValueError(f"Attempting to set value for element of type {element_type} to {element_value}. This is lower than the minimum value of {cutoff_value}, so SQcircuit will terminate.")


def raise_value_out_of_bounds_warning(element_type, cutoff_value, element_value):
    if cutoff_value < element_value:
        warnings.warn(f"Attempting to set value for element of type {element_type} to {element_value} during optimization. This is greater than the maximum value of {cutoff_value}, so SQcircuit will automatically set this element's value to the maximum.")
    if cutoff_value > element_value:
        warnings.warn(f"Attempting to set value for element of type {element_type} to {element_value} during optimization. This is lower than the minimum value of {cutoff_value}, so SQcircuit will automatically set this element's value to the minimum.")