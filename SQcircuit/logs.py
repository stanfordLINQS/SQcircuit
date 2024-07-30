"""logs.py contains the error and warning messages for SQcircuit"""
from typing import List, Union

from SQcircuit.settings import (
    get_optim_mode,
    get_engine
)


UNIT_ERROR = ('The input unit is not valid. Look at the API documentation for '
              'the allowed unit formats.')


def raise_unit_error():
    raise ValueError(UNIT_ERROR)


def raise_optim_error_if_needed():
    if get_optim_mode() is False:
        raise ModeError('PyTorch')


class ModeError(Exception):
    """Exception raised when trying to use a feature of SQcircuit which
    is not supported by the currently activated engine."""
    def __init__(self, correct_engine: Union[str, List[str]]):
        self.correct_engine = correct_engine
    def __str__(self):
        if isinstance(self.correct_engine, list):
            correct_eng_str = ' or '.join(
                [f"'{eng}'" for eng in self.correct_engine]
            )
        else:
            correct_eng_str = f"'{self.correct_engine}'"
        return (f'The current operation is not supported by the \'{get_engine()}\' '
                f'engine. Please set the engine to {correct_eng_str}.')


class CircuitStateError(Exception):
    """Exception raised when the circuit state does not permit a method to be
    run."""
