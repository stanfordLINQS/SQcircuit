"""
environment.py contains the properties of the environment and noise parameters
"""

import numpy as np

import SQcircuit.units as unt


ENV = {
    "T":  0.015,    # experiment time
    "omega_low": 2 * np.pi,    # low-frequency cut off
    "omega_high": 2 * np.pi * 3 * 1e9,    # high-frequency cut off
    "t_exp": 10e-6,    # experiment time
}


def set_temp(T: float) -> None:
    """
    Set the temperature of the circuit.

    Parameters
    ----------
        T: float
            The temperature in Kelvin
    """

    global ENV

    ENV["T"] = T


def set_low_freq(value: float, unit: str) -> None:
    """
    Set the low-frequency cut-off.

    Parameters
    ----------
        value:
            The value of the frequency.
        unit:
            The unit of the input value in hertz unit that can be
            ``"THz"``, ``"GHz"``, ``"MHz"``,and ,etc.
    """

    global ENV

    ENV["omega_low"] = 2 * np.pi * value * unt.freq_list[unit]


def set_high_freq(value: float, unit: str) -> None:
    """
    Set the high-frequency cut-off.

    Parameters
    ----------
        value:
            The value of the frequency.
        unit:
            The unit of the input value in hertz unit that can be
            ``"THz"``, ``"GHz"``, ``"MHz"``,and ,etc.
    """

    global ENV

    ENV["omega_high"] = 2 * np.pi * value * unt.freq_list[unit]


def set_t_exp(value: float, unit: str) -> None:
    """
    Set the measurement time.

    Parameters
    ----------
        value:
            The value of the measurement time.
        unit:
            The unit of the input value in time unit that can be
            ``"s"``, ``"ms"``, ``"us"``,and ,etc.
    """

    global ENV

    ENV["t_exp"] = value * unt.time_list[unit]


def reset_to_default() -> None:
    """ Reset the ENV parameters back to SQcircuit default"""

    set_temp(0.015)

    set_low_freq(1, 'Hz')

    set_high_freq(3, 'GHz')

    set_t_exp(10, 'us')
