"""units.py module contain the physical constants and units"""


henry_list = {'H': 1.0, 'mH': 1.0e-3, 'uH': 1.0e-6,
              'nH': 1.0e-9, 'pH': 1.0e-12, 'fH': 1.0e-15}

# farad units
farad_list = {'F': 1, 'mF': 1.0e-3, 'uF': 1.0e-6,
              'nF': 1.0e-9, 'pF': 1.0e-12, 'fF': 1.0e-15}

# frequency unit list
freq_list = {'Hz': 1.0, 'kHz': 1.0e3, 'MHz': 1.0e6,
             'GHz': 1.0e9, 'THz': 1.0e12}

time_list = {'s': 1.0, 'ms': 1.0e-3, 'us': 1.0e-6,
             'ns': 1.0e-9, 'ps': 1.0e-12, 'fs': 1.0e-15}

# reduced Planck constant
hbar = 1.0545718e-34

# magnetic flux quantum
Phi0 = 2.067833e-15

# electron charge
e = 1.6021766e-19

k_B = 1.38e-23

# main frequency unit of the SQcircuit
_unit_freq = freq_list["GHz"]

# default unit of capacitors
_unit_cap = "GHz"

# default unit of inductors
_unit_ind = "GHz"

# default unit of JJs
_unit_JJ = "GHz"


def set_unit_freq(unit: str) -> None:
    """
    Change the main frequency unit of the SQcircuit.

    Parameters
    ----------
        unit:
            The desired frequency unit, which can be "THz", "GHz", and ,etc.
    """
    assert unit in freq_list, "The input format is not correct."

    global _unit_freq

    _unit_freq = freq_list[unit]


def get_unit_freq() -> float:
    """
    get current frequency unit of the SQcircuit.

    Returns
    ----------
        unit:
            frequency unit of the SQcircuit in hertz
    """

    return _unit_freq


def set_unit_cap(unit: str) -> None:
    """
    Change the default unit for capacitors

    Parameters
    ----------
        unit:
            The desired capacitor default unit, which can be "THz", "GHz", and,
            etc., or "fF", "pF", and ,etc.
    """
    if unit not in freq_list and unit not in farad_list:
        error = ("The input unit is not correct. Look at the documentation "
                 "for the correct input format.")
        raise ValueError(error)

    global _unit_cap

    _unit_cap = unit


def get_unit_cap() -> str:
    """
    Get current unit of capacitor

    Returns
    ----------
        unit:
            capacitor unit
    """

    return _unit_cap


def set_unit_ind(unit: str) -> None:
    """
    Change the default unit for inductors

    Parameters
    ----------
        unit:
             The desired inductor default unit, which can be "THz", "GHz", and,
             etc., or "fH", "pH", and ,etc.
    """
    if unit not in freq_list and unit not in henry_list:
        error = ("The input unit is not correct. Look at the documentation "
                 "for the correct input format.")
        raise ValueError(error)

    global _unit_ind

    _unit_ind = unit


def get_unit_ind() -> str:
    """
    Get current unit of inductor

    Returns
    ----------
        unit:
            inductor unit
    """

    return _unit_ind


def set_unit_JJ(unit: str) -> None:
    """
    Change the default unit for Josephson junctions.

    Parameters
    ----------
        unit:
            The desired Josephson junction default unit, which can be "THz",
            "GHz", and ,etc.
    """
    assert unit in freq_list, "The input format is not correct."

    global _unit_JJ

    _unit_JJ = unit


def get_unit_JJ() -> str:
    """
    Get current unit of Josephson junction unit

    Returns
    ----------
        str:
            Josephson junction unit
    """

    return _unit_JJ
