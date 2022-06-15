"""physParam.py module contain the physical constants and units"""


henryList = {'H': 1.0, 'mH': 1.0e-3, 'uH': 1.0e-6, 'nH': 1.0e-9, 'pH': 1.0e-12, 'fH': 1.0e-15}

# Farad units
faradList = {'F': 1, 'mF': 1.0e-3, 'uF': 1.0e-6, 'nF': 1.0e-9, 'pF': 1.0e-12, 'fF': 1.0e-15}

# frequency unit list
freqList = {'Hz': 1.0, 'kHz': 1.0e3, 'MHz': 1.0e6, 'GHz': 1.0e9, 'THz': 1.0e12}

timeList = {'s': 1.0, 'ms': 1.0e-3, 'us': 1.0e-6, 'ns': 1.0e-9, 'ps': 1.0e-12, 'fs': 1.0e-15}

# reduced Planck constant
hbar = 1.0545718e-34

# magnetic flux quantum
Phi0 = 2.067833e-15

# electron charge
e = 1.6021766e-19

k_B = 1.38e-23

# main frequency unit of the SQcircuit
freq = freqList["GHz"]

# default unit of capacitors
capU = "GHz"

# default unit of inductors
indU = "GHz"

# default unit of JJs
junU = "GHz"


def set_unit_freq(unit: str):
    """
    Change the main frequency unit of the SQcircuit.

    Parameters
    ----------
        -- unit: str
        The desired frequency unit, which can be "THz", "GHz", and ,etc.
    """
    assert unit in freqList, "The input format is not correct."

    global freq

    freq = freqList[unit]


def set_unit_cap(unit: str):
    """
    Change the default unit for capacitors

    Parameters
    ----------
        -- unit: str
        The desired capacitor default unit, which can be "THz", "GHz", and, etc., or "fF", "pF", and ,etc.
    """
    if unit not in freqList and unit not in faradList:
        error = "The input unit is not correct. Look at the documentation for the correct input format."
        raise ValueError(error)

    global capU

    capU = unit


def set_unit_ind(unit: str):
    """
    Change the default unit for inductors

    Parameters
    ----------
        -- unit: str
         The desired inductor default unit, which can be "THz", "GHz", and, etc., or "fH", "pH", and ,etc.
    """
    if unit not in freqList and unit not in henryList:
        error = "The input unit is not correct. Look at the documentation for the correct input format."
        raise ValueError(error)

    global indU

    indU = unit


def set_unit_JJ(unit: str):
    """
    Change the default unit for Josephson junctions.

    Parameters
    ----------
        -- unit: str
        The desired Josephson junction default unit, which can be "THz", "GHz", and ,etc.
    """
    assert unit in freqList, "The input format is not correct."

    global junU

    junU = unit
