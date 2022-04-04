"""physParam.py module contain the physical constants and units"""


henryList = {'H': 1.0, 'mH': 1.0e-3, 'uH': 1.0e-6, 'nH': 1.0e-9, 'pH': 1.0e-12, 'fH': 1.0e-15}

# Farad units
faradList = {'F': 1, 'mF': 1.0e-3, 'uF': 1.0e-6, 'nF': 1.0e-9, 'pF': 1.0e-12, 'fF': 1.0e-15}

# frequency unit list
freqList = {'Hz': 1.0, 'kHz': 1.0e3, 'MHz': 1.0e6, 'GHz': 1.0e9, 'THz': 1.0e12}

# reduced Planck constant
hbar = 1.0545718e-34

# magnetic flux quantum
Phi0 = 2.067833e-15

# electron charge
e = 1.6021766e-19

k_B = 1.38e-23

# main frequency unit of the SQcircuit
freq = freqList["GHz"]


def setFreq(fUnit):
    """
    changes the main frequency unit of the SQcircuit
    inputs:
        -- fUnit: the desired frequency unit
    """
    assert fUnit in freqList, "The input format is not correct."

    global freq

    freq = freqList[fUnit]

