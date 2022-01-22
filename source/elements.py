"""
elements.py contains the classes for the circuit elements:
capacitors, inductors, and josephson junctions.
"""

from units import *
import numpy as np


class Capacitor:
    """
    class that contains the capacitor properties.
    """

    def __init__(self, value, cUnit):
        """
        inputs:
            -- value: the value of the capacitor
            -- units: the unit of input value
        """

        if cUnit not in unit.freqList and cUnit not in unit.faradList:
            error = "The input unit for the inductor is not correct. Look at the documentation for the correct input " \
                    "format."
            raise ValueError(error)

        self.cValue = value
        self.cUnit = cUnit

    def value(self):
        """
        returns the value of the capacitor in Farad units
        """
        if self.cUnit in unit.faradList:
            return self.cValue * unit.faradList[self.cUnit]
        else:
            E_c = self.cValue * unit.freqList[self.cUnit] * (2 * np.pi * unit.hbar)
            return unit.e ** 2 / 2 / E_c

    def energy(self):
        """
        returns the charging energy of the capacitor in Hz.
        """
        if self.cUnit in unit.freqList:
            return self.cValue * unit.freqList[self.cUnit]
        else:
            c = self.cValue * unit.faradList[self.cUnit]
            return unit.e ** 2 / 2 / c / (2 * np.pi * unit.hbar)


class Inductor:
    """
    class that contains the inductor properties.
    """

    def __init__(self, value, lUnit):
        """
        inputs:
            -- value: the value of the inductor
            -- units: the unit of input value
        """

        if lUnit not in unit.freqList and lUnit not in unit.henryList:
            error = "The input unit for the inductor is not correct. Look at the documentation for the correct input " \
                    "format."
            raise ValueError(error)

        self.lValue = value
        self.lUnit = lUnit

    def value(self):
        """
        returns the value of the inductor in Henry units
        """
        if self.lUnit in unit.henryList:
            return self.lValue * unit.henryList[self.lUnit]
        else:
            E_l = self.lValue * unit.freqList[self.lUnit] * (2 * np.pi * unit.hbar)
            return (unit.Phi0 / 2 / np.pi) ** 2 / E_l

    def energy(self):
        """
        returns the inductive energy of the inductor in Hz.
        """
        if self.lUnit in unit.freqList:
            return self.lValue * unit.freqList[self.lUnit]
        else:
            l = self.lValue * unit.henryList[self.lUnit]
            return (unit.Phi0 / 2 / np.pi) ** 2 / l / (2 * np.pi * unit.hbar)


class Junction:
    """
    class that contains the Josephson Junction properties.
    """

    def __init__(self, value, jUnit):
        """
        inputs:
            -- value: the value of the inductor
            -- units: the unit of input value
        """

        if jUnit not in unit.freqList:
            error = "The input unit for the Josephson Junction is not correct. Look at the documentation for the" \
                    "correct input format."
            raise ValueError(error)

        self.jValue = value
        self.jUnit = jUnit

    def value(self):
        """
        returns the value of the Josephson Junction in angular frequency
        """
        return self.jValue * unit.freqList[self.jUnit] * 2 * np.pi
