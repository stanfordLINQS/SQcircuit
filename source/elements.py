"""
elements.py contains the classes for the circuit elements:
capacitors, inductors, and josephson junctions.
"""

from units import Units
import numpy as np


class Capacitor:
    """
    class that contains the capacitor properties.
    """

    def __init__(self, value, unit):
        """
        inputs:
            -- value: the value of the capacitor
            -- units: the unit of input value
        """

        # object that contains the units and physic constants
        self.ut = Units()
        if unit not in self.ut.freq and unit not in self.ut.farad:
            error = "The input unit for the inductor is not correct. Look at the documentation for the correct input " \
                    "format."
            raise ValueError(error)

        self.cValue = value
        self.cUnit = unit

    def value(self):
        """
        returns the value of the capacitor in Farad units
        """
        if self.cUnit in self.ut.farad:
            return self.cValue * self.ut.farad[self.cUnit]
        else:
            E_c = self.cValue * self.ut.freq[self.cUnit] * (2 * np.pi * self.ut.hbar)
            return self.ut.e ** 2 / 2 / E_c

    def energy(self):
        """
        returns the charging energy of the capacitor in Hz.
        """
        if self.cUnit in self.ut.freq:
            return self.cValue * self.ut.freq[self.cUnit]
        else:
            c = self.cValue * self.ut.farad[self.cUnit]
            return self.ut.e ** 2 / 2 / c / (2 * np.pi * self.ut.hbar)


class Inductor:
    """
    class that contains the inductor properties.
    """

    def __init__(self, value, unit):
        """
        inputs:
            -- value: the value of the inductor
            -- units: the unit of input value
        """

        # object that contains the units and physic constants
        self.ut = Units()
        if unit not in self.ut.freq and unit not in self.ut.farad:
            error = "The input unit for the inductor is not correct. Look at the documentation for the correct input " \
                    "format."
            raise ValueError(error)

        self.lValue = value
        self.lUnit = unit

    def value(self):
        """
        returns the value of the inductor in Farad units
        """
        if self.lUnit in self.ut.farad:
            return self.lValue * self.ut.farad[self.lUnit]
        else:
            E_l = self.lValue * self.ut.freq[self.lUnit] * (2 * np.pi * self.ut.hbar)
            return (self.ut.Phi0 / 2 / np.pi) ** 2 / E_l

    def energy(self):
        """
        returns the inductive energy of the inductor in Hz.
        """
        if self.lUnit in self.ut.freq:
            return self.lValue * self.ut.freq[self.lUnit]
        else:
            l = self.lValue * self.ut.farad[self.lUnit]
            return (self.ut.Phi0 / 2 / np.pi) ** 2 / l / (2 * np.pi * self.ut.hbar)


class Junction:
    """
    class that contains the Josephson Junction properties.
    """

    def __init__(self, value, unit):
        """
        inputs:
            -- value: the value of the inductor
            -- units: the unit of input value
        """

        # object that contains the units and physic constants
        self.ut = Units()
        if unit not in self.ut.freq:
            error = "The input unit for the Josephson Junction is not correct. Look at the documentation for the" \
                    "correct input format."
            raise ValueError(error)

        self.jValue = value
        self.jUnit = unit

    def value(self):
        """
        returns the value of the Josephson Junction in angular frequency
        """
        return self.jValue * self.ut.freq[self.jUnit] * 2 * np.pi
