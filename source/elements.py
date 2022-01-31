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

    def __init__(self, value, cUnit, Q=None, error=0):
        """
        inputs:
            -- value: The value of the capacitor.
            -- units: The unit of input value.
            -- error: The error in fabrication.( as a percentage)
            -- Q: quality factor of the dielectric in the capacitor which is one over tangent loss
        """

        if cUnit not in unit.freqList and cUnit not in unit.faradList:
            error = "The input unit for the inductor is not correct. Look at the documentation for the correct input " \
                    "format."
            raise ValueError(error)

        self.cValue = value
        self.cUnit = cUnit
        self.Q = Q
        self.error = error
        self.type = type(self)

    def value(self, random: bool = False):
        """
        returns the value of the capacitor in Farad units. If random flag is true, it samples from a normal
        distribution.
        inputs:
            -- random: A flag which specifies whether the output is picked deterministically or randomly.
        """
        if self.cUnit in unit.faradList:
            cMean = self.cValue * unit.faradList[self.cUnit]
        else:
            E_c = self.cValue * unit.freqList[self.cUnit] * (2 * np.pi * unit.hbar)
            cMean = unit.e ** 2 / 2 / E_c

        if not random:
            return cMean
        else:
            return np.random.normal(cMean, cMean * self.error / 100, 1)[0]

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

    def __init__(self, value, lUnit, Q=None, error=0):
        """
        inputs:
            -- value: The value of the inductor.
            -- units: The unit of input value.
            -- error: The error in fabrication.( as a percentage)
            -- Q: quality factor of the inductor.
        """

        if lUnit not in unit.freqList and lUnit not in unit.henryList:
            error = "The input unit for the inductor is not correct. Look at the documentation for the correct input " \
                    "format."
            raise ValueError(error)

        self.lValue = value
        self.lUnit = lUnit
        self.Q = Q
        self.error = error
        self.type = type(self)

    def value(self, random: bool = False):
        """
        returns the value of the inductor in Henry units. If random flag is true, it samples from a normal
        distribution.
        inputs:
            -- random: A flag which specifies whether the output is picked deterministically or randomly.
        """
        if self.lUnit in unit.henryList:
            lMean = self.lValue * unit.henryList[self.lUnit]
        else:
            E_l = self.lValue * unit.freqList[self.lUnit] * (2 * np.pi * unit.hbar)
            lMean = (unit.Phi0 / 2 / np.pi) ** 2 / E_l

        if not random:
            return lMean
        else:
            return np.random.normal(lMean, lMean * self.error / 100, 1)[0]

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

    def __init__(self, value, jUnit, error=0):
        """
        inputs:
            -- value: The value of the inductor.
            -- units: The unit of input value.
            -- error: The error in fabrication.( as a percentage)
        """

        if jUnit not in unit.freqList:
            error = "The input unit for the Josephson Junction is not correct. Look at the documentation for the" \
                    "correct input format."
            raise ValueError(error)

        self.jValue = value
        self.jUnit = jUnit
        self.error = error
        self.type = type(self)

    def value(self, random: bool = False):
        """
        returns the value of the Josephson Junction in angular frequency. If random flag is true, it samples
        from a normal distribution.
        inputs:
            -- random: A flag which specifies whether the output is picked deterministically or randomly.
        """
        jMean = self.jValue * unit.freqList[self.jUnit] * 2 * np.pi

        if not random:
            return jMean
        else:
            return np.random.normal(jMean, jMean * self.error / 100, 1)[0]


class Flux:
    """
    class that contains the flux bias properties.
    """

    def __init__(self, value=0, noise=0):
        """
        inputs:
            -- value: The value of the bias.
            -- noise: The amplitude of the flux noise.
        """
        self.fValue = value
        self.noise = noise

    def value(self, random: bool = False):
        """
        returns the value of flux. If random flag is true, it samples from a normal distribution.
        inputs:
            -- random: A flag which specifies whether the output is picked deterministically or randomly.
        """
        if not random:
            return self.fValue
        else:
            return np.random.normal(self.fValue, self.noise, 1)[0]


class Charge:
    """
    class that contains the charge offset properties.
    """

    def __init__(self, value=0, noise=0):
        """
       inputs:
            -- value: The value of the offset.
            -- noise: The amplitude of the charge noise.
        """
        self.chValue = value
        self.noise = noise

    def value(self, random: bool = False):
        """
        returns the value of charge bias. If random flag is true, it samples from a normal distribution.
        inputs:
            -- random: A flag which specifies whether the output is picked deterministically or randomly.
        """
        if not random:
            return self.chValue
        else:
            return np.random.normal(self.chValue, self.noise, 1)[0]
