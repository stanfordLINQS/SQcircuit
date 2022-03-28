"""
elements.py contains the classes for the circuit elements:
capacitors, inductors, and josephson junctions.
"""

from SQcircuit.units import *
import numpy as np
from scipy.special import kn


class Capacitor:
    """
    Class that contains the capacitor properties.

    Parameters
    ----------
    value: float
        The value of the capacitor.
    unit: string
        The unit of input value. If `unit` is "THz", "GHz", and ,etc., the value specifies the
        charging energy of the capacitor. If `unit` is "fF", "pF", and ,etc., the value specifies
        the capacitance in farad.
    Q:
        Quality factor of the dielectric of the capacitor which is one over tangent loss. It can be either
        a float number or a Python function of angular frequency.
    error: float
        The error in fabrication as a percentage.
    """

    def __init__(self, value=1e-20, cUnit="F", Q="default", error=0):

        if cUnit not in unit.freqList and cUnit not in unit.faradList:
            error = "The input unit for the inductor is not correct. Look at the documentation for the correct input " \
                    "format."
            raise ValueError(error)

        self.cValue = value
        self.cUnit = cUnit
        self.error = error
        self.type = type(self)

        if Q == "default":
            self.Q = lambda omega: 1e6 * (2 * np.pi * 6e9 / np.abs(omega)) ** 0.7
        elif isinstance(Q, float):
            self.Q = lambda omega: Q
        else:
            self.Q = Q

    def value(self, random: bool = False):
        """
        Return the value of the capacitor in farad units. If `random` is `True`, it
        samples from a normal distribution with variance defined by the fabrication error.

        Parameters
        ----------
            random: bool
                A boolean flag which specifies whether the output is deterministic or random.
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
        Return the charging energy of the capacitor in frequency unit of SQcircuit (gigahertz
        by default).
        """
        if self.cUnit in unit.freqList:
            return self.cValue * unit.freqList[self.cUnit]
        else:
            c = self.cValue * unit.faradList[self.cUnit]
            return unit.e ** 2 / 2 / c / (2 * np.pi * unit.hbar)


class Inductor:
    """
    Class that contains the inductor properties.

    Parameters
    ----------
    value: float
        The value of the inductor.
    unit: string
        The unit of input value. If `unit` is "THz", "GHz", and ,etc., the value specifies the
        inductive energy of the inductor. If `unit` is "fH", "pH", and ,etc., the value specifies
        the inductance in henry.
    loops: list
        List of loops in which the inductor resides.
    cap: SQcircuit.Capacitor
        Capacitor associated to the inductor, necessary for correct time-dependent external fluxes
        scheme.
    Q:
        Quality factor of the inductor needed for inductive loss calculation. It can be either
        a float number or a Python function of angular frequency and temperature.
    error: float
        The error in fabrication as a percentage.
    """

    def __init__(self, value, lUnit, cap=Capacitor(Q=None), Q="default", error=0, loops=[]):

        if loops is None:
            loops = []
        if lUnit not in unit.freqList and lUnit not in unit.henryList:
            error = "The input unit for the inductor is not correct. Look at the documentation for the correct input " \
                    "format."
            raise ValueError(error)

        self.lValue = value
        self.lUnit = lUnit
        self.cap = cap
        self.error = error
        self.type = type(self)
        self.loops = loops

        def qInd(omega, T):
            alpha = unit.hbar * 2 * np.pi * 0.5e9 / (2 * unit.k_B * T)
            beta = unit.hbar * omega / (2 * unit.k_B * T)

            return 500e6 * (kn(0, alpha) * np.sinh(alpha)) / (kn(0, beta) * np.sinh(beta))

        if Q == "default":
            self.Q = qInd
        elif isinstance(Q, float):
            self.Q = lambda omega, T: Q
        else:
            self.Q = Q

    def value(self, random: bool = False):
        """
        Return the value of the inductor in henry units. If `random` is `True`, it
        samples from a normal distribution with variance defined by the fabrication error.

        Parameters
        ----------
            random: bool
                A boolean flag which specifies whether the output is deterministic or random.
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
        Return the inductive energy of the capacitor in frequency unit of SQcircuit (gigahertz
        by default).
        """
        if self.lUnit in unit.freqList:
            return self.lValue * unit.freqList[self.lUnit]
        else:
            l = self.lValue * unit.henryList[self.lUnit]
            return (unit.Phi0 / 2 / np.pi) ** 2 / l / (2 * np.pi * unit.hbar)


class Junction:
    """
    Class that contains the Josephson junction properties.

    Parameters
    ----------
    value: float
        The value of the Josephson junction.
    unit: string
        The unit of input value. The `unit` can be "THz", "GHz", and ,etc., that specifies the
        junction energy of the inductor.
    loops: list
        List of loops in which the Josephson junction reside.
    cap: SQcircuit.Capacitor
        Capacitor associated to the josephson junction, necessary for the correct time-dependent
        external fluxes scheme.
    A: float
        Normalized noise amplitude related to critical current noise.
    x: float
        Quasiparticle density
    delta: float
        Superconducting gap
    Y:
        Real part of admittance.
    error: float
        The error in fabrication as a percentage.
    """

    def __init__(self, value, jUnit, cap=Capacitor(Q=None), A=1e-7, x=3e-06, delta=3.4e-4,
                 Y="default", error=0, loops=[]):

        if loops is None:
            loops = []
        if jUnit not in unit.freqList:
            error = "The input unit for the Josephson Junction is not correct. Look at the documentation for the" \
                    "correct input format."
            raise ValueError(error)

        self.jValue = value
        self.jUnit = jUnit
        self.cap = cap
        self.error = error
        self.type = type(self)
        self.loops = loops
        self.A = A

        def yQP(omega, T):
            alpha = unit.hbar * omega / (2 * unit.k_B * T)
            y = np.sqrt(2 / np.pi) * (8 / (delta * 1.6e-19) / (unit.hbar * 2 * np.pi / unit.e ** 2)) \
                * (2 * (delta * 1.6e-19) / unit.hbar / omega) ** 1.5 \
                * x * np.sqrt(alpha) * kn(0, alpha) * np.sinh(alpha)
            return y

        if Y == "default":
            self.Y = yQP
        else:
            self.Y = Y

    def value(self, random: bool = False):
        """
        Return the value of the Josephson Junction in angular frequency. If `random` is `True`, it samples
        from a normal distribution with variance defined by the fabrication error.

        Parameters
        ----------
            random: bool
                A boolean flag which specifies whether the output is deterministic or random.
        """
        jMean = self.jValue * unit.freqList[self.jUnit] * 2 * np.pi

        if not random:
            return jMean
        else:
            return np.random.normal(jMean, jMean * self.error / 100, 1)[0]


class Loop:
    """
    Class that contains the inductive loop properties, closed path of inductive elements.

    Parameters
    ----------
        value: float
            Value of the external flux at the loop.
        A: float
            Normalized noise amplitude related to flux noise.
    """

    def __init__(self, value=0, A=1e-6):

        self.lpValue = value
        self.A = A
        # indices of inductive elements.
        self.indices = []
        # k1 matrix related to this specific loop
        self.K1 = []

    def reset(self):
        self.K1 = []
        self.indices = []

    def value(self, random: bool = False):
        """
        Return the value of the external flux. If `random` is `True`, it samples from a normal distribution
        with variance defined by the flux noise amplitude.

        Parameters
        ----------
            random: bool
                A boolean flag which specifies whether the output is deterministic or random.
        """
        if not random:
            return self.lpValue
        else:
            return np.random.normal(self.lpValue, self.A, 1)[0]

    def setFlux(self, value):
        """
        Set the external flux associated to the loop.

        Parameters
        ----------
            value: float
                The external flux value
        """
        self.lpValue = value

    def addIndex(self, index):
        self.indices.append(index)

    def addK1(self, w):
        self.K1.append(w)

    def getP(self):
        K1 = np.array(self.K1)
        a = np.zeros_like(K1)
        select = np.sum(K1 != a, axis=0) != 0
        # eliminate the zero columns
        K1 = K1[:, select]
        if K1.shape[0] == K1.shape[1]:
            K1 = K1[:, 0:-1]
        b = np.zeros((1, K1.shape[0]))
        b[0, 0] = 1
        p = np.linalg.inv(np.concatenate((b, K1.T), axis=0)) @ b.T
        return p.T

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