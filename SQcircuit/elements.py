"""
elements.py contains the classes for the circuit elements:
capacitors, inductors, and josephson junctions.
"""

import SQcircuit.physParam as phPar
import numpy as np
from scipy.special import kn


class Capacitor:
    """
    Class that contains the capacitor properties.

    Parameters
    ----------
    value: float
        The value of the capacitor.
    unit: str
        The unit of input value. If ``unit`` is "THz", "GHz", and ,etc., the value specifies the
        charging energy of the capacitor. If ``unit`` is "fF", "pF", and ,etc., the value specifies
        the capacitance in farad. If ``unit`` is ``None``, the default unit of capacitor is "GHz".
    Q:
        Quality factor of the dielectric of the capacitor which is one over tangent loss. It can be either
        a float number or a Python function of angular frequency.
    error: float
        The error in fabrication as a percentage.
    idStr: str
        ID string for the capacitor.
    """

    def __init__(self, value, unit=None, Q="default", error=0, idStr=None):

        if unit not in phPar.freqList and unit not in phPar.faradList and unit is not None:
            error = "The input unit for the capacitor is not correct. Look at the documentation for the correct input " \
                    "format."
            raise ValueError(error)

        self.cValue = value
        self.error = error
        self.type = type(self)

        if unit is None:
            self.unit = phPar.capU
        else:
            self.unit = unit

        if Q == "default":
            self.Q = lambda omega: 1e6 * (2 * np.pi * 6e9 / np.abs(omega)) ** 0.7
        elif isinstance(Q, float) or isinstance(Q, int):
            self.Q = lambda omega: Q
        else:
            self.Q = Q

        if idStr is None:
            self.idStr = "C_{}_{}".format(value, self.unit)
        else:
            self.idStr = idStr

    def value(self, random: bool = False):
        """
        Return the value of the capacitor in farad units. If `random` is `True`, it
        samples from a normal distribution with variance defined by the fabrication error.

        Parameters
        ----------
            random: bool
                A boolean flag which specifies whether the output is deterministic or random.
        """
        if self.unit in phPar.faradList:
            cMean = self.cValue * phPar.faradList[self.unit]
        else:
            E_c = self.cValue * phPar.freqList[self.unit] * (2 * np.pi * phPar.hbar)
            cMean = phPar.e ** 2 / 2 / E_c

        if not random:
            return cMean
        else:
            return np.random.normal(cMean, cMean * self.error / 100, 1)[0]

    def energy(self):
        """
        Return the charging energy of the capacitor in frequency unit of SQcircuit (gigahertz
        by default).
        """
        if self.unit in phPar.freqList:
            return self.cValue * phPar.freqList[self.unit]/phPar.freq
        else:
            c = self.cValue * phPar.faradList[self.unit]
            return phPar.e ** 2 / 2 / c / (2 * np.pi * phPar.hbar)/phPar.freq


class Inductor:
    """
    Class that contains the inductor properties.

    Parameters
    ----------
    value: float
        The value of the inductor.
    unit: str
        The unit of input value. If ``unit`` is "THz", "GHz", and ,etc., the value specifies the
        inductive energy of the inductor. If ``unit`` is "fH", "pH", and ,etc., the value specifies
        the inductance in henry. If ``unit`` is ``None``, the default unit of inductor is "GHz".
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
    idStr: str
        ID string for the inductor.
    """

    def __init__(self, value, unit=None, cap=None, Q="default", error=0, loops=None, idStr=None):

        if unit not in phPar.freqList and unit not in phPar.henryList and unit is not None:
            error = "The input unit for the inductor is not correct. Look at the documentation for the correct input " \
                    "format."
            raise ValueError(error)

        self.lValue = value
        self.error = error
        self.type = type(self)
        self.idStr = idStr

        if unit is None:
            self.unit = phPar.indU
        else:
            self.unit = unit

        if cap is None:
            self.cap = Capacitor(1e-20, "F", Q=None)
        else:
            self.cap = cap

        if loops is None:
            self.loops = []
        else:
            self.loops = loops

        def qInd(omega, T):
            alpha = phPar.hbar * 2 * np.pi * 0.5e9 / (2 * phPar.k_B * T)
            beta = phPar.hbar * omega / (2 * phPar.k_B * T)

            return 500e6 * (kn(0, alpha) * np.sinh(alpha)) / (kn(0, beta) * np.sinh(beta))

        if Q == "default":
            self.Q = qInd
        elif isinstance(Q, float) or isinstance(Q, int):
            self.Q = lambda omega, T: Q
        else:
            self.Q = Q

        if idStr is None:
            self.idStr = "L_{}_{}".format(value, self.unit)
        else:
            self.idStr = idStr

    def value(self, random: bool = False):
        """
        Return the value of the inductor in henry units. If `random` is `True`, it
        samples from a normal distribution with variance defined by the fabrication error.

        Parameters
        ----------
            random: bool
                A boolean flag which specifies whether the output is deterministic or random.
        """
        if self.unit in phPar.henryList:
            lMean = self.lValue * phPar.henryList[self.unit]
        else:
            E_l = self.lValue * phPar.freqList[self.unit] * (2 * np.pi * phPar.hbar)
            lMean = (phPar.Phi0 / 2 / np.pi) ** 2 / E_l

        if not random:
            return lMean
        else:
            return np.random.normal(lMean, lMean * self.error / 100, 1)[0]

    def energy(self):
        """
        Return the inductive energy of the capacitor in frequency unit of SQcircuit (gigahertz
        by default).
        """
        if self.unit in phPar.freqList:
            return self.lValue * phPar.freqList[self.unit]/phPar.freq
        else:
            l = self.lValue * phPar.henryList[self.unit]
            return (phPar.Phi0 / 2 / np.pi) ** 2 / l / (2 * np.pi * phPar.hbar)/phPar.freq


class Junction:
    """
    Class that contains the Josephson junction properties.

    Parameters
    -----------
    value: float
        The value of the Josephson junction.
    unit: str
        The unit of input value. The ``unit`` can be "THz", "GHz", and ,etc., that specifies the
        junction energy of the inductor. If ``unit`` is ``None``, the default unit of junction is "GHz".
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
    idStr: str
        ID string for the junction.
    """

    def __init__(self, value, unit=None, cap=None, A=1e-7, x=3e-06, delta=3.4e-4,
                 Y="default", error=0, loops=None, idStr=None):

        if unit not in phPar.freqList and unit is not None:
            error = "The input unit for the Josephson Junction is not correct. Look at the documentation for the" \
                    "correct input format."
            raise ValueError(error)

        self.jValue = value
        self.error = error
        self.type = type(self)
        self.A = A
        self.idStr = idStr

        if unit is None:
            self.unit = phPar.junU
        else:
            self.unit = unit

        if cap is None:
            self.cap = Capacitor(1e-20, "F", Q=None)
        else:
            self.cap = cap

        if loops is None:
            self.loops = []
        else:
            self.loops = loops

        def yQP(omega, T):
            alpha = phPar.hbar * omega / (2 * phPar.k_B * T)
            y = np.sqrt(2 / np.pi) * (8 / (delta * 1.6e-19) / (phPar.hbar * 2 * np.pi / phPar.e ** 2)) \
                * (2 * (delta * 1.6e-19) / phPar.hbar / omega) ** 1.5 \
                * x * np.sqrt(alpha) * kn(0, alpha) * np.sinh(alpha)
            return y

        if Y == "default":
            self.Y = yQP
        else:
            self.Y = Y

        if idStr is None:
            self.idStr = "JJ_{}_{}".format(value, self.unit)
        else:
            self.idStr = idStr

    def value(self, random: bool = False):
        """
        Return the value of the Josephson Junction in angular frequency. If `random` is `True`, it samples
        from a normal distribution with variance defined by the fabrication error.

        Parameters
        ----------
            random: bool
                A boolean flag which specifies whether the output is deterministic or random.
        """
        jMean = self.jValue * phPar.freqList[self.unit] * 2 * np.pi

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
        idStr: str
            ID string for the loop.
    """

    def __init__(self, value=0, A=1e-6, idStr=None):

        self.lpValue = value * 2 * np.pi
        self.A = A * 2 * np.pi
        # indices of inductive elements.
        self.indices = []
        # k1 matrix related to this specific loop
        self.K1 = []

        if idStr is None:
            self.idStr = "loop"
        else:
            self.idStr = idStr

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

    def set_flux(self, value):
        """
        Set the external flux associated to the loop.

        Parameters
        ----------
            value: float
                The external flux value
        """
        self.lpValue = value * 2 * np.pi

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
    class that contains the charge island properties.
    """

    def __init__(self, value=0, A=1e-4):
        """
       inputs:
            -- value: The value of the offset.
            -- noise: The amplitude of the charge noise.
        """
        self.chValue = value
        self.A = A

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

    def setOffset(self, value):
        self.chValue = value

    def setNoise(self, A):
        self.A = A

