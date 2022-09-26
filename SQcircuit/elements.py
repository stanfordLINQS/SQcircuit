"""
elements.py contains the classes for the circuit elements:
capacitors, inductors, and josephson junctions.
"""

from typing import List, Any, Optional, Union, Callable

import numpy as np

from scipy.special import kn

import SQcircuit.units as unt
import SQcircuit.functions as sqf

class Element:
    def __init__(self, value: float = 0):
        self.value = value

    def set_value(self,
                     value: float,
                     unit: Optional[str] = None):
        self.value = sqf.abs(sqf.cast(value))
        if unit is not None:
            self.unit = unit

    def get_unit_scale(self):
        pass

    def get_value(self, *args):
        pass

class Capacitor(Element):
    """
    Class that contains the capacitor properties.
    Parameters
    ----------
    value:
        The value of the capacitor.
    unit:
        The unit of input value. If ``unit`` is "THz", "GHz", and ,etc.,
        the value specifies the charging energy of the capacitor. If ``unit``
        is "fF", "pF", and ,etc., the value specifies the capacitance in
        farad. If ``unit`` is ``None``, the default unit of capacitor is "GHz".
    Q:
        Quality factor of the dielectric of the capacitor which is one over
        tangent loss. It can be either a float number or a Python function of
        angular frequency.
    error:
        The error in fabrication as a percentage.
    id_str:
        ID string for the capacitor.
    """

    def __init__(
        self,
        value: float,
        unit: Optional[str] = None,
        Q: Union[Any, Callable[[float], float]] = "default",
        error: float = 0,
        id_str: Optional[str] = None,
    ) -> None:

        Element.__init__(self)
        self.set_value(value, unit)
        self.error = error
        self.type = type(self)

        if Q == "default":
            self.Q = lambda omega: 1e6 * (
                    2 * np.pi * 6e9 / np.abs(omega)) ** 0.7
        elif isinstance(Q, float) or isinstance(Q, int):
            self.Q = lambda omega: Q
        else:
            self.Q = Q

        if id_str is None:
            self.id_str = f"C_{value}_{self.unit}"
        else:
            self.id_str = id_str

    def set_value(self,
                     value: float,
                     unit: Optional[str] = None):
        if (unit not in unt.freq_list and
                unit not in unt.farad_list and
                unit is not None):
            error = "The input unit for the capacitor is not correct. " \
                    "Look at the documentation for the correct input format."
            raise ValueError(error)
        if unit is None:
            unit = unt.get_unit_cap()
        Element.set_value(self, value, unit)

    def get_value(self,
                  element_units: bool = False,
                  random: bool = False) -> float:
        """
        Return the value of the capacitor, by default in units of farads. If
        `random` is `True`, it samples from a normal distribution with
        variance defined by the fabrication error.
        Parameters
        ----------
            element_units:
                A boolean flag that determines whether to keep units
                in element's preset units (True) or return in farads (False).
            random:
                A boolean flag which specifies whether the output is
                deterministic or random.
        """
        if self.unit in unt.farad_list:
            cMean = self.value
        else:
            E_c = self.value * (2 * np.pi * unt.hbar)
            cMean = unt.e ** 2 / 2 / E_c
        if not element_units:
            cMean *= unt.farad_list[self.unit]

        if not random:
            return cMean
        else:
            return sqf.normal(cMean, cMean * self.error / 100)

    def get_unit_scale(self):
        return unt.farad_list[self.unit]

    def energy(self) -> float:
        """
        Return the charging energy of the capacitor in frequency unit of
        SQcircuit (gigahertz by default).
        """
        if self.unit in unt.freq_list:
            return self.value * unt.freq_list[
                self.unit] / unt.get_unit_freq()
        else:
            c = self.value * unt.farad_list[self.unit]
            return unt.e ** 2 / 2 / c / (
                    2 * np.pi * unt.hbar) / unt.get_unit_freq()


class VerySmallCap(Capacitor):

    def __init__(self):
        super().__init__(1e-20, "F", Q=None)


class VeryLargeCap(Capacitor):

    def __init__(self):
        super().__init__(1e20, "F", Q=None)


class Inductor(Element):
    """
    Class that contains the inductor properties.
    Parameters
    ----------
    value:
        The value of the inductor.
    unit:
        The unit of input value. If ``unit`` is "THz", "GHz", and ,etc.,
        the value specifies the inductive energy of the inductor. If ``unit``
        is "fH", "pH", and ,etc., the value specifies the inductance in henry.
        If ``unit`` is ``None``, the default unit of inductor is "GHz".
    loops:
        List of loops in which the inductor resides.
    cap:
        Capacitor associated to the inductor, necessary for correct
        time-dependent external fluxes scheme.
    Q:
        Quality factor of the inductor needed for inductive loss calculation.
        It can be either a float number or a Python function of angular
        frequency and temperature.
    error:
        The error in fabrication as a percentage.
    id_str:
        ID string for the inductor.
    """

    def __init__(
            self,
            value: float,
            unit: str = None,
            cap: Optional["Capacitor"] = None,
            Q: Union[Any, Callable[[float, float], float]] = "default",
            error: float = 0,
            loops: Optional[List["Loop"]] = None,
            id_str: Optional[str] = None
    ) -> None:

        Element.__init__(self)
        self.set_value(value, unit)
        self.error = error
        self.type = type(self)
        self.id_str = id_str

        if cap is None:
            self.cap = VerySmallCap()
        else:
            self.cap = cap

        if loops is None:
            self.loops = []
        else:
            self.loops = loops

        def qInd(omega, T):
            alpha = unt.hbar * 2 * np.pi * 0.5e9 / (2 * unt.k_B * T)
            beta = unt.hbar * omega / (2 * unt.k_B * T)

            return 500e6 * (kn(0, alpha) * np.sinh(alpha)) / (
                    kn(0, beta) * np.sinh(beta))

        if Q == "default":
            self.Q = qInd
        elif isinstance(Q, float) or isinstance(Q, int):
            self.Q = lambda omega, T: Q
        else:
            self.Q = Q

        if id_str is None:
            self.id_str = "L_{}_{}".format(value, self.unit)
        else:
            self.id_str = id_str

    def set_value(self,
                  value: float,
                  unit: Optional[str] = None):
        if (unit not in unt.freq_list and
                unit not in unt.henry_list and
                unit is not None):
            error = "The input unit for the inductor is not correct. " \
                    "Look at the documentation for the correct input format."
            raise ValueError(error)

        if unit is None:
            unit = unt.get_unit_ind()
        Element.set_value(self, value, unit)


    def get_value(self,
                  element_units: bool = False,
                  angular_frequency: bool = True,
                  random: bool = False) -> float:
        """
        Return the value of the inductor, by default in units of henries.
        If `random` is `True`, it samples from a normal distribution
        with variance defined by the fabrication error.
        Parameters
        ----------
            element_units:
                A boolean flag that determines whether to keep units
                in element's preset units (True) or return in henries (False).
            random:
                A boolean flag which specifies whether the output is
                deterministic or random.
        """
        if self.unit in unt.henry_list:
            lMean = self.value
            if not element_units:
                lMean *= unt.henry_list[self.unit]
        else:
            E_l = self.value * unt.hbar
            if angular_frequency:
                E_l *= (2 * np.pi)
            if not element_units:
                E_l *= unt.freq_list[self.unit]
            lMean = (unt.Phi0 / 2 / np.pi) ** 2 / E_l

        if not random:
            return lMean
        else:
            return sqf.normal(lMean, lMean * self.error / 100)

    def get_unit_scale(self):
        return unt.henry_list[self.unit]

    def energy(self) -> float:
        """
        Return the inductive energy of the capacitor in frequency unit of
        SQcircuit (gigahertz by default).
        """
        if self.unit in unt.freq_list:
            return self.value * unt.freq_list[
                self.unit] / unt.get_unit_freq()
        else:
            l = self.value * unt.henry_list[self.unit]
            return (unt.Phi0 / 2 / np.pi) ** 2 / l / (
                    2 * np.pi * unt.hbar) / unt.get_unit_freq()


class Junction(Element):
    """
    Class that contains the Josephson junction properties.
    Parameters
    -----------
    value:
        The value of the Josephson junction.
    unit: str
        The unit of input value. The ``unit`` can be "THz", "GHz", and ,etc.,
        that specifies the junction energy of the inductor. If ``unit`` is
        ``None``, the default unit of junction is "GHz".
    loops:
        List of loops in which the Josephson junction reside.
    cap:
        Capacitor associated to the josephson junction, necessary for the
        correct time-dependent external fluxes scheme.
    A:
        Normalized noise amplitude related to critical current noise.
    x:
        Quasiparticle density
    delta:
        Superconducting gap
    Y:
        Real part of admittance.
    error:
        The error in fabrication as a percentage.
    id_str:
        ID string for the junction.
    """

    def __init__(
        self,
        value: float,
        unit: Optional[str] = None,
        cap: Optional[str] = None,
        A: float = 1e-7,
        x: float = 3e-06,
        delta: float = 3.4e-4,
        Y: Union[Any, Callable[[float, float], float]] = "default",
        error: float = 0,
        loops: Optional[List["Loop"]] = None,
        id_str: Optional[str] = None,
    ) -> None:

        Element.__init__(self)
        self.set_value(value, unit)
        self.error = error
        self.type = type(self)
        self.A = A
        self.id_str = id_str

        if cap is None:
            self.cap = VerySmallCap()
        else:
            self.cap = cap

        if loops is None:
            self.loops = []
        else:
            self.loops = loops

        def yQP(omega, T):
            alpha = unt.hbar * omega / (2 * unt.k_B * T)
            y = np.sqrt(2 / np.pi) * (8 / (delta * 1.6e-19) / (
                    unt.hbar * 2 * np.pi / unt.e ** 2)) \
                * (2 * (delta * 1.6e-19) / unt.hbar / omega) ** 1.5 \
                * x * np.sqrt(alpha) * kn(0, alpha) * np.sinh(alpha)
            return y

        if Y == "default":
            self.Y = yQP
        else:
            self.Y = Y

        if id_str is None:
            self.id_str = "JJ_{}_{}".format(value, self.unit)
        else:
            self.id_str = id_str

    def set_value(self,
                     value: float,
                     unit: Optional[str] = None):
        if (unit not in unt.freq_list and
                unit is not None):
            error = "The input unit for the Josephson Junction is not " \
                    "correct. Look at the documentation for the correct " \
                    "input format."
            raise ValueError(error)

        if unit is None:
            unit = unt.get_unit_JJ()
        self.unit = unit
        Element.set_value(self, value / (2 * np.pi), unit)

    def get_value(self,
                  element_units: bool = False,
                  angular_frequency: bool = True,
                  random: bool = False) -> float:
        """
        Return the value of the Josephson junction in terms of angular
        frequency, by default in units of hertz. If `random` is `True`,
        it samples from a normal distribution with variance defined by
        the fabrication error.
        Parameters
        ----------
            element_units:
                A boolean flag that determines whether to keep units
                in element's preset units (True) or return in Hz (False).
            random:
                A boolean flag which specifies whether the output
                is deterministic or random.
        """
        jMean = self.value
        if angular_frequency:
            jMean *= (2 * np.pi)
        if not element_units:
            jMean *= unt.freq_list[self.unit]

        if not random:
            return jMean
        else:
            return sqf.normal(jMean, jMean * self.error / 100)

    def get_unit_scale(self):
        return unt.freq_list[self.unit]


class Loop(Element):
    """
    Class that contains the inductive loop properties, closed path of
    inductive elements.
    Parameters
    ----------
        value:
            Value of the external flux at the loop.
        A:
            Normalized noise amplitude related to flux noise.
        id_str:
            ID string for the loop.
    """

    def __init__(
        self,
        value: float = 0,
        A: float = 1e-6,
        id_str: Optional[str] = None
    ) -> None:

        lpValue = value * 2 * np.pi
        Element.__init__(self)
        Element.set_value(self, lpValue)
        self.A = A * 2 * np.pi
        # indices of inductive elements.
        self.indices = []
        # k1 matrix related to this specific loop
        self.K1 = []

        if id_str is None:
            self.id_str = "loop"
        else:
            self.id_str = id_str

    def reset(self) -> None:
        self.K1 = []
        self.indices = []

    def get_value(self, random: bool = False) -> float:
        """
        Return the value of the external flux. If `random` is `True`, it
        samples from a normal distribution with variance defined by the flux
        noise amplitude.
        Parameters
        ----------
            random:
                A boolean flag which specifies whether the output is
                deterministic or random.
        """
        if not random:
            return self.value
        else:
            return sqf.normal(self.value, self.A)

    def set_flux(self, value: float) -> None:
        """
        Set the external flux associated to the loop.
        Parameters
        ----------
            value:
                The external flux value
        """
        self.value = value * 2 * np.pi

    def add_index(self, index):
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


class Charge(Element):
    """
    class that contains the charge island properties.
    """

    def __init__(self, value: float = 0, A: float = 1e-4) -> None:
        """
       inputs:
            -- value: The value of the offset.
            -- noise: The amplitude of the charge noise.
        """
        Element.__init__(self)
        self.value = value
        self.A = A

    def get_value(self, random: bool = False) -> float:
        """
        returns the value of charge bias. If random flag is True, it samples
        from a normal distribution.
        inputs:
            -- random: A flag which specifies whether the output is picked
                deterministically or randomly.
        """
        if not random:
            return self.value
        else:
            return np.random.normal(self.value, self.noise, 1)[0]

    def setOffset(self, value: float) -> None:
        self.value = value

    def setNoise(self, A: float) -> None:
        self.A = A