"""elements.py contains the classes for the circuit elements:
capacitors, inductors, and josephson junctions.
"""

from typing import List, Any, Optional, Union, Callable

import torch
import numpy as np

from scipy.special import kn
from torch import Tensor

import SQcircuit.units as unt

from SQcircuit.logs import raise_unit_error, raise_optim_error_if_needed
from SQcircuit.settings import get_optim_mode


class Element:
    """Class that contains general properties of elements."""

    _unit = None
    _error = None
    _value = None

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def error(self) -> float:
        return self._error

    @error.setter
    def error(self, e: float) -> None:
        self._error = e

    @property
    def requires_grad(self) -> bool:
        raise_optim_error_if_needed()

        return self._value.requires_grad

    @requires_grad.setter
    def requires_grad(self, f: bool) -> None:

        raise_optim_error_if_needed()

        self._value.requires_grad = f

    def set_value_with_error(self, mean, error):

        mean_th = torch.as_tensor(mean, dtype=float)
        error_th = torch.as_tensor(error, dtype=float)

        self._value = torch.normal(mean_th, mean_th*error_th/100)

        if not get_optim_mode():
            self._value = float(self._value.detach().cpu().numpy())

    def get_value(self):
        pass

    @staticmethod
    def get_default_id_str(s: str, v: float, u: str) -> str:
        """Get the default string ID for the element.
        Parameters
        ----------
        s:
            The initial string of the id_string.
        v:
            The value of the element as float number.
        u:
            The unit of the element as string.
        """
        assert isinstance(s, str), "The input must have string format."

        return (s + "_{}_{}").format(v, u)


class Capacitor(Element):
    """Class that contains the capacitor properties.
    Parameters
    ----------
    value:
        The value of the capacitor.
    unit:
        The unit of input value. If ``unit`` is "THz", "GHz", and ,etc.,
        the value specifies the charging energy of the capacitor. If ``unit``
        is "fF", "pF", and ,etc., the value specifies the capacitance in
        farad. If ``unit`` is ``None``, the default unit of capacitor is "GHz".
    requires_grad:
        A boolean variable specifies if the autograd should record operation
        on this element.
    Q:
        Quality factor of the dielectric of the capacitor which is one over
        tangent loss. It can be either a float number or a Python function of
        angular frequency.
    error:
        The error of fabrication in percentage.
    id_str:
        ID string for the capacitor.
    """

    def __init__(
        self,
        value: float,
        unit: Optional[str] = None,
        requires_grad: bool = False,
        Q: Union[Any, Callable[[float], float]] = "default",
        error: float = 0,
        id_str: Optional[str] = None,
    ) -> None:

        self.set_value(value, unit, error)
        self.type = type(self)

        if requires_grad:
            self.requires_grad = requires_grad

        if Q == "default":
            self.Q = self._default_Q_cap
        elif isinstance(Q, float) or isinstance(Q, int):
            self.Q = lambda omega: Q
        else:
            self.Q = Q

        if id_str is None:
            self.id_str = self.get_default_id_str("C", value, unit)
        else:
            self.id_str = id_str

    @staticmethod
    def _check_unit_format(u):
        """Check if the unit input has the correct format."""

        if (u not in unt.freq_list and
                u not in unt.farad_list and
                u is not None):
            raise_unit_error()

    @Element.unit.setter
    def unit(self, u: Optional[str]) -> None:

        self._check_unit_format(u)

        if u is None:
            self._unit = unt.get_unit_cap()
        else:
            self._unit = u

    def set_value(self, v: float, u: str, e: float = 0.0) -> None:
        """Set the value for the capacitor.
        Parameters
        ----------
            v:
                The value of the element.
            u:
                The unit of input value.
            e:
                The fabrication error in percentage.
        """
        self.unit = u
        self.error = e

        if self.unit in unt.farad_list:
            mean = v * unt.farad_list[self.unit]
        else:
            E_c = v * unt.freq_list[self.unit] * (2*np.pi*unt.hbar)
            mean = unt.e ** 2 / 2 / E_c

        self.set_value_with_error(mean, e)

    def get_value(self, u: str = "F") -> Union[float, Tensor]:
        """Return the value of the element in specified unit.
        Parameters
        ----------
            u:
                The unit of input value. The default is "F".
        """

        if u in unt.farad_list:
            return self._value / unt.farad_list[u]

        elif u in unt.freq_list:
            E_c = unt.e**2/2/self._value/(2*np.pi*unt.hbar)/unt.freq_list[u]
            return E_c

        else:
            raise_unit_error()

    @staticmethod
    def _default_Q_cap(omega):
        """Default function for capacitor quality factor."""

        return 1e6 * (2 * np.pi * 6e9 / np.abs(omega))**0.7


class VerySmallCap(Capacitor):

    def __init__(self):
        super().__init__(1e-20, "F", Q=None)


class VeryLargeCap(Capacitor):

    def __init__(self):
        super().__init__(1e20, "F", Q=None)


class Inductor(Element):
    """Class that contains the inductor properties.
    Parameters
    ----------
    value:
        The value of the inductor.
    unit:
        The unit of input value. If ``unit`` is "THz", "GHz", and ,etc.,
        the value specifies the inductive energy of the inductor. If ``unit``
        is "fH", "pH", and ,etc., the value specifies the inductance in henry.
        If ``unit`` is ``None``, the default unit of inductor is "GHz".
    requires_grad:
        A boolean variable specifies if the autograd should record operation
        on this element.
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
            requires_grad: bool = False,
            cap: Optional["Capacitor"] = None,
            Q: Union[Any, Callable[[float, float], float]] = "default",
            error: float = 0,
            loops: Optional[List["Loop"]] = None,
            id_str: Optional[str] = None
    ) -> None:

        self.set_value(value, unit, error)
        self.type = type(self)

        if requires_grad:
            self.requires_grad = requires_grad

        if cap is None:
            self.cap = VerySmallCap()
        else:
            self.cap = cap

        if loops is None:
            self.loops = []
        else:
            self.loops = loops

        if Q == "default":
            self.Q = self._default_Q_ind
        elif isinstance(Q, float) or isinstance(Q, int):
            self.Q = lambda omega, T: Q
        else:
            self.Q = Q

        if id_str is None:
            self.id_str = self.get_default_id_str("L", value, unit)
        else:
            self.id_str = id_str

    @staticmethod
    def _check_unit_format(u):
        """Check if the unit input has the correct format."""

        if (u not in unt.freq_list and
                u not in unt.henry_list and
                u is not None):
            raise_unit_error()

    @Element.unit.setter
    def unit(self, u: Optional[str]) -> None:

        self._check_unit_format(u)

        if u is None:
            self._unit = unt.get_unit_ind()
        else:
            self._unit = u

    def set_value(self, v: float, u: str, e: float = 0.0) -> None:
        """Set the value for the element.
        Parameters
        ----------
            v:
                The value of the element.
            u:
                The unit of input value.
            e:
                The fabrication error in percentage.
        """
        self.unit = u
        self.error = e

        if self.unit in unt.henry_list:
            mean = v * unt.henry_list[self.unit]
        else:
            E_l = v * unt.freq_list[self.unit] * (2*np.pi*unt.hbar)
            mean = (unt.Phi0/2/np.pi)**2 / E_l

        self.set_value_with_error(mean, e)

    def get_value(self, u: str = "H") -> float:
        """Return the value of the element in specified unit.
        Parameters
        ----------
            u:
                The unit of input value. The default is "H".
        """

        if u in unt.henry_list:
            return self._value / unt.henry_list[u]

        elif u in unt.freq_list:
            l = self._value
            E_l = (unt.Phi0/2/np.pi)**2/l/(2*np.pi*unt.hbar)/unt.freq_list[u]
            return E_l

        else:
            raise_unit_error()

    @staticmethod
    def _default_Q_ind(omega, T):
        """Default function for inductor quality factor."""

        alpha = unt.hbar * 2 * np.pi * 0.5e9 / (2 * unt.k_B * T)
        beta = unt.hbar * omega / (2 * unt.k_B * T)

        return 500e6*(kn(0, alpha)*np.sinh(alpha))/(kn(0, beta)*np.sinh(beta))

    def get_key(self, edge, B_idx, *_):
        """Return the inductor key.

        Parameters
        ----------
            edge:
                Edge that element is part of.
            B_idx:
                The inductive element index
        """

        return edge, self, B_idx

    def get_cap_for_flux_dist(self, flux_dist):

        if flux_dist == 'all':
            return self.cap.get_value()
        elif flux_dist == "junctions":
            return VeryLargeCap().get_value()
        elif flux_dist == "inductors":
            return VerySmallCap().get_value()


class Junction(Element):
    """Class that contains the Josephson junction properties.
    Parameters
    -----------
    value:
        The value of the Josephson junction.
    unit: str
        The unit of input value. The ``unit`` can be "THz", "GHz", and ,etc.,
        that specifies the junction energy of the inductor. If ``unit`` is
        ``None``, the default unit of junction is "GHz".
    requires_grad:
        A boolean variable specifies if the autograd should record operation
        on this element.
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
        requires_grad: bool = False,
        cap: Optional[str] = None,
        A: float = 1e-7,
        x: float = 3e-06,
        delta: float = 3.4e-4,
        Y: Union[Any, Callable[[float, float], float]] = "default",
        error: float = 0,
        loops: Optional[List["Loop"]] = None,
        id_str: Optional[str] = None,
    ) -> None:

        self.set_value(value, unit, error)
        self.type = type(self)
        self.A = A

        if requires_grad:
            self.requires_grad = requires_grad

        if cap is None:
            self.cap = VerySmallCap()
        else:
            self.cap = cap

        if loops is None:
            self.loops = []
        else:
            self.loops = loops

        if Y == "default":
            self.Y = self._get_default_Y_func(delta, x)
        else:
            self.Y = Y

        if id_str is None:
            self.id_str = self.get_default_id_str("JJ", value, unit)
        else:
            self.id_str = id_str

    @staticmethod
    def _check_unit_format(u):
        """Check if the unit input has the correct format."""

        if u not in unt.freq_list and u is not None:
            raise_unit_error()

    @Element.unit.setter
    def unit(self, u: Optional[str]) -> None:

        self._check_unit_format(u)

        if u is None:
            self._unit = unt.get_unit_JJ()
        else:
            self._unit = u

    def set_value(self, v: float, u: str, e: float = 0.0) -> None:
        """Set the value for the element.
        Parameters
        ----------
            v:
                The value of the element.
            u:
                The unit of input value.
            e:
                The fabrication error in percentage.
        """
        self.unit = u
        self.error = e

        mean = v * unt.freq_list[self.unit] * 2 * np.pi

        self.set_value_with_error(mean, e)

    def get_value(self, u: str = "Hz") -> float:
        """Return the value of the element in specified unit.
        Parameters
        ----------
            u:
                The unit of input value. The default is "Hz".
        """

        if u in unt.freq_list:
            return self._value / unt.freq_list[u]

        else:
            raise_unit_error()

    def get_key(self, edge, B_idx, W_idx, *_):
        """Return the junction key.

        Parameters
        ----------
            edge:
                Edge that element is part of.
            B_idx:
                The inductive element index
            W_idx:
                The JJ index
        """

        return edge, self, B_idx, W_idx

    def get_cap_for_flux_dist(self, flux_dist):

        if flux_dist == 'all':
            return self.cap.get_value()
        elif flux_dist == "junctions":
            return VerySmallCap().get_value()
        elif flux_dist == "inductors":
            return VeryLargeCap().get_value()

    @staticmethod
    def _get_default_Y_func(
        delta: float,
        x: float
    ) -> Callable[[Union[float, Tensor]], float]:

        def _default_Y_junc(
            omega: Union[float, Tensor],
            T: float
        ) -> Union[float, Tensor]:
            """Default function for junction admittance."""

            alpha = unt.hbar * omega / (2 * unt.k_B * T)

            y = np.sqrt(2 / np.pi) * (8 / (delta * 1.6e-19) / (
                    unt.hbar * 2 * np.pi / unt.e ** 2)) \
                * (2 * (delta * 1.6e-19) / unt.hbar / omega) ** 1.5 \
                * x * np.sqrt(alpha) * kn(0, alpha) * np.sinh(alpha)
            return y

        return _default_Y_junc


class Loop:
    """Class that contains the inductive loop properties, closed path of
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

        self.lpValue = value * 2 * np.pi
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

    def value(self, random: bool = False) -> float:
        """Return the value of the external flux. If `random` is `True`, it
        samples from a normal distribution with variance defined by the flux
        noise amplitude.
        Parameters
        ----------
            random:
                A boolean flag which specifies whether the output is
                deterministic or random.
        """
        if not random:
            return self.lpValue
        else:
            return np.random.normal(self.lpValue, self.A, 1)[0]

    def set_flux(self, value: float) -> None:
        """Set the external flux associated to the loop.
        Parameters
        ----------
            value:
                The external flux value
        """
        self.lpValue = value * 2 * np.pi

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


class Charge:
    """Class that contains the charge island properties.
    """

    def __init__(self, value: float = 0, A: float = 1e-4) -> None:
        """
       inputs:
            -- value: The value of the offset.
            -- noise: The amplitude of the charge noise.
        """
        self.chValue = value
        self.A = A

    def value(self, random: bool = False) -> float:
        """
        returns the value of charge bias. If random flag is true, it samples
        from a normal distribution.
        inputs:
            -- random: A flag which specifies whether the output is picked
                deterministically or randomly.
        """
        if not random:
            return self.chValue
        else:
            return np.random.normal(self.chValue, self.noise, 1)[0]

    def setOffset(self, value: float) -> None:
        self.chValue = value

    def setNoise(self, A: float) -> None:
        self.A = A