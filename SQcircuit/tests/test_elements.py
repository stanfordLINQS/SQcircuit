"""test_elements contains the test cases for the SQcircuit ``elements.py``
functionalities.
"""
import pytest
import numpy as np
from typing import Union

import torch
from torch import Tensor

import SQcircuit as sq

from SQcircuit.elements import Capacitor, Inductor, Junction
from SQcircuit.exceptions import UNIT_ERROR, ModeError
import SQcircuit.functions as sqf
from SQcircuit.settings import set_optim_mode, get_optim_mode, SUPPORTED_ENGINES


def float_torch_to_python(x: Tensor) -> float:
    """Helper function to transform the torch float variable to python float
    variable."""

    return float(x.detach().cpu().numpy())


def conditional_to_tensor(x: float) -> Union[float, Tensor]:
    if get_optim_mode():
        return torch.tensor(x, dtype=torch.float64)
    
    return x


###############################################################################
# Capacitor Tests
###############################################################################


def test_capacitor_error_massages():
    with pytest.raises(ValueError, match=UNIT_ERROR):
        Capacitor(10, 'H')


def test_capacitor_energy():
    cap = Capacitor(10, 'GHz')
    assert cap.get_value('GHz') == 10

    assert cap.get_value('MHz') == 10 * 1000

    val = cap.get_value()
    cap2 = Capacitor(val, 'F')
    assert cap2.get_value('GHz') == 10


def test_capacitor_q():
    for engine in SUPPORTED_ENGINES:
        sq.set_engine(engine)

        # Test constant Q
        cap = Capacitor(10, 'GHz', Q=1)
        assert cap.Q(conditional_to_tensor(-1)) == 1
        assert cap.Q(conditional_to_tensor(10)) == 1
        assert cap.Q(conditional_to_tensor(0)) == 1

        cap = Capacitor(10, 'GHz', Q=1e7)
        assert cap.Q(conditional_to_tensor(-1)) == 1e7
        assert cap.Q(conditional_to_tensor(10)) == 1e7
        assert cap.Q(conditional_to_tensor(0)) == 1e7

        # Test custom Q function
        Q = lambda omega: omega ** 2
        cap = Capacitor(10, 'GHz', Q=Q)
        assert cap.Q(conditional_to_tensor(2)) == 4

        # Test default Q
        cap = Capacitor(10, 'GHz', Q='default')
        assert np.isclose(sqf.to_numpy(cap.Q(conditional_to_tensor(10e9))),
                          2531813.9189372803)
        assert np.isclose(sqf.to_numpy(cap.Q(conditional_to_tensor(6e9 * 2 * np.pi))),
                          1e6)

        cap = sq.Capacitor(3.4, 'GHz', Q='default')
        assert np.isclose(sqf.to_numpy(cap.Q(conditional_to_tensor(0.56e9))),
                          19041456.468334354)


def test_capacitor_unit():
    cap = Capacitor(10)
    val = cap.get_value()
    assert np.isclose(val, 2e-15)

    sq.set_unit_cap('F')
    cap = Capacitor(val)
    assert cap.get_value('GHz') == 10
    # this is global, so need to set back
    sq.set_unit_cap('GHz')


def test_capacitor_grad():
    sq.set_engine('NumPy')

    # First check error massages
    with pytest.raises(ModeError):
        Capacitor(10, requires_grad=True)

    with pytest.raises(ModeError):
        assert not Capacitor(10).requires_grad

    cap_value_no_grad = Capacitor(10).get_value()

    set_optim_mode(True)

    cap_value_with_grad = Capacitor(10, requires_grad=True).get_value()

    assert cap_value_no_grad == float_torch_to_python(cap_value_with_grad)

    set_optim_mode(False)


###############################################################################
# Inductor Tests
###############################################################################


def test_inductor_error_massages():
    with pytest.raises(ValueError, match=UNIT_ERROR):
        Inductor(10, 'F')


def test_inductor_energy():
    ind = Inductor(10, 'GHz')
    assert ind.get_value('GHz') == 10

    val = ind.get_value()
    ind2 = Inductor(val, 'H')
    assert ind2.get_value('GHz') == 10


def test_inductor_q():
    for engine in SUPPORTED_ENGINES:
        sq.set_engine(engine)

        # Test constant Q
        ind = Inductor(10, 'GHz', Q=1)
        assert ind.Q(conditional_to_tensor(-1), 0) == 1
        assert ind.Q(conditional_to_tensor(10), 12) == 1
        assert ind.Q(conditional_to_tensor(0), 11) == 1

        ind = Inductor(10, 'GHz', Q=1e7)
        assert ind.Q(conditional_to_tensor(-1), 5) == 1e7
        assert ind.Q(conditional_to_tensor(10), 6) == 1e7
        assert ind.Q(conditional_to_tensor(0), 1) == 1e7

        # Test custom function Q
        Q = lambda omega, T: omega ** 2
        ind = Inductor(10, 'GHz', Q=Q)
        assert ind.Q(conditional_to_tensor(2), 10) == 4

        # Test default Q
        ind = Inductor(10, 'GHz')
        assert np.isclose(sqf.to_numpy(ind.Q(conditional_to_tensor(2 * np.pi * 0.5e9), 10)),
                          500e6)
        assert np.isclose(sqf.to_numpy(ind.Q(conditional_to_tensor(2 * np.pi * 0.5e9), 1)),
                          500e6)

        ind = Inductor(10, 'GHz', Q='default')
        assert np.isclose(sqf.to_numpy(ind.Q(conditional_to_tensor(100), 0.015)),
                          559912673679772.0)


def test_inductor_unit():
    ind = Inductor(10)
    val = ind.get_value()
    assert np.isclose(val, 1.6346e-8)

    sq.set_unit_ind('H')
    ind = Inductor(val)
    assert ind.get_value('GHz') == 10
    # this is global, so need to set back
    sq.set_unit_ind('GHz')


def test_inductor_grad():
    sq.set_engine('NumPy')

    # First check error massages
    with pytest.raises(ModeError):
        Inductor(10, requires_grad=True)

    with pytest.raises(ModeError):
        assert not Inductor(10).requires_grad

    ind_value_no_grad = Inductor(10).get_value()

    set_optim_mode(True)

    ind_value_with_grad = Inductor(10, requires_grad=True).get_value()

    assert ind_value_no_grad == float_torch_to_python(ind_value_with_grad)

    set_optim_mode(False)

###############################################################################
# Josephson Junction Tests
###############################################################################


def test_junction_error_massages():
    with pytest.raises(ValueError, match=UNIT_ERROR):
        Junction(10, 'F')


def test_junction_y():
    for engine in SUPPORTED_ENGINES:
        sq.set_engine(engine)

        # Test custom Y
        y_func = lambda omega, T: omega * T
        JJ = Junction(10, 'GHz', Y=y_func)
        assert JJ.Y(conditional_to_tensor(10), 2) == 20

        # Test default Y
        JJ = Junction(10, 'GHz', Y='default')
        assert np.isclose(sqf.to_numpy(JJ.Y(conditional_to_tensor(1e9 * 2 * np.pi), 0.015)),
                          1.6206254741014418e+16)
        JJ = Junction(0.142, 'GHz')
        assert np.isclose(sqf.to_numpy(JJ.Y(conditional_to_tensor(50e9), 0.015)),
                          793439739057240.9)


def test_junction_unit():
    JJ = Junction(10)
    val = JJ.get_value()
    assert np.isclose(val, 10 * 2 * np.pi * 1e9)

    sq.set_unit_JJ("Hz")
    JJ = Junction(val / 2 / np.pi)
    assert np.isclose(JJ.get_value('GHz'),  10 * 2 * np.pi)
    # this is global, so need to set back
    sq.set_unit_JJ('GHz')


def test_junction_grad():
    sq.set_engine('NumPy')
    # First check error massages
    with pytest.raises(ModeError):
        Junction(10, requires_grad=True)

    with pytest.raises(ModeError):
        assert not Junction(10).requires_grad

    junc_value_no_grad = Junction(10).get_value()

    set_optim_mode(True)

    junc_value_with_grad = Junction(10, requires_grad=True).get_value()

    assert junc_value_no_grad == float_torch_to_python(junc_value_with_grad)

    set_optim_mode(False)
