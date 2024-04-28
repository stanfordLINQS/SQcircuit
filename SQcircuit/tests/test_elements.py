"""test_elements contains the test cases for the SQcircuit ``elements.py``
functionalities.
"""
import pytest
import numpy as np

from torch import Tensor

import SQcircuit as sq

from SQcircuit.elements import Capacitor, Inductor, Junction
from SQcircuit.logs import UNIT_ERROR, OPTIM_ERROR


def float_torch_to_python(x: Tensor) -> float:
    """Helper function to transform the torch float variable to python float
    variable."""

    return float(x.detach().cpu().numpy())


###############################################################################
# Capacitor Tests
###############################################################################


def test_capacitor_error_messages():
    with pytest.raises(TypeError, match=UNIT_ERROR):
        Capacitor(10, "H")


def test_capacitor_energy():
    cap = Capacitor(10, "GHz", min_value=0)
    assert cap.get_value("GHz") == 10

    assert cap.get_value("MHz") == 10 * 1000

    val = cap.get_value()
    cap2 = Capacitor(val, "F", min_value=0)
    assert cap2.get_value("GHz") == 10


def test_capacitor_Q():
    cap = Capacitor(10, "GHz", Q=1)
    assert cap.Q(-1) == 1
    assert cap.Q(10) == 1
    assert cap.Q(0) == 1

    cap = Capacitor(10, "GHz", Q=1e7)
    assert cap.Q(-1) == 1e7
    assert cap.Q(10) == 1e7
    assert cap.Q(0) == 1e7

    Q = lambda omega: omega ** 2
    cap = Capacitor(10, "GHz", Q=Q)
    assert cap.Q(2) == 4


def test_capacitor_unit():
    cap = Capacitor(10)
    val = cap.get_value()
    assert cap.unit == "GHz"

    sq.set_unit_cap("F")
    cap = Capacitor(val)
    assert cap.get_value("GHz") == 10
    assert cap.unit == "F"


def test_capacitor_grad():

    cap_value, cap_unit = 10, 'fF'
    # First check error messages
    with pytest.raises(ValueError, match=OPTIM_ERROR):
        Capacitor(cap_value, cap_unit, requires_grad=True)

    with pytest.raises(ValueError, match=OPTIM_ERROR):
        assert not Capacitor(cap_value, cap_unit).requires_grad

    cap_value_no_grad = Capacitor(cap_value, cap_unit).get_value()

    sq.set_optim_mode(True)

    cap_value_with_grad = Capacitor(cap_value, cap_unit, requires_grad=True).get_value()

    assert cap_value_no_grad == float_torch_to_python(cap_value_with_grad)

    sq.set_optim_mode(False)


###############################################################################
# Inductor Tests
###############################################################################


def test_inductor_error_messages():
    with pytest.raises(TypeError, match=UNIT_ERROR):
        Inductor(10, "F")


def test_inductor_energy():
    ind = Inductor(10, "GHz")
    assert ind.get_value("GHz") == 10

    val = ind.get_value()
    ind2 = Inductor(val, "H")
    assert ind2.get_value("GHz") == 10


def test_inductor_Q():
    ind = Inductor(10, "GHz", Q=1)
    assert ind.Q(-1, 0) == 1
    assert ind.Q(10, 12) == 1
    assert ind.Q(0, 11) == 1

    ind = Inductor(10, "GHz", Q=1e7)
    assert ind.Q(-1, 5) == 1e7
    assert ind.Q(10, 6) == 1e7
    assert ind.Q(0, 1) == 1e7

    ind = Inductor(10, "GHz")
    assert np.isclose(ind.Q(2 * np.pi * 0.5e9, 10), 500e6)
    assert np.isclose(ind.Q(2 * np.pi * 0.5e9, 1), 500e6)

    Q = lambda omega, T: omega ** 2
    ind = Inductor(10, "GHz", Q=Q)
    assert ind.Q(2, 10) == 4


def test_inductor_unit():
    ind = Inductor(10)
    val = ind.get_value()
    assert ind.unit == "GHz"

    sq.set_unit_ind("H")
    ind = Inductor(val)
    assert ind.get_value("GHz") == 10
    assert ind.unit == "H"


def test_inductor_grad():
    inductor_value, inductor_unit = 10, 'uH'

    # First check error messages
    with pytest.raises(ValueError, match=OPTIM_ERROR):
        Inductor(inductor_value, inductor_unit, requires_grad=True)

    with pytest.raises(ValueError, match=OPTIM_ERROR):
        assert not Inductor(inductor_value, inductor_unit).requires_grad

    ind_value_no_grad = Inductor(inductor_value, inductor_unit).get_value()

    sq.set_optim_mode(True)

    ind_value_with_grad = Inductor(inductor_value, inductor_unit, requires_grad=True).get_value()

    assert ind_value_no_grad == float_torch_to_python(ind_value_with_grad)

    sq.set_optim_mode(False)

###############################################################################
# Josephson Junction Tests
###############################################################################


def test_junction_error_messages():
    with pytest.raises(TypeError, match=UNIT_ERROR):
        Junction(10, "F")


def test_junction_Y():
    y_func = lambda omega, T: omega * T
    JJ = Junction(10, "GHz", Y=y_func)
    assert JJ.Y(10, 2) == 20


def test_junction_unit():
    JJ = Junction(10)
    assert JJ.unit == "GHz"

    sq.set_unit_JJ("MHz")
    JJ = Junction(10, min_value=0)

    assert JJ.unit == "MHz"

def test_junction_grad():

    # First check error messages
    with pytest.raises(ValueError, match=OPTIM_ERROR):
        Junction(10, requires_grad=True, min_value=0)

    with pytest.raises(ValueError, match=OPTIM_ERROR):
        assert not Junction(10, min_value=0).requires_grad

    junc_value_no_grad = Junction(10, min_value=0).get_value()

    sq.set_optim_mode(True)

    junc_value_with_grad = Junction(10, requires_grad=True, min_value=0).get_value()

    assert junc_value_no_grad == float_torch_to_python(junc_value_with_grad)

    sq.set_optim_mode(False)

