"""
test_elements contains the test cases for the SQcircuit elements functionalities.
"""
import pytest
from SQcircuit.elements import *
import SQcircuit as sq

#######################################
# Capacitor Tests
#######################################


def test_capacitorErrors():
    error = "The input unit for the capacitor is not correct. Look at the documentation for the correct input " \
            "format."
    with pytest.raises(ValueError, match=error):
        Capacitor(10, "H")


def test_capacitorEnergy():
    cap = Capacitor(10, "GHz")
    assert cap.energy() == 10
    # change the default frequency of the SQcirucit
    sq.units.setFreq("MHz")
    # check back the energy
    assert cap.energy() == 10 * 1000
    # set back the default frequency back to GHz
    sq.units.setFreq("GHz")
    # check the energy functionality from setting the value
    val = cap.value()
    cap2 = Capacitor(val, "F")
    assert cap2.energy() == 10


def test_capacitorQ():
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


def test_capacitorUnit():
    cap = Capacitor(10)
    val = cap.value()
    assert cap.energy() == 10
    assert cap.unit == "GHz"

    sq.units.setCap("F")
    cap = Capacitor(val)
    assert cap.energy() == 10
    assert cap.unit == "F"


#######################################
# Inductor Tests
#######################################


def test_inductorErrors():
    error = "The input unit for the inductor is not correct. Look at the documentation for the correct input " \
            "format."
    with pytest.raises(ValueError, match=error):
        Inductor(10, "F")


def test_inductorEnergy():
    ind = Inductor(10, "GHz")
    assert ind.energy() == 10
    # change the default frequency of the SQcirucit
    sq.units.setFreq("MHz")
    # check back the energy
    assert ind.energy() == 10 * 1000
    # set back the default frequency back to GHz
    sq.units.setFreq("GHz")
    # check the energy functionality from setting the value
    val = ind.value()
    ind2 = Inductor(val, "H")
    assert ind2.energy() == 10


def test_inductorQ():
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


def test_inductorUnit():
    ind = Inductor(10)
    val = ind.value()
    assert ind.energy() == 10
    assert ind.unit == "GHz"

    sq.units.setInd("H")
    ind = Inductor(val)
    assert ind.energy() == 10
    assert ind.unit == "H"


#######################################
# Josephson Junction Tests
#######################################

def test_junctionErrors():
    error = "The input unit for the Josephson Junction is not correct. Look at the documentation for the" \
                    "correct input format."
    with pytest.raises(ValueError, match=error):
        Junction(10, "F")


def test_junctionY():
    yFunc = lambda omega, T: omega*T
    JJ = Junction(10, "GHz", Y=yFunc)
    assert JJ.Y(10, 2) == 20


def test_junctionUnit():
    JJ = Junction(10)
    assert JJ.unit == "GHz"

    sq.units.setJJ("MHz")
    JJ = Junction(10)
    assert JJ.unit == "MHz"
