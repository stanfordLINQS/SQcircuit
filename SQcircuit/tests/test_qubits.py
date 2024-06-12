"""
test_qubits contains the test cases for well-known qubits.
"""

import numpy as np

import SQcircuit as sq

from SQcircuit.tests.conftest import QubitTest


class TestZeroPi(QubitTest):
    """
    class for testing the zero-pi qubit
    """
    @classmethod
    def setup_class(cls):
        cls.fileName = "zeroPi_1"


class TestInductivelyShunted(QubitTest):
    """
    class for testing the zero-pi qubit
    """
    @classmethod
    def setup_class(cls):
        cls.fileName = "inductivelyShunted_1"


class TestFluxonium(QubitTest):
    """
    class for testing the zero-pi qubit
    """
    @classmethod
    def setup_class(cls):
        cls.fileName = "Fluxonium_1"


class TestFluxonium2(QubitTest):
    """
    class for testing the zero-pi qubit
    """
    @classmethod
    def setup_class(cls):
        cls.fileName = "Fluxonium_2"


class TestTransmon(QubitTest):
    """
    class for testing the zero-pi qubit
    """
    @classmethod
    def setup_class(cls):
        cls.fileName = "Transmon_1"


def test_resonator():
    """
    function for testing simple resonator.
    """
    C = sq.Capacitor(1 / 2 / np.pi, 'pF', Q=1e6)
    L = sq.Inductor(1 / 2 / np.pi, 'uH', Q=500e6)

    circuitElements = {
        (0, 1): [L, C],
    }

    cr = sq.Circuit(circuitElements)

    assert np.isclose(cr.omega/2/np.pi/1e9, 1)[0]

    # check if cr.description() run without error
    cr.description(_test=True)


def test_coupled_fluxonium_transmon():
    """Function for testing coupled fluxonium to the transmon."""

    target_efreqs = np.array([
        0., 2.07145177, 2.34187898, 3.09228726, 4.92917697, 5.19354651,
        5.87412504, 7.49054673, 7.74403853, 8.2146904
    ])

    loop1 = sq.Loop()

    # define the circuit elements
    C = sq.Capacitor(1, 'GHz')
    L = sq.Inductor(1, 'GHz', loops=[loop1])
    JJ = sq.Junction(1, 'GHz', loops=[loop1])
    JJ2 = sq.Junction(1, 'GHz')
    C_c = sq.Capacitor(1, 'GHz')

    elements = {
        (0, 1): [L, JJ, C],
        (0, 2): [JJ2, C],
        (1, 2): [C_c],
    }

    cr = sq.Circuit(elements)

    cr.set_trunc_nums([30, 10])

    efreqs, _ = cr.diag(10)

    efreqs = efreqs - efreqs[0]

    assert np.allclose(efreqs, target_efreqs)
