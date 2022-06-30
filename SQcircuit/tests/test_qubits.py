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

