"""
test_qubits contains the test cases for well-known qubits.
"""

from SQcircuit.tests.conftest import QubitTest


class TestZeroPi(QubitTest):
    """
    class for testing the zero-pi qubit
    """
    @classmethod
    def setup_class(cls):
        cls.fileName = "zeroPi_1"



