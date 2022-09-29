"""
test_elements contains the test cases for the SQcircuit elements
functionalities.
"""

import pytest
from SQcircuit.elements import Capacitor, Inductor, Junction
from SQcircuit.settings import get_optim_mode, set_optim_mode
from SQcircuit.circuit import Circuit

def test_T1_linearization():
    optim = True
    set_optim_mode(optim)

    from SQcircuit.elements import Capacitor
    # Define the circuit elements
    cap_unit, ind_unit = 'pF', 'GHz'
    C = Capacitor(7.746, cap_unit, Q=1e6, requires_grad=optim)
    J = Junction(12, ind_unit, requires_grad=optim)

    # Define the circuit
    elements = {
        (0, 1): [C, J],
    }

    # Diagonalize the circuit
    cr_transmon = Circuit(elements)
    cr_transmon.set_trunc_nums([4, ])
    cr_transmon.diag(2)
    T1 = cr_transmon.dec_rate('capacitive', (0, 1))

    T1.backward()