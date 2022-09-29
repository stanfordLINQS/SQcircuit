"""
test_elements contains the test cases for the SQcircuit elements
functionalities.
"""

from SQcircuit.elements import Capacitor, Junction
from SQcircuit.settings import set_optim_mode
from SQcircuit.circuit import Circuit

def test_T1_linearization():
    optim = True
    set_optim_mode(optim)

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
    T1_inv = cr_transmon.dec_rate('capacitive', (0, 1))

    T1_inv.backward()
    print(f"Grad: {C.get_value().grad}")
    assert False
    set_optim_mode(False)