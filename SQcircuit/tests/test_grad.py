'''"""
test_elements contains the test cases for the SQcircuit elements
functionalities.
"""

from SQcircuit.elements import Capacitor, Junction
from SQcircuit.settings import set_optim_mode
from SQcircuit.circuit import Circuit, unt

import numpy as np

def test_T1():
    optim = True
    set_optim_mode(optim)

    # Construct the circuit
    cap_unit, ind_unit = 'pF', 'GHz'
    C = Capacitor(7.746, cap_unit, Q=1e6, requires_grad=optim)
    J = Junction(12, ind_unit, requires_grad=optim)

    elements = {
        (0, 1): [C, J],
    }

    cr_transmon = Circuit(elements)
    cr_transmon.set_trunc_nums([4, ])
    tensors, _, _ = cr_transmon.diag(3)
    T1_inv = cr_transmon.dec_rate('capacitive', (0, 1))
    T1_inv_0 = float(T1_inv)
    print(f"Initial T1^-1 (torch): {T1_inv_0}")

    # Compute gradient with autodifferentiation
    T1_inv.backward()
    print(f"Val grad C (torch): {C._value.grad * unt.farad_list[C.unit]}")
    print(f"Val grad J (torch): {J._value.grad * unt.freq_list[J.unit] / (2 * np.pi)}")
    set_optim_mode(False)

    assert False

def test_T1_linearization():
    # Compute gradient with linear approximation
    set_optim_mode(False)
    cap_unit, ind_unit = 'pF', 'GHz'
    delta = 1e-8
    C = Capacitor(7.746, cap_unit, Q=1e6)
    J = Junction(12, ind_unit)
    elements = {
        (0, 1): [C, J],
    }

    # Default T1 value
    cr_transmon = Circuit(elements)
    cr_transmon.set_trunc_nums([4, ])
    cr_transmon.diag(3)
    T1_inv = cr_transmon.dec_rate('capacitive', (0, 1))
    print(f"T1^-1 (numpy): {T1_inv}")

    delta_omega_0 = cr_transmon._efreqs[0]
    delta_omega_1 = cr_transmon._efreqs[1]

    # Gradient of capacitor
    C_delta = Capacitor(7.746 + delta, cap_unit, Q=1e6)
    elements = {
        (0, 1): [C_delta, J],
    }
    cr_transmon = Circuit(elements)
    cr_transmon.set_trunc_nums([4, ])
    cr_transmon.diag(3)
    T1_inv_delta_C = cr_transmon.dec_rate('capacitive', (0, 1))
    dT1InvdC = (T1_inv_delta_C - T1_inv) / delta
    print(f"d(T1^-1)/dC: {dT1InvdC}")

    delta_omega_0_prime = cr_transmon._efreqs[0]
    delta_omega_1_prime = cr_transmon._efreqs[1]

    # Gradient of junction
    J_delta = Junction(12 + delta, ind_unit)
    elements = {
        (0, 1): [C, J_delta],
    }
    cr_transmon = Circuit(elements)
    cr_transmon.set_trunc_nums([4, ])
    cr_transmon.diag(3)
    T1_inv_delta_J = cr_transmon.dec_rate('capacitive', (0, 1))
    dT1InvdEJ = (T1_inv_delta_J - T1_inv) / delta
    print(f"d(T1^-1)/dEJ: {dT1InvdEJ}")

    assert False'''