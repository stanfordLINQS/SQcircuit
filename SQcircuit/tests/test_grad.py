"""
test_elements contains the test cases for the SQcircuit elements
functionalities.
"""

from SQcircuit.elements import Capacitor, Junction
from SQcircuit.settings import set_optim_mode
from SQcircuit.circuit import Circuit, unt

import numpy as np

trunc_num = 60
eigen_count = 60
tolerance = 1e-3


def test_omega():
    ### Numpy
    # Compute gradient with linear approximation
    set_optim_mode(False)
    cap_unit, ind_unit = 'pF', 'GHz'
    delta = 1e-6
    C = Capacitor(7.746, cap_unit, Q=1e6)
    J = Junction(12, ind_unit)
    elements = {
        (0, 1): [C, J],
    }

    # Solve for omega
    cr_transmon = Circuit(elements)
    cr_transmon.set_trunc_nums([trunc_num, ])
    cr_transmon.diag(eigen_count)
    omega = cr_transmon._efreqs[1] - cr_transmon._efreqs[0]
    print(f"omega numpy: {omega}")

    # Linear gradient of omega vs. C
    C_delta = Capacitor(7.746 + delta, cap_unit, Q=1e6)
    elements = {
        (0, 1): [C_delta, J],
    }
    cr_transmon = Circuit(elements)
    cr_transmon.set_trunc_nums([trunc_num, ])
    cr_transmon.diag(eigen_count)
    omega_delta_C = cr_transmon._efreqs[1] - cr_transmon._efreqs[0]
    domega_dC_numpy = (omega_delta_C - omega) / (delta * unt.farad_list[cap_unit])
    print(f"dw/dC (linear approx): {domega_dC_numpy}")

    # Linear gradient of omega vs. J
    J_delta = Junction(12 + delta, ind_unit)
    elements = {
        (0, 1): [C, J_delta],
    }
    cr_transmon = Circuit(elements)
    cr_transmon.set_trunc_nums([trunc_num, ])
    cr_transmon.diag(eigen_count)
    omega_delta_J = cr_transmon._efreqs[1] - cr_transmon._efreqs[0]
    domega_dJ_numpy = (omega_delta_J - omega) / (delta * unt.freq_list[ind_unit])
    print(f"dw/dJ (linear approx): {domega_dJ_numpy}")

    ### PyTorch
    set_optim_mode(True)
    C = Capacitor(7.746, cap_unit, Q=1e6, requires_grad=True)
    J = Junction(12, ind_unit, requires_grad=True)

    elements = {
        (0, 1): [C, J],
    }

    # Create circuit, backprop
    cr_transmon = Circuit(elements)
    cr_transmon.set_trunc_nums([trunc_num, ])
    tensors, eigenvals, _ = cr_transmon.diag(eigen_count)
    omega_torch = (eigenvals[1] - eigenvals[0]) * 2 * np.pi * 1e9
    print(f"omega torch: {omega_torch}")
    omega_torch.backward()

    # Read off C grad, J grad
    domega_dC_torch = C._value.grad
    print(f"dw/dC (torch): {C._value.grad}")
    domega_dJ_torch = 2 * np.pi * J._value.grad
    print(f"dw/dJ (torch): {domega_dJ_torch}")

    assert np.abs(domega_dC_torch / domega_dC_numpy) < 1 + tolerance
    assert np.abs(domega_dC_torch / domega_dC_numpy) > 1 - tolerance

    assert np.abs(domega_dJ_torch / domega_dJ_numpy) < 1 + tolerance
    assert np.abs(domega_dJ_torch / domega_dJ_numpy) > 1 - tolerance

    set_optim_mode(False)


def test_T1():
    ### Numpy
    # Compute gradient with linear approximation
    set_optim_mode(False)
    cap_unit, ind_unit = 'mF', 'GHz'
    delta = 1e-6
    C = Capacitor(7.746, cap_unit, Q=1e6)
    J = Junction(12, ind_unit)
    elements = {
        (0, 1): [C, J],
    }

    # Solve for T1_inv
    cr_transmon = Circuit(elements)
    cr_transmon.set_trunc_nums([trunc_num, ])
    cr_transmon.diag(eigen_count)
    Gamma1 = cr_transmon.dec_rate('capacitive', (0, 1))
    print(f"Gamma1 numpy: {Gamma1}")

    # Linear gradient of omega vs. C
    C_delta = Capacitor(7.746 + delta, cap_unit, Q=1e6)
    elements = {
        (0, 1): [C_delta, J],
    }
    cr_transmon = Circuit(elements)
    cr_transmon.set_trunc_nums([trunc_num, ])
    cr_transmon.diag(eigen_count)
    Gamma1_delta_C = cr_transmon.dec_rate('capacitive', (0, 1))
    dGamma1_dC_numpy = (Gamma1_delta_C - Gamma1) / (delta * unt.farad_list[cap_unit])
    print(f"dGamma1/dC (linear approx): {dGamma1_dC_numpy}")

    # Linear gradient of omega vs. J
    J_delta = Junction(12 + delta, ind_unit)
    elements = {
        (0, 1): [C, J_delta],
    }
    cr_transmon = Circuit(elements)
    cr_transmon.set_trunc_nums([trunc_num, ])
    cr_transmon.diag(eigen_count)
    Gamma1_delta_J = cr_transmon.dec_rate('capacitive', (0, 1))
    dGamma1_dJ_numpy = (Gamma1_delta_J - Gamma1) / (delta * unt.freq_list[ind_unit])
    print(f"dGamma1/dJ (linear approx): {dGamma1_dJ_numpy}")

    ### PyTorch
    set_optim_mode(True)
    C = Capacitor(7.746, cap_unit, Q=1e6, requires_grad=True)
    J = Junction(12, ind_unit, requires_grad=True)

    elements = {
        (0, 1): [C, J],
    }

    # Create circuit, backprop
    cr_transmon = Circuit(elements)
    cr_transmon.set_trunc_nums([trunc_num, ])
    tensors, eigenvals, _ = cr_transmon.diag(eigen_count)
    Gamma1_torch = cr_transmon.dec_rate('capacitive', (0, 1))
    print(f"Gamma1 torch: {Gamma1_torch}")
    Gamma1_torch.backward()

    # Read off C grad, J grad
    dGamma1_dC_torch = C._value.grad
    print(f"dGamma1/dC (torch): {C._value.grad}")
    dGamma1_dJ_torch = 2 * np.pi * J._value.grad
    print(f"dGamma1/dJ (torch): {dGamma1_dJ_torch}")

    assert np.abs(dGamma1_dC_torch / dGamma1_dC_numpy) < 1 + tolerance
    assert np.abs(dGamma1_dC_torch / dGamma1_dC_numpy) > 1 - tolerance

    assert np.abs(dGamma1_dJ_torch / dGamma1_dJ_numpy) < 1 + tolerance
    assert np.abs(dGamma1_dJ_torch / dGamma1_dJ_numpy) > 1 - tolerance

    set_optim_mode(False)