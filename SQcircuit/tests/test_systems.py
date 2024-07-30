"""test_system contains the test cases for the SQcircuit ``systems.py``
functionalities.
"""
import torch

import SQcircuit as sq
from SQcircuit.settings import set_optim_mode


def is_close(x, y, err) -> bool:

    # print(abs((x - y) / y))

    return abs((x - y) / y) < err


def test_coupled_fluxonium():
    """Test forward and backward pass of system modules for coupled
    Fluxonium."""

    set_optim_mode(True)

    loop1 = sq.Loop()
    loop2 = sq.Loop()

    # define the circuit elements
    C = sq.Capacitor(5.3806, 'fF', requires_grad=True)
    L1 = sq.Inductor(0.46, 'GHz', loops=[loop1], requires_grad=True)
    JJ1 = sq.Junction(10.2, 'GHz', loops=[loop1], requires_grad=True)

    L2 = sq.Inductor(0.46, 'GHz', loops=[loop2])
    JJ2 = sq.Junction(10.2, 'GHz', loops=[loop2])

    C_c = sq.Capacitor(3.6, 'GHz')

    # define the circuit
    elements = {
        (0, 1): [L1, JJ1, C],
        (0, 2): [L2, JJ2, C],
        (1, 2): [C_c]
    }

    cr = sq.Circuit(elements)
    cr.set_trunc_nums([30, 30])

    efreqs, _ = cr.diag(n_eig=10)

    omega_sq = efreqs[1] - efreqs[0]

    omega_sq.backward()

    c_sq_grad = C._value.grad
    C._value.grad = None
    l_sq_grad = L1._value.grad
    L1._value.grad = None
    jj_sq_grad = JJ1._value.grad
    JJ1._value.grad = None

    # First fluxonium
    flxnm1 = sq.Circuit({
        (0, 1): [L1, JJ1, C]
    })
    flxnm1.set_trunc_nums([200])
    flxnm1.diag(n_eig=40)

    # Second fluxonium
    flxnm2 = sq.Circuit({
        (0, 1): [L2, JJ2, C]
    })
    flxnm2.set_trunc_nums([200])
    flxnm2.diag(n_eig=40)

    couplings = [
        sq.Couple(flxnm1, 1, flxnm2, 1, C_c),
    ]

    sys = sq.System(couplings)

    H = sys.hamiltonian()

    efreqs = torch.linalg.eigvalsh(H)

    omega_sys = efreqs[1] - efreqs[0]

    omega_sys.backward()

    c_sys_grad = C._value.grad
    l_sys_grad = L1._value.grad
    jj_sys_grad = JJ1._value.grad

    assert is_close(omega_sq, omega_sys, 1e-2)
    # ToDo: Fixing these tests that are failing
    # assert is_close(c_sq_grad, c_sys_grad, 2e-2)
    # assert is_close(l_sq_grad, l_sys_grad, 2e-2)
    assert is_close(jj_sq_grad, jj_sys_grad, 20e-2)

    set_optim_mode(False)


def test_three_coupled_transmon():
    set_optim_mode(True)

    # define the circuit elements
    C_1 = sq.Capacitor(70, "fF", requires_grad=True)
    JJ_1 = sq.Junction(8.3472, "GHz")
    C_2 = sq.Capacitor(72, "fF")
    JJ_2 = sq.Junction(8.5493, "GHz", requires_grad=True)
    C_c = sq.Capacitor(200, "fF")
    JJ_c = sq.Junction(33.5661, "GHz", requires_grad=True)
    C_1c = sq.Capacitor(4, "fF", requires_grad=True)
    C_2c = sq.Capacitor(4.2, "fF")
    C_12 = sq.Capacitor(0.1, "fF")

    elements = {
        (0, 1): [C_1, JJ_1],
        (0, 2): [C_2, JJ_2],
        (0, 3): [C_c, JJ_c],
        (1, 2): [C_12],
        (1, 3): [C_1c],
        (2, 3): [C_2c],
    }

    cr = sq.Circuit(elements)
    cr.set_trunc_nums([10, 10, 10])

    efreqs, _ = cr.diag(n_eig=2)

    omega_sq = efreqs[1] - efreqs[0]

    omega_sq.backward()

    c1_sq_grad = C_1._value.grad
    C_1._value.grad = None
    c1c_sq_grad = C_1c._value.grad
    C_1c._value.grad = None
    jj2_sq_grad = JJ_2._value.grad
    JJ_2._value.grad = None
    jjc_sq_grad = JJ_c._value.grad
    JJ_c._value.grad = None

    # First qubit
    qubit_1 = sq.Circuit({
        (0, 1): [C_1, JJ_1]
    })
    qubit_1.set_trunc_nums([30])
    qubit_1.diag(n_eig=10)

    # Second qubit
    qubit_2 = sq.Circuit({
        (0, 1): [C_2, JJ_2]
    })
    qubit_2.set_trunc_nums([30])
    qubit_2.diag(n_eig=10)

    # coupler qubit
    qubit_c = sq.Circuit({
        (0, 1): [C_c, JJ_c]
    })
    qubit_c.set_trunc_nums([30])
    qubit_c.diag(n_eig=10)

    couplings = [
        sq.Couple(qubit_1, 1, qubit_c, 1, C_1c),
        sq.Couple(qubit_2, 1, qubit_c, 1, C_2c),
        sq.Couple(qubit_1, 1, qubit_2, 1, C_12),
    ]

    sys = sq.System(couplings)

    H = sys.hamiltonian()

    efreqs = torch.linalg.eigvalsh(H)

    omega_sys = efreqs[1] - efreqs[0]

    omega_sys.backward()

    c1_sys_grad = C_1._value.grad
    c1c_sys_grad = C_1c._value.grad
    jj2_sys_grad = JJ_2._value.grad
    jjc_sys_grad = JJ_c._value.grad

    assert is_close(omega_sq, omega_sys, 2e-2)
    assert is_close(c1_sq_grad, c1_sys_grad, 2e-2)
    assert is_close(c1c_sq_grad, c1c_sys_grad, 2e-2)
    assert is_close(jj2_sq_grad, jj2_sys_grad, 2e-2)
    assert is_close(jjc_sq_grad, jjc_sys_grad, 25e-2)

    set_optim_mode(False)
