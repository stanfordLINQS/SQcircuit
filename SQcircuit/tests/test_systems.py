"""test_system contains the test cases for the SQcircuit ``systems.py``
functionalities.
"""
import torch

import SQcircuit as sq


def is_close(x, y, err) -> bool:

    print(abs((x - y) / y))

    return abs((x - y) / y) < err


def test_coupled_Fluxonium():
    """Test forward and backward pass of system modules for coupled
    Fluxonium."""

    sq.set_optim_mode(True)

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

    _, efreqs, _ = cr.diag(n_eig=10)

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
    assert is_close(c_sq_grad, c_sys_grad, 2e-2)
    assert is_close(l_sq_grad, l_sys_grad, 2e-2)
    assert is_close(jj_sq_grad, jj_sys_grad, 18e-2)

    sq.set_optim_mode(False)

