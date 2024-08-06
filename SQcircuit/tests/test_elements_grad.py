import SQcircuit as sq
from SQcircuit import ENV
import SQcircuit.functions as sqf
from SQcircuit.settings import set_optim_mode
from SQcircuit.tests.test_grad import function_grad_test


def fluxonium_numpy(trunc_num, phi_ext=0):
    set_optim_mode(False)
    loop = sq.Loop(phi_ext)
    C_numpy = sq.Capacitor(3.6, 'GHz', Q='default')
    L_numpy = sq.Inductor(0.46, 'GHz', Q='default', loops=[loop])
    JJ_numpy = sq.Junction(10.2, 'GHz', Y='default', loops=[loop])
    circuit_numpy = sq.Circuit({(0, 1): [C_numpy, L_numpy, JJ_numpy], }, flux_dist='junctions')
    circuit_numpy.set_trunc_nums([trunc_num, ])
    return circuit_numpy


def fluxonium_torch(trunc_num, phi_ext=0):
    set_optim_mode(True)
    loop = sq.Loop(phi_ext, requires_grad=True)
    C_torch = sq.Capacitor(3.6, 'GHz', Q='default', requires_grad=True)
    L_torch = sq.Inductor(0.46, 'GHz', Q='default', loops=[loop], requires_grad=True)
    JJ_torch = sq.Junction(10.2, 'GHz', Y='default', loops=[loop], requires_grad=True)
    circuit_torch = sq.Circuit({(0, 1): [C_torch, L_torch, JJ_torch], }, flux_dist='junctions')
    circuit_torch.set_trunc_nums([trunc_num, ])
    return circuit_torch


def test_capacitor_q_grad():
    cr_numpy = fluxonium_numpy(120, 0.1)
    cr_torch = fluxonium_torch(120, 0.1)

    test_q_cap = lambda cr: cr.elements[(0, 1)][0].Q(
        sqf.abs(cr.efreqs[1] - cr.efreqs[0]),
    )

    function_grad_test(
        cr_numpy,
        test_q_cap,
        cr_torch,
        test_q_cap
    )


def test_ind_q_grad():
    cr_numpy = fluxonium_numpy(120, 0.1)
    cr_torch = fluxonium_torch(120, 0.1)

    test_q_ind = lambda cr: cr.elements[(0, 1)][1].Q(
        sqf.abs(cr.efreqs[1] - cr.efreqs[0]),
        ENV['T']
    )

    function_grad_test(
        cr_numpy,
        test_q_ind,
        cr_torch,
        test_q_ind
    )


def test_junction_y_grad():
    cr_numpy = fluxonium_numpy(120, 0.1)
    cr_torch = fluxonium_torch(120, 0.1)

    test_y = lambda cr: cr.elements[(0, 1)][2].Y(
        sqf.abs(cr.efreqs[1] - cr.efreqs[0]),
        ENV['T']
    )

    function_grad_test(
        cr_numpy,
        test_y,
        cr_torch,
        test_y
    )
