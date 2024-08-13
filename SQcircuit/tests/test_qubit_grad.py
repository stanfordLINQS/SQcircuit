"""
test_qubits_grad.py check the gradient calculation of SQcircuit for several
qubits.
"""

import numpy as np

import SQcircuit as sq
from SQcircuit.settings import set_optim_mode


def _get_grad_val_test(val1, val2, delta):
    return (val2 - val1) / delta


def _fluxonium(
    c_val: float,
    l_val: float,
    j_val: float,
    flux: float,
) -> sq.Circuit:
    """Return the eigenfrequency of the Fluxonium qubit.

    Parameters
    ----------
        c_val:
            Capacitor value in unit of "fF".
        l_val:
            Inductor value in unit of "uH".
        j_val:
            Junction value in unit of "GHz".
        flux:
            External flux of qubit.
    """

    loop1 = sq.Loop(flux)

    C = sq.Capacitor(c_val, 'fF')
    L = sq.Inductor(l_val, 'uH', loops=[loop1])
    JJ = sq.Junction(j_val, 'GHz', loops=[loop1])

    # define the circuit
    elements = {
        (0, 1): [C, L, JJ]
    }

    cr = sq.Circuit(elements)

    cr.set_trunc_nums([100])

    return cr


def _get_fluxonium_efreqs(
    c_val: float,
    l_val: float,
    j_val: float,
    flux: float,
    m: int = 1,
) -> float:
    """Return the eigenfrequency of the Fluxonium qubit.

    Parameters
    ----------
        c_val:
            Capacitor value in unit of "fF".
        l_val:
            Inductor value in unit of "uH".
        j_val:
            Junction value in unit of "GHz".
        flux:
            External flux of qubit.
        m:
            Index of desired eigenfrequency.
    """

    cr = _fluxonium(c_val, l_val, j_val, flux)

    efreq, evec = cr.diag(n_eig=10)

    return efreq[m] - efreq[0]


def test_fluxonium_grad():
    set_optim_mode(False)
    c_val = 3
    l_val = 0.353
    j_val = 10.2
    flux = 0.25

    delta = 1e-7

    efreq1 = _get_fluxonium_efreqs(c_val, l_val, j_val, flux)
    efreq2 = _get_fluxonium_efreqs(c_val + delta, l_val, j_val, flux)
    efreq3 = _get_fluxonium_efreqs(c_val, l_val + delta, j_val, flux)
    efreq4 = _get_fluxonium_efreqs(c_val, l_val, j_val + delta, flux)
    efreq5 = _get_fluxonium_efreqs(c_val, l_val, j_val, flux + delta)

    gr_C = _get_grad_val_test(efreq1, efreq2, delta)
    gr_L = _get_grad_val_test(efreq1, efreq3, delta)
    gr_J = _get_grad_val_test(efreq1, efreq4, delta)
    gr_P = _get_grad_val_test(efreq1, efreq5, delta)

    cr = _fluxonium(c_val, l_val, j_val, flux)
    cr.diag(10)

    C = cr.elements[(0, 1)][0]
    L = cr.elements[(0, 1)][1]
    JJ = cr.elements[(0, 1)][2]
    loop = cr.loops[0]

    freq_scale = 2 * np.pi * 1e9
    scale = 2 * np.pi
    assert np.isclose(cr.get_partial_omega(C, m=1)/freq_scale/1e15, gr_C)
    assert np.isclose(cr.get_partial_omega(L, m=1)/freq_scale/1e6, gr_L)
    assert np.isclose(cr.get_partial_omega(JJ, m=1), gr_J, atol=1e-6)
    assert np.isclose(cr.get_partial_omega(loop, m=1)/freq_scale*scale, gr_P)


def _zeropi(
    c_val: float,
    cj_val: float,
    l_val: float,
    j_val: float,
    flux: float,
) -> sq.Circuit:
    """Return the eigenfrequency of the Fluxonium qubit.

    Parameters
    ----------
        c_val:
            Capacitor value in unit of "fF".
        cj_val:
            Junction capacitor value in unit of "fF".
        l_val:
            Inductor value in unit of "uH".
        j_val:
            Junction value in unit of "GHz".
        flux:
            External flux of qubit.
    """

    loop1 = sq.Loop(flux)

    C = sq.Capacitor(c_val, 'fF')
    CJ = sq.Capacitor(cj_val, "fF")
    L = sq.Inductor(l_val, 'uH', loops=[loop1])
    JJ = sq.Junction(j_val, 'GHz', loops=[loop1])

    # define the circuit
    elements = {(0, 1): [CJ, JJ],
                (0, 2): [L],
                (0, 3): [C],
                (1, 2): [C],
                (1, 3): [L],
                (2, 3): [CJ, JJ]}

    cr = sq.Circuit(elements)

    cr.set_trunc_nums([45, 45])

    return cr


def _get_zeropi_efreqs(
    c_val: float,
    cj_val: float,
    l_val: float,
    j_val: float,
    flux: float,
    m: int = 1,
) -> float:
    """Return the eigenfrequency of the Fluxonium qubit.

    Parameters
    ----------
        c_val:
            Capacitor value in unit of "fF".
        l_val:
            Inductor value in unit of "uH".
        j_val:
            Junction value in unit of "GHz".
        flux:
            External flux of qubit.
        m:
            Index of desired eigenfrequency.
    """

    cr = _zeropi(c_val, cj_val, l_val, j_val, flux)

    efreq, evec = cr.diag(n_eig=10)

    return efreq[m] - efreq[0]


def test_zeropi_grad():
    set_optim_mode(False)
    c_val = 129
    cj_val = 1.93
    l_val = 1.257
    j_val = 5
    flux = 0.25
    delta = 1e-7

    efreq1 = _get_zeropi_efreqs(c_val, cj_val, l_val, j_val, flux)
    efreq2 = _get_zeropi_efreqs(c_val+delta, cj_val, l_val, j_val, flux)
    efreq3 = _get_zeropi_efreqs(c_val, cj_val+delta, l_val, j_val, flux)
    efreq4 = _get_zeropi_efreqs(c_val, cj_val, l_val+delta, j_val, flux)
    efreq5 = _get_zeropi_efreqs(c_val, cj_val, l_val, j_val+delta, flux)
    efreq6 = _get_zeropi_efreqs(c_val, cj_val, l_val, j_val, flux+delta)

    gr_C = _get_grad_val_test(efreq1, efreq2, delta)
    gr_CJ = _get_grad_val_test(efreq1, efreq3, delta)
    gr_L = _get_grad_val_test(efreq1, efreq4, delta)
    gr_J = _get_grad_val_test(efreq1, efreq5, delta)
    gr_P = _get_grad_val_test(efreq1, efreq6, delta)

    cr = _zeropi(c_val, cj_val, l_val, j_val, flux)
    cr.diag(10)

    CJ = cr.elements[(0, 1)][0]
    JJ = cr.elements[(0, 1)][1]
    C = cr.elements[(0, 3)][0]
    L = cr.elements[(0, 2)][0]
    loop = cr.loops[0]

    freq_scale = 2 * np.pi * 1e9
    scale = 2 * np.pi
    atol = 1e-2

    print(gr_CJ)
    print(cr.get_partial_omega(CJ, m=1)/freq_scale/1e15)

    assert np.isclose(cr.get_partial_omega(C, m=1)/freq_scale/1e15, gr_C,
                      atol=atol)
    assert np.isclose(cr.get_partial_omega(CJ, m=1)/freq_scale/1e15, gr_CJ,
                      atol=atol)
    assert np.isclose(cr.get_partial_omega(L, m=1)/freq_scale/1e6, gr_L,
                      atol=atol)
    assert np.isclose(cr.get_partial_omega(JJ, m=1), gr_J, atol=atol)
    assert np.isclose(cr.get_partial_omega(loop, m=1)/freq_scale*scale, gr_P,
                      atol=atol)
