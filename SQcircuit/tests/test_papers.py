"""Tests example code in published papers."""

import os

import numpy as np
import torch

import SQcircuit as sq
import SQcircuit.functions as sqf
from SQcircuit.tests.test_grad import function_grad_test
from SQcircuit.tests.conftest import (
    create_JJL_numpy,
    create_JJL_torch,
)

TESTDIR = os.path.dirname(os.path.abspath(__file__))
PHASEDIR = os.path.join(TESTDIR, 'data', 'phase_coord')
SPECDIR = os.path.join(TESTDIR, 'data', 'spectra')

def _test_paper_1_a() -> None:
    target_freqs = np.array([0.69721321, 1.43920434, 1.90074875, 2.05449921])

    sq.set_engine('NumPy')

    loop1 = sq.Loop(value=0)

    C = sq.Capacitor(value=0.15, unit="GHz")
    CJ = sq.Capacitor(value=10, unit="GHz")
    L = sq.Inductor(value=0.13, unit="GHz", loops = [loop1])
    JJ = sq.Junction(value=5, unit="GHz", loops=[loop1])

    elements = {
        (0, 1): [CJ, JJ],
        (0, 2): [L],
        (0, 3): [C],
        (1, 2): [C],
        (1, 3): [L],
        (2, 3): [CJ, JJ]
    }

    cr = sq.Circuit(elements)

    cr.set_charge_offset(mode=3, ng=1)

    cr.set_trunc_nums([25, 1, 25])

    efreqs, _ = cr.diag(n_eig=5)

    ## Perform test
    efreqs_minus_ground = efreqs[1:] - efreqs[0]
    assert np.allclose(efreqs_minus_ground, target_freqs)
    ##

    cr.set_charge_offset(mode=3, ng=0)

    # phi = np.linspace(0, 1,100)
    # # spectrum of the circuit
    # n_eig=5
    # spec = np.zeros((n_eig, len(phi)))
    # for i in range(len(phi)):
    #     # set the external flux for the loop
    #     loop1.set_flux(phi[i])
    #     # diagonalize the circuit
    #     spec[:, i], _ = cr.diag(n_eig)

    # ## Perform test
    # with open(os.path.join(SPECDIR, 'zeropi_paper.npy'), 'rb') as f:
    #     test_spec = np.load(f)
    # assert np.allclose(test_spec, spec)
    # ##

    # duplicates test_phase_coords, except for truncation numbers
    loop1.set_flux(0)
    _, _ = cr.diag(n_eig=2)
    # create a range for each mode
    phi1 = np.pi*np.linspace(-1,1,100)
    phi2 = 0
    phi3 = np.pi*np.linspace(-0.5,1.5,100)
    # the ground state
    state0 = cr.eig_phase_coord(k=0, grid=[phi1, phi2, phi3])
    # the first excited state
    state1 = cr.eig_phase_coord(k=1, grid=[phi1, phi2, phi3])

    ##
    with open(os.path.join(PHASEDIR, 'zeropi_paper_0.npy'), 'rb') as f:
        test_state0 = np.load(f)

    assert np.allclose(np.abs(test_state0), np.abs(state0), rtol=1e-4, atol=1e-3)

    with open(os.path.join(PHASEDIR, 'zeropi_paper_1.npy'), 'rb') as f:
        test_state1 = np.load(f)
    assert np.allclose(np.abs(test_state1), np.abs(state1), rtol=1e-4, atol=1e-3)
    ##


def test_paper_1_b() -> None:
    # define the loop of the circuit
    loop1 = sq.Loop()

    # define the circuit’s elements
    C = sq.Capacitor(3.6, "GHz", Q=1e6)
    L = sq.Inductor(0.46, "GHz", loops=[loop1])
    JJ = sq.Junction(10.2, "GHz", cap=C, A=5e-7, loops=[loop1])

    # define the fluxonium circuit
    elements = {(0, 1): [L, JJ]}
    cr = sq.Circuit(elements, flux_dist="all")

    # set the truncation numbers
    cr.set_trunc_nums([100])

    # external flux for sweeping over
    phi = np.linspace(0, 1, 300)

    # T_1 and T_phi
    T_1 = np.zeros_like(phi)
    T_phi = np.zeros_like(phi)

    for i in range(len(phi)):
        # set the external flux for the loop
        loop1.set_flux(phi[i])
            
        # diagonalize the circuit
        _, _ = cr.diag(n_eig=2)

        # get the T_1 for capacitive loss
        T_1[i] = 1/cr.dec_rate(dec_type="capacitive", states=(1,0))
    
        # get the T_phi for cc noise
        T_phi[i] = 1/cr.dec_rate(dec_type="cc", states=(1,0))


def test_paper_2_a() -> None:
    ## First, do the code verbatim

    # Set the engine
    sq.set_engine('PyTorch')

    # Define the transmon elements
    C = sq.Capacitor(12, 'fF', requires_grad=True)
    JJ = sq.Junction(10, 'GHz')
    elements = {(0, 1): [C, JJ]}

    # Define the transmon circuit
    transmon = sq.Circuit(elements)

    # Set truncation number
    transmon.set_trunc_nums([100])

    # Diagonalize the circuit
    efreqs, evecs = transmon.diag(n_eig=10)

    # Calculate the matrix element
    g_10 = 1 / C.get_value() * torch.abs(
        evecs[1].conj()
        @ transmon.charge_op(1, 'original')
        @ evecs[0]
    )

    # Backpropagation step
    g_10.backward()

    # Access the gradient
    print(C.grad)


    ## Now, check that it's correct
    # First clear old grad
    C.grad = None

    # Build equivalent numpy circuit
    sq.set_engine('NumPy')
    C = sq.Capacitor(12, 'fF')
    JJ = sq.Junction(10, 'GHz')
    elements = {(0, 1): [C, JJ]}
    transmon_numpy = sq.Circuit(elements)
    transmon_numpy.set_trunc_nums([100])
    transmon_numpy.diag(n_eig=10)

    #  Function to get matrix element
    def g_10(cr):
        evecs = cr.evecs
        C = cr.elements[(0, 1)][0]
        return 1 / C.get_value() * sqf.abs(
            sqf.dag(evecs[1])
            @ transmon.charge_op(1, 'original')
            @ evecs[0]
        )

    function_grad_test(
        transmon_numpy,
        g_10,
        transmon,
        g_10,
        delta=1e-6
    )

def test_paper_2_b() -> None:
    epsilon_star = 1e-5

    circuit = create_JJL_numpy(30, 0.1)
    circuit.diag(10)

    circuit.check_convergence(
        t = 2,
        threshold = epsilon_star
    )

    circuit = create_JJL_torch(30, 0.1)
    circuit.diag(10)

    circuit.check_convergence(
        t = 2,
        threshold = epsilon_star
    )

    # TOOD: test epsilon matches predicted value
