"""Tests example code in published papers."""

import numpy as np
import torch

import SQcircuit as sq
import SQcircuit.functions as sqf
from SQcircuit.tests.test_grad import function_grad_test
from SQcircuit.tests.conftest import (
    create_JJL_numpy,
    create_JJL_torch,
)


def test_paper_1_a() -> None:
    target_freqs = np.array([0.69721321, 1.43920434, 1.90074875, 2.05449921])

    sq.set_engine('NumPy')

    C = sq.Capacitor(value=0.15, unit="GHz")
    CJ = sq.Capacitor(value=10, unit="GHz")
    loop1 = sq.Loop(value=0)
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

    efreqs_minus_ground = efreqs[1:] - efreqs[0]

    assert np.allclose(efreqs_minus_ground, target_freqs)

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

    circuit.check_convergence (
        t = 2,
        threshold = epsilon_star
    )

    circuit = create_JJL_torch(30, 0.1)
    circuit.diag(10)

    circuit.check_convergence(
        t = 2,
        threshold = epsilon_star
    )

    # TOOD: test epsilon mathces predicted value
