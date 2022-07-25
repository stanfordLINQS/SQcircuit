import os

import numpy as np
import dill as pickle

import SQcircuit as sq

TESTDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(TESTDIR, "data/phase_coord")

###############################################################################
# zeropi phase coord representation test
###############################################################################


def test_phase_coord_zeropi():

    loop1 = sq.Loop()

    C = sq.Capacitor(0.15, "GHz")
    CJ = sq.Capacitor(10, "GHz")
    JJ = sq.Junction(5, "GHz", loops=[loop1])
    L = sq.Inductor(0.13, "GHz", loops=[loop1])

    elements = {
        (0, 1): [CJ, JJ],
        (0, 2): [L],
        (3, 0): [C],
        (1, 2): [C],
        (1, 3): [L],
        (2, 3): [CJ, JJ],
    }

    # cr is an object of Qcircuit
    zrpi = sq.Circuit(elements)

    loop1.set_flux(0.9)
    zrpi.set_trunc_nums([35, 1, 6])
    _, _ = zrpi.diag(2)

    # create a range for each mode
    phi1 = np.pi * np.linspace(-1, 1, 100)
    phi2 = 0
    phi3 = np.pi * np.linspace(-0.5, 1.5, 100)

    # the ground state
    state0 = zrpi.eig_phase_coord(0, grid=[phi1, phi2, phi3])

    with open(DATADIR + '/zeropi_0', 'rb') as inp:
        state0_data = pickle.load(inp)

    assert np.allclose(state0, state0_data, rtol=1e-4, atol=1e-3)

    # the first excited state
    state1 = zrpi.eig_phase_coord(1, grid=[phi1, phi2, phi3])

    with open(DATADIR + '/zeropi_1', 'rb') as inp:
        state1_data = pickle.load(inp)

    assert np.allclose(state1, state1_data, rtol=1e-4, atol=1e-3)
