import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('../..'))

import SQcircuit as sq

tests = {"zeroPi": False,
         "inductivelyShunted": False,
         "Fluxonium": False,
         "Transmon": True,
         "tunableTransmon": True,
         }

#######################################
# zero-pi qubit
#######################################

if tests["zeroPi"]:
    loop1 = sq.Loop()

    C = sq.Capacitor(0.15, "GHz")
    CJ = sq.Capacitor(10, "GHz")
    JJ = sq.Junction(5, "GHz", loops=[loop1])
    L = sq.Inductor(0.13, "GHz", loops=[loop1])

    elements = {(0, 1): [CJ, JJ],
                (0, 2): [L],
                (3, 0): [C],
                (1, 2): [C],
                (1, 3): [L],
                (2, 3): [CJ, JJ]}

    # cr is an object of Qcircuit
    cr = sq.Circuit(elements)

    cr.truncationNumbers([25, 1, 25])

    sweep1 = sq.Sweep(cr, numEig=5)

    phi = np.linspace(0, 1, 50) * 2 * np.pi

    sweep1.sweepFlux([loop1], [phi], plotF=True, toFile='data/zeroPi_1')

#######################################
# inductively shunted qubit
#######################################

if tests["inductivelyShunted"]:
    loop1 = sq.Loop()

    # define the circuit’s elements
    C_r = sq.Capacitor(20.3, "fF")
    C_q = sq.Capacitor(5.3, "fF")
    L_r = sq.Inductor(15.6, "nH")
    L_q = sq.Inductor(386, "nH", loops=[loop1])
    L_s = sq.Inductor(4.5, "nH", loops=[loop1])
    JJ = sq.Junction(6.2, "GHz", loops=[loop1])

    # define the circuit
    elements = {(0, 1): [C_r],
                (1, 2): [L_r],
                (0, 2): [L_s],
                (2, 3): [L_q],
                (0, 3): [JJ, C_q]}

    cr = sq.Circuit(elements)

    cr.truncationNumbers([1, 9, 23])

    sweep1 = sq.Sweep(cr, numEig=10)

    phi = np.linspace(-0.1, 0.6, 50) * 2 * np.pi

    sweep1.sweepFlux([loop1], [phi], plotF=True, toFile='data/inductivelyShunted_1')

#######################################
# Fluxonium
#######################################

if tests["Fluxonium"]:
    # fluxonium with charge island( added by Yudan)

    loop1 = sq.Loop()
    C = sq.Capacitor(11, 'fF', Q=1e6)
    Cg = sq.Capacitor(0.5, 'fF', Q=1e6)
    L = sq.Inductor(1, 'GHz', Q=500e6, loops=[loop1])
    JJ = sq.Junction(3, 'GHz', cap=C, A=5e-7, loops=[loop1])

    circuitElements = {
        (0, 1): [Cg],
        (1, 2): [L, JJ],
        (0, 2): [Cg]
    }

    cr = sq.Circuit(circuitElements)

    cr.truncationNumbers([30, 1])

    sweep1 = sq.Sweep(cr, numEig=10)

    phi = np.linspace(0.0, 1.0, 50) * 2 * np.pi

    sweep1.sweepFlux([loop1], [phi], plotF=True, toFile='data/Fluxonium_1')

    # standard fluxonium with loss calculation

    loop1 = sq.Loop(A=1e-6)

    C = sq.Capacitor(3.6, 'GHz', Q=1e6)
    L = sq.Inductor(0.46, 'GHz', Q=500e6, loops=[loop1])
    JJ = sq.Junction(10.2, 'GHz', cap=C, A=1e-7, x=3e-06, loops=[loop1])

    circuitElements = {
        (0, 1): [L, JJ],
    }

    cr = sq.Circuit(circuitElements)

    cr.truncationNumbers([200])

    sweep1 = sq.Sweep(cr, numEig=10, properties=["efreq", "loss"])

    phi = np.linspace(0.1, 0.9, 50) * 2 * np.pi

    sweep1.sweepFlux([loop1], [phi], plotF=True, toFile='data/Fluxonium_2')


if tests["Transmon"]:

    C = sq.Capacitor(20, 'GHz', Q=1e6)
    JJ = sq.Junction(15.0, 'GHz', A=1e-7)

    circuitElements = {
        (0, 1): [C, JJ]
    }

    cr1 = sq.Circuit(circuitElements)

    cr1.truncationNumbers([30])

    numEig = 6
    ng = np.linspace(-2, 2, 200)
    eigenValues = np.zeros((numEig, len(ng)))

    decay = {'capacitive': np.zeros_like(ng),
             "cc_noise": np.zeros_like(ng),
             "charge_noise": np.zeros_like(ng)}

    for i in range(len(ng)):
        cr1.linkCharges({0: sq.Charge(ng[i], noise=1e-4)})
        eigenValues[:, i], _ = cr1.diag(numEig)

        for decType in decay:
            decay[decType][i] = cr1.decRate(decType=decType, states=(1, 0))
