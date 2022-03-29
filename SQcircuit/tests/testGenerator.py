import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('../..'))

import SQcircuit as sq

tests ={"zeroPi": False,
        "inductivelyShunted": False,
        "Fluxonium": False
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
