import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('../..'))

import SQcircuit as sq

loop1 = sq.Loop()

C = sq.Capacitor(0.15, "GHz")
CJ = sq.Capacitor(10, "GHz")
JJ = sq.Junction(5, "GHz", loops=[loop1])
L = sq.Inductor(0.13, "GHz", loops=[loop1])

circuitElements = {(0, 1): [CJ, JJ],
                   (0, 2): [L],
                   (0, 3): [C],
                   (1, 2): [C],
                   (1, 3): [L],
                   (2, 3): [CJ, JJ]}

# cr is an object of Qcircuit
cr1 = sq.Circuit(circuitElements)

cr1.truncationNumbers([25, 1, 25])

sweep1 = sq.Sweep(cr1, numEig=5)

phi = np.linspace(0, 1, 50) * 2 * np.pi

sweep1.sweepFlux([loop1], [phi], plotF=True, toFile='data/zeroPi_test1')


