"""
conftest.py contains the general test classes.
"""
import os

import numpy as np
from SQcircuit.sweep import *
from SQcircuit.storage import SQdata
from SQcircuit.circuit import Circuit

TESTDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(TESTDIR, "data")


class TestQubit:
    """
    class that contains the general tests
    """

    @classmethod
    def setup_class(cls):
        cls.fileName = None

    def test_eigFreq(self):
        # load the test circuit
        data = SQdata.load(DATADIR + self.fileName)
        testCr = data.cr

        # build the new circuit based on test circuit parameters
        newCr = Circuit(testCr.circuitElements)
        newCr.truncationNumbers(testCr.m)

        # diagonalize and calculate the spectrum of the circuit based on the data type
        efreq = np.zeros(data.efreq.shape)
        if data.type == "sweepFlux":

            for indices in product(*Sweep._gridIndex(data.grid)):

                # set flux for each loop
                for i, ind in enumerate(indices):
                    data.params[i].setFlux(data.grid[i][ind])

                evec, _ = newCr.diag(data.numEig)
                efreq[:, indices] = evec.reshape(efreq[:, indices].shape)

        assert np.allclose(efreq, data.efreq)
