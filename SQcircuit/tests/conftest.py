"""
conftest.py contains the general test classes.
"""
import os

import numpy as np

from SQcircuit.sweep import *
from SQcircuit.storage import SQdata
from SQcircuit.circuit import Circuit

TESTDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(TESTDIR, "data/qubits")


class QubitTest:
    """
    class that contains the general tests
    """

    @classmethod
    def setup_class(cls):
        cls.fileName = None

    def test_transform_process(self):
        # load the data
        data = SQdata.load(DATADIR + "/" + self.fileName)

        # build the new circuit based on data circuit parameters
        newCr = Circuit(data.cr.elements)

        # check the modes and natural frequencies
        assert np.allclose(newCr.omega, data.cr.omega)
        # check the transformed w matrix
        assert np.allclose(newCr.wTrans, data.cr.wTrans)

    def test_if_description_run(self):
        """ Test if description run without error"""
        # load the data
        data = SQdata.load(DATADIR + "/" + self.fileName)

        # build the new circuit based on data circuit parameters
        newCr = Circuit(data.cr.elements)

        newCr.description(_test=True)

    def test_data(self):
        # load the data
        data = SQdata.load(DATADIR + "/" + self.fileName)

        efreq = None
        dec = None

        # build the new circuit based on data circuit parameters
        newCr = Circuit(data.cr.elements)
        newCr.set_trunc_nums(data.cr.m)

        if data.dec:
            properties = ["efreq", "loss"]
        else:
            properties = None

        numEig = data.efreq.shape[0]

        sweep1 = Sweep(newCr, numEig, properties)

        if data.type == "sweepFlux":
            efreq, dec = sweep1.sweepFlux(data.params, data.grid)
        elif data.type == "sweepCharge":
            efreq, dec = sweep1.sweepCharge(data.params, data.grid)

        for i in range(efreq.shape[0]):
            assert np.allclose(efreq[i, :], data.efreq[i, :],
                               rtol=1e-4, atol=1e-3)

        if data.dec:
            for decType in data.dec.keys():
                assert np.allclose(dec[decType], data.dec[decType]),\
                    "The \"{}\" loss has issue".format(decType)
