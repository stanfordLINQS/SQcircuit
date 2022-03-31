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


class QubitTest:
    """
    class that contains the general tests
    """

    @classmethod
    def setup_class(cls):
        cls.fileName = None

    def test_transformProcess(self):
        # load the data
        data = SQdata.load(DATADIR + "/" + self.fileName)

        # build the new circuit based on data circuit parameters
        newCr = Circuit(data.cr.circuitElements)

        # check the first transformation
        # assert np.allclose(newCr.R1, data.cr.R1)
        # assert np.allclose(newCr.S1, data.cr.S1)
        # assert np.allclose(newCr.omega, data.cr.omega)
        # # check the second transformation
        # assert np.allclose(newCr.R2, data.cr.R2)
        # assert np.allclose(newCr.S2, data.cr.S2)
        # # check the third transformation
        # assert np.allclose(newCr.R3, data.cr.R3)
        # assert np.allclose(newCr.S3, data.cr.S3)

        assert np.allclose(newCr.S, data.cr.S)
        assert np.allclose(newCr.R, data.cr.R)

    def test_Wand(self):
        # load the data
        data = SQdata.load(DATADIR + "/" + self.fileName)

        # build the new circuit based on data circuit parameters
        newCr = Circuit(data.cr.circuitElements)
        assert np.allclose(newCr.omega, data.cr.omega)
        assert np.allclose(newCr.wTrans, data.cr.wTrans)

    def test_data(self):
        # load the data
        data = SQdata.load(DATADIR + "/" + self.fileName)

        efreq = None
        dec = None

        # build the new circuit based on data circuit parameters
        newCr = Circuit(data.cr.circuitElements)
        newCr.truncationNumbers(data.cr.m)

        if data.dec:
            properties = ["efreq", "loss"]
        else:
            properties = None

        numEig = data.efreq.shape[0]

        sweep1 = Sweep(newCr, numEig, properties)

        if data.type == "sweepFlux":
            efreq, dec = sweep1.sweepFlux(data.params, data.grid)

        assert np.allclose(efreq, data.efreq)

        if data.dec:
            for decType in data.dec.keys():
                assert np.allclose(dec[decType], data.dec[decType]), "The \"{}\" loss has issue".format(decType)

    # def test_eigFreq(self):
    #     # load the data
    #     data = SQdata.load(DATADIR + "/" + self.fileName)
    #     # print(self.fileName, data.dec)
    #
    #     # build the new circuit based on data circuit parameters
    #     newCr = Circuit(data.cr.circuitElements)
    #     newCr.truncationNumbers(data.cr.m)
    #
    #     # table of eigenfrequencies that we want to calculate
    #     efreq = np.zeros(data.efreq.shape)
    #
    #     # dictionary that contains the decoherence rate for each loss mechanism
    #     dec = None
    #     if data.dec:
    #         dec = {key: np.zeros(Sweep._gridDims(data.grid)) for key in data.dec.keys()}
    #
    #     if data.type == "sweepFlux":
    #
    #         for indices in product(*Sweep._gridIndex(data.grid)):
    #
    #             # set flux for each loop
    #             for i, ind in enumerate(indices):
    #                 data.params[i].setFlux(data.grid[i][ind])
    #
    #             evec, _ = newCr.diag(data.numEig)
    #             efreq[:, indices] = evec.reshape(efreq[:, indices].shape)
    #
    #             if data.dec:
    #                 for decType in data.dec.keys():
    #                     dec[decType][indices] = newCr.decRate(decType=decType, states=(1, 0))
    #
    #     assert np.allclose(efreq, data.efreq)
    #
    #     if data.dec:
    #         for decType in data.dec.keys():
    #             assert np.allclose(dec[decType], data.dec[decType]), "The \"{}\" loss has issue".format(decType)