"""
sweep.py contains the classes for sweeping
"""

from itertools import product

from SQcircuit.circuit import *
from SQcircuit.storage import *


class Sweep(SQdata):
    """
    Class that contains the sweeping methods.

    Parameters
    ----------
    cr: Circuit
        An object of Circuit class that we want to sweep over its parameters.
    numEig: int
        Number of eigenvalues.
    """

    def __init__(self, cr: Circuit, numEig: int, properties=None):

        self.numEig = numEig

        if properties is None:
            properties = ["efreq"]

        self.properties = properties
        # type of the data
        self.type = None
        # circuit of the data
        self.cr = cr
        # parameters related to data
        self.params = None
        # grid related to data
        self.grid = None
        # eigenfrequencies of the circuit
        self.efreq = None
        # the decayRates
        if "loss" in properties:
            self.dec = {
                "capacitive": None,
                "inductive": None,
                "quasiparticle": None,
                "charge": None,
                "cc": None,
                "flux": None
            }
        else:
            self.dec = None

    @staticmethod
    def _gridDims(grid):
        """
        return the dimensions for each side of the grid as a tuple
        """
        return tuple(map(len, grid))

    @staticmethod
    def _gridIndex(grid):
        """
        return the range of indices for the grid
        """
        return tuple(map(range, map(len, grid)))

    def sweepFlux(self, loops: list, grid: list,
                  toFile: str = None, plotF: bool = False):

        self.type = "sweepFlux"
        self.params = loops
        self.grid = grid

        # table of eigenfrequencies that we want to calculate
        self.efreq = np.zeros((self.numEig, *self._gridDims(grid)))

        if "loss" in self.properties:
            # dictionary that contains the decoherence rate for each loss
            # mechanism
            self.dec = {key: np.zeros(self._gridDims(grid))
                        for key in self.dec.keys()}

        for indices in product(*self._gridIndex(grid)):

            # set flux for each loop
            for i, ind in enumerate(indices):
                loops[i].set_flux(grid[i][ind])

            evec, _ = self.cr.diag(self.numEig)
            self.efreq[:, indices] = evec.reshape(self.efreq[:, indices].shape)
            if "loss" in self.properties:
                for dec_type in self.dec.keys():
                    self.dec[dec_type][indices] = \
                        self.cr.dec_rate(dec_type=dec_type, states=(1, 0))

        print("Sweeping process is finished!")

        if plotF:
            self.plot("flux")

        if toFile:
            self.save(toFile)
            pass
        else:
            return self.efreq, self.dec

    def sweepCharge(self, modes: list, grid: list,
                    toFile: str = None, plotF: bool = False):

        self.type = "sweepCharge"
        self.params = modes
        self.grid = grid

        # table of eigenfrequencies that we want to calculate
        self.efreq = np.zeros((self.numEig, *self._gridDims(grid)))

        if "loss" in self.properties:
            # dictionary that contains the decoherence rate for each
            # loss mechanism
            self.dec = {key: np.zeros(self._gridDims(grid))
                        for key in self.dec.keys()}

        for indices in product(*self._gridIndex(grid)):

            # set flux for each loop
            for i, ind in enumerate(indices):
                self.cr.set_charge_offset(mode=modes[i], ng=grid[i][ind])

            evec, _ = self.cr.diag(self.numEig)
            self.efreq[:, indices] = evec.reshape(self.efreq[:, indices].shape)
            if "loss" in self.properties:
                for dec_type in self.dec.keys():
                    self.dec[dec_type][indices] = \
                        self.cr.dec_rate(dec_type=dec_type, states=(1, 0))

        print("Sweeping process is finished!")

        if plotF:
            self.plot("charge")

        if toFile:
            self.save(toFile)
            pass
        else:
            return self.efreq, self.dec
