"""
sweep.py contains the classes for sweeping
"""

from .circuit import *
from .storage import *
from itertools import product


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

    def __init__(self, cr: Circuit, numEig: int):

        self.numEig = numEig

        # type of the data
        self.type = None
        # circuit of the data
        self.cr = cr
        # eigenfrequencies of the circuit
        self.efreq = None
        # parameters related to data
        self.params = None
        # grid related to data
        self.grid = None

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

    def _save(self, sweepType, params, gird, toFile):
        """
        Save the eigenfrequency data to file.
        """
        pass

    def calProperties(self):
        pass

    def sweepFlux(self, loops: list, grid: list, toFile: str = None, plotF: bool = False):

        self.type = "sweepFlux"
        self.params = loops
        self.grid = grid

        # table of eigenfrequencies that we want to calculate
        self.efreq = np.zeros((self.numEig, *self._gridDims(grid)))

        for indices in product(*self._gridIndex(grid)):

            # set flux for each loop
            for i, ind in enumerate(indices):
                loops[i].setFlux(grid[i][ind])

            evec, _ = self.cr.diag(self.numEig)
            self.efreq[:, indices] = evec.reshape(self.efreq[:, indices].shape)

        print("Sweeping process is finished!")

        if plotF:
            self.plot()

        if toFile:
            self.save(toFile)
            pass
        else:
            return self.efreq

    def sweepChargeOffset(self, modes: list, ranges: list, toFile: str = None):
        pass

