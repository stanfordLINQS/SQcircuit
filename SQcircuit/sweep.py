"""
sweep.py contains the classes for sweeping
"""

from .circuit import *
from itertools import product
import matplotlib.pyplot as plt


class Sweep:
    """
    Class that contains the sweeping methods.

    Parameters
    ----------
    cr: Circuit
        An object of Circuit class that we want to sweep over its parameters.
    numEig: int
        Number of eigenvalues.
    properties: list
        A list of properties that we want to calculate via sweeping.
    """

    def __init__(self, cr: Circuit, numEig: int):
        self.cr = cr
        self.numEig = numEig
        # eigen frequency table
        self.efreq = None
        # grid for sweeping
        self.grid = None
        # self.properties = properties

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

    def calProperties(self):
        pass

    def sweepFlux(self, loops: list, grid: list, toFile: str = None, plotF: bool = False):

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
            assert len(self.efreq.shape) <= 2, "SQcircuit can only plot 1D sweep for now"

            plt.figure(figsize=(6.5, 5), linewidth=1)
            plt.tight_layout()
            # plt.title("Energy Spectrum", fontsize=18)
            for i in range(self.numEig):
                plt.plot(grid[0] / 2 / np.pi, self.efreq[i, :] - self.efreq[0, :],
                         linewidth=2.2)
            plt.xlabel(r"$\Phi_{ext}/\Phi_0$", fontsize=18)
            plt.ylabel(r"$f_i-f_0$[GHz]", fontsize=18)
            plt.show()

        if toFile:
            # save the sweeping
            pass
        else:
            return self.efreq

    def sweepChargeOffset(self, modes: list, ranges: list, toFile: str = None):
        pass


