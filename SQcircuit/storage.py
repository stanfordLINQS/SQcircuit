"""
storage.py contains the classes for storing the data
"""
import dill as pickle
import matplotlib.pyplot as plt
import numpy as np


class SQdata:
    """
    Class for processing and representing the data related to a circuit.
    """
    # type of the data
    type = None
    # circuit of the data
    cr = None
    # parameters related to data
    params = None
    # grid related to data
    grid = None
    # eigenfrequencies of the circuit
    efreq = None
    # the decayRates
    dec = None

    def save(self, toFile: str):
        with open(toFile, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(fromFile: str):
        with open(fromFile, 'rb') as inp:
            return pickle.load(inp)

    def plot(self, typeP):
        assert len(self.efreq.shape) <= 2, \
            "SQcircuit can only plot 1D grid data for now"

        plt.figure(figsize=(6.5, 5), linewidth=1)
        plt.tight_layout()
        # plt.title("Energy Spectrum", fontsize=18)
        for i in range(self.numEig):
            plt.plot(self.grid[0], self.efreq[i, :] - self.efreq[0, :],
                     linewidth=2.2)
        if typeP == "flux":
            plt.xlabel(r"$\Phi_{ext}/\Phi_0$", fontsize=18)
        elif typeP == "charge":
            plt.xlabel(r"$n_g$", fontsize=18)
        plt.ylabel(r"$f_i-f_0$[GHz]", fontsize=18)
        plt.show()
