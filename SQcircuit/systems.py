"""systems.py contains the classes for the interaction and connection of
superconducting circuits.
"""

from typing import List, Tuple, Union

import numpy as np
import qutip as qt

from numpy import ndarray
from qutip.qobj import Qobj

from SQcircuit.circuit import Circuit
from SQcircuit.elements import Capacitor, Inductor, Junction


class Couple:
    """Class  that stores the coupling between two different circuits.

    Parameters
    ----------
        cr1:
            Circuit of the system one.
        attr1:
            node or edge related to system one.
        cr2:
            Circuit of the system two.
        attr2:
            node or edge related to system two.
        el:
            The element of the coupling. (For now only capacitors)
    """
    def __init__(
            self,
            cr1: Circuit,
            attr1: Union[int, Tuple[int, int]],
            cr2: Tuple[Circuit, int],
            attr2: Union[int, Tuple[int, int]],
            el: Union[Capacitor, Inductor, Junction]
    ) -> None:

        # information of the circuits
        self.circuits = [cr1, cr2]

        # attribute of each circuit
        self.attrs = [attr1, attr2]

        # element of coupling
        self.el = el


class System:
    """Class that contains method for calculating the spectrum and properties
    of a system of coupled circuits.

    Parameters
    ----------
        couplings:
            List of couplings that describes the interaction between
            different part of the system.
    """

    def __init__(self, couplings: List["Couple"]) -> None:

        self.couplings = couplings

        # list of circuits
        self.circuits = self._get_all_circuits()

    def _get_all_circuits(self) -> List[Circuit]:
        """Return all the circuits described in ``System.couplings`` as a
        list."""

        # list of circuits
        circuits = []

        for couple in self.couplings:
            for circuit in couple.circuits:

                if circuit not in circuits:
                    circuits.append(circuit)

        return circuits

    def _node_idx_in_sys(self, cr: Circuit, node: int) -> int:
        """Return node index in the general system"""

        # number of nodes up to the cr circuit.
        N = 0
        for circuit in self.circuits:
            if circuit == cr:
                break
            N += circuit.n

        return N+node-1
