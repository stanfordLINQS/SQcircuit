"""systems.py contains the classes for the interaction and connection of
superconducting circuits.
"""

from typing import List, Tuple, Union

from circuit import Circuit
from elements import Capacitor, Inductor, Junction


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
        self.circuits = self._get_all_circuits()

    def _get_all_circuits(self) -> List[Circuit]:
        """Get all the circuits described in ``System.couplings`` as a list.
        """
        # list of circuits
        circuits = []

        for couple in self.couplings:
            for circuit in couple.circuits:

                if circuit not in circuits:
                    circuits += circuit

        return circuits
