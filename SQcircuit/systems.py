"""systems.py contains the classes for the interaction and connection of
superconducting circuits.
"""

from typing import Tuple, Union

from circuit import Circuit


class Connect:
    """Class  that stores the connection between two circuit.

    Parameters
    ----------
        sys1:
            Tuple of the first circuit and the node of the circuit that we are
            connected to.
        sys2:
            Tuple of the second circuit and the node of the circuit that we are
            connected to.
        el:
            The element of the connection. (For now only capacitors)
    """
    def __init__(
            self,
            sys1: Tuple[Circuit, int],
            sys2: Tuple[Circuit, int],
            el: Union[Capacitor]
    ) -> None:

        # information of the system one
        self.sys1 = sys1

        # information of the system two
        self.sys2 = sys2

        # element of connection
        self.el = el


class System:
    """Class that contains method for calculating the spectrum of a system of
    circuits connected with specific type of coupling (either capacitively or
    inductively).
    """