"""systems.py contains the classes for the interaction and connection of
superconducting circuits.
"""

from typing import Tuple, Union

from circuit import Circuit


class Connect:
    """Class  that stores the connection between two circuit.

    Parameters
    ----------
        elements:
            A dictionary that contains the circuit's elements at each branch
            of the circuit.
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
        