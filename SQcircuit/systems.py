"""systems.py contains the classes for the interaction and connection of
superconducting circuits.
"""

from typing import List, Tuple, Union

import numpy as np
import qutip as qt

from numpy import ndarray
from qutip.qobj import Qobj

import SQcircuit.units as unt
import SQcircuit.functions as sqf

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

    def __init__(self, couplings: List['Couple']) -> None:

        self.couplings = couplings

        # list of circuits
        self.circuits = self._get_all_circuits()

        # number of sub-circuits (subsystems).
        self.n_sub = len(self.circuits)

        # overall number of nodes
        self.n_N = sum([circ.n for circ in self.circuits])

        # truncation numbers for each sub-circuit which is the number of
        # eigenvalue of each sub-circuit.
        self.trunc_nums = self._get_all_circuits_num_eig()

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

    def _get_all_circuits_num_eig(self) -> List[int]:
        """Return the number of eigenvalues for all circuit as a list.
        """

        num_eig_list = []

        for circ in self.circuits:
            num_eig_list.append(len(circ.efreqs))

        return num_eig_list

    def _node_idx_in_sys(self, cr: Circuit, i: int) -> int:
        """Return node index in the general system.

        Parameters
        ----------
            cr:
                Circuit of interest as a subsystem.
            i:
                Integer specifies the node index in the sub-circuit.
        """

        # number of nodes up to the cr circuit.
        N = 0
        for circ in self.circuits:
            if circ == cr:
                break
            N += circ.n

        return N+i-1

    def _node_idx_in_sub_sys(self, i: int) -> Tuple[int, int]:
        """Return circuit and node index related to node index in the overall
        system.

        Parameters
        ----------
            i:
                node index in the overall system.
        """

        # number of nodes
        N = 0

        # node in sub-circuit
        node_in_subsys: int = 0

        # index of the sub-circuit in overall system.
        sub_idx: int = 0

        for circ in self.circuits:
            if i < N+circ.n:
                # node in sub-circuit
                node_in_subsys = i-N+1

                # index of the circuit in overall system.
                sub_idx = self.circuits.index(circ)
                break
            N += circ.n

        return node_in_subsys, sub_idx

    def _bare_cap_matrix(self) -> ndarray:
        """Return the capacitance matrix of the entire system as``ndarray``
        without considering the coupling capacitors."""

        # list of capacitance matrix for each circuit
        cap_matrices = [circ.C for circ in self.circuits]

        return sqf.block_diag(*cap_matrices)

    def cap_matrix(self) -> ndarray:
        """Return the capacitance matrix of the entire system as``ndarray``."""

        C = self._bare_cap_matrix()

        for couple in self.couplings:

            if isinstance(couple.el, Capacitor):

                i = self._node_idx_in_sys(couple.circuits[0], couple.attrs[0])

                j = self._node_idx_in_sys(couple.circuits[1], couple.attrs[1])

                C[i, i] += couple.el.get_value()
                C[j, j] += couple.el.get_value()
                C[i, j] -= couple.el.get_value()
                C[j, i] -= couple.el.get_value()

        return C

    def _op_in_sys(self, sub_op: Qobj, sub_idx: int) -> Qobj:
        """Return the subsystem operator in the tensor product space of the
        overall system as ``Qutip.Qobj`` format.

        Parameters
        ----------
            sub_op:
                Subsystem operator with format of ``Qutip.Qobj``.
            sub_idx:
                Integer that specifies the index of the subsystem in the
                overall system.
        """

        op_list: List[Qobj] = []

        for i in range(self.n_sub):

            if i == sub_idx:
                op_list.append(sub_op)
            else:
                op_list.append(sqf.eye(self.trunc_nums[i]))

        return sqf.tensor_product(*op_list)

    def _op_times_op_in_sys(
            self,
            sub_op_1: Qobj,
            sub_idx_1: int,
            sub_op_2: Qobj,
            sub_idx_2: int
    ) -> Qobj:
        """Return the multiplication of two subsystem operators in the
        tensor product space of the overall system as ``Qutip.Qobj`` format.

        Parameters
        ----------
            sub_op_1:
                First subsystem operator with format of ``Qutip.Qobj``.
            sub_idx_1:
                Integer that specifies the first index of the subsystem in the
                overall system.
                        sub_op_1:
                Second subsystem operator with format of ``Qutip.Qobj``.
            sub_idx_1:
                Integer that specifies the second index of the subsystem in the
                overall system.
        """

        # If both operators are in the same subsystems.
        if sub_idx_1 == sub_idx_2:
            return self._op_in_sys(sqf.mat_mul(sub_op_1, sub_op_2), sub_idx_1)

        op_list: List[Qobj] = []

        for i in range(self.n_sub):

            if i == sub_idx_1:
                op_list.append(sub_op_1)
            elif i == sub_idx_2:
                op_list.append(sub_op_2)
            else:
                op_list.append(sqf.eye(self.trunc_nums[i]))

        return sqf.tensor_product(*op_list)

    def _QQ_op(self, m: int, n: int) -> Qobj:
        """Return Q_m*Q_n operator.

        Parameters
        ----------
            m:
                Integer that specifies the index of the first charge operator
                in the overall system.
            n:
                Integer that specifies the index of the second charge operator
                in the overall system.
        """

        node_in_subsys, m_sub_idx = self._node_idx_in_sub_sys(m)
        Q_m = self.circuits[m_sub_idx].charge_op(node_in_subsys, basis='eig')

        node_in_subsys, n_sub_idx = self._node_idx_in_sub_sys(n)
        Q_n = self.circuits[n_sub_idx].charge_op(node_in_subsys, basis='eig')

        return self._op_times_op_in_sys(Q_m, m_sub_idx, Q_n, n_sub_idx)

    def _quadratic_Q(self, A: ndarray) -> Qobj:
        """Return quadratic form of 1/2 * Q^T * A * Q

        Parameters
        ----------
            A:
                ndarray matrix that specifies the coefficient for
                quadratic expression.
        """

        op = 0

        for i in range(self.n_N):
            for j in range(self.n_N):
                if i == j:
                    op += 0.5 * A[i, i] * self._QQ_op(i, j)
                elif j > i:
                    op += A[i, j] * self._QQ_op(i, j)

        return op

    def _H_local(self) -> Qobj:
        """Return summation of local Hamiltonian in default frequency unit
        of SQcircuit as ``Qutip.Qobj`` format.
        """

        op = 0

        for i, circ in enumerate(self.circuits):

            op += self._op_in_sys(sqf.mat_to_op(sqf.diag(circ.efreqs)), i)

        return op

    def _H_int(self) -> Qobj:
        """Return interaction Hamiltonian in default frequency unit
        of SQcircuit as ``Qutip.Qobj`` format.
        """

        delta_C_inv = (sqf.mat_inv(self.cap_matrix())
                       - sqf.mat_inv(self._bare_cap_matrix()))

        R = sqf.block_diag(*[sqf.array(circ.R) for circ in self.circuits])

        return self._quadratic_Q(R.T @ delta_C_inv @ R) / (
                2*np.pi*unt.get_unit_freq())

    def hamiltonian(self) -> Qobj:
        """Return Hamiltonian of the overall system in default frequency
        unit of SQcircuit as ``Qutip.Qobj`` format.
        """

        return self._H_local() + self._H_int()
