"""circuit.py contains the classes for the circuit and their properties
"""

from collections import defaultdict, OrderedDict
from copy import copy, deepcopy
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Type
from typing_extensions import Self

import numpy as np
import qutip as qt
import scipy.special
import scipy.sparse
import torch

import mpmath
from numpy import ndarray
from qutip import Qobj
from scipy.linalg import sqrtm, block_diag
from scipy.special import eval_hermitenorm, hyperu
from scipy.sparse.linalg import ArpackNoConvergence

from torch import Tensor

import SQcircuit.units as unt
import SQcircuit.functions as sqf
import SQcircuit.torch_extensions as sqtorch

from SQcircuit.elements import (
    Element,
    Capacitor,
    Inductor,
    Junction,
    Loop,
    Charge
)
from SQcircuit.texts import HamilTxt, is_notebook
from SQcircuit import symbolic
from SQcircuit.noise import ENV
from SQcircuit.settings import ACC, get_optim_mode
from SQcircuit.exceptions import raise_optim_error_if_needed, CircuitStateError

logger = logging.getLogger(__name__)


class CircuitEdge:
    """Class that contains the properties of an edge in the circuit.

    Parameters
    ----------
        circ:
            Circuit that edge is part of.
        edge:
            The tuple represents the edge of the circuit.
    """

    def __init__(
        self,
        circ: 'Circuit',
        edge: Tuple[int, int],
    ) -> None:

        self.circ = circ
        self.edge = edge

        self.w = self._set_w_at_edge()
        self.mat_rep = self._set_matrix_rep()

        self.edge_elems_by_type = {
            Capacitor: [],
            Inductor: [],
            Junction: []
        }

        # check if edge is processed
        self.processed = False

    def _set_w_at_edge(self) -> list:
        """Get the w_k vector as list at the edge."""

        # i1 and i2 are the nodes of the edge
        i1, i2 = self.edge

        w = (self.circ.n + 1) * [0]

        if i1 == 0 or i2 == 0:
            w[i1 + i2] += 1
        else:
            w[i1] += 1
            w[i2] -= 1

        return w[1:]

    def _set_matrix_rep(self) -> ndarray:
        """Special form of matrix representation for an edge of a graph.
        This helps to construct the capacitance and susceptance matrices.
        """

        edge_mat = np.zeros((self.circ.n, self.circ.n))

        if 0 in self.edge:
            i = self.edge[0] + self.edge[1] - 1
            edge_mat[i, i] = 1
        else:
            i = self.edge[0] - 1
            j = self.edge[1] - 1

            edge_mat[i, i] = 1
            edge_mat[j, j] = 1
            edge_mat[i, j] = -1
            edge_mat[j, i] = -1

        return edge_mat

    def update_circuit_loop_from_element(
        self,
        el: Union[Inductor, Junction],
        b_id: int,
    ) -> None:
        """Update loop properties related to circuit from element in the edge
        with its inductive index (b_id).

        Parameters
        ----------
            el:
                Inductive element.
            b_id:
                Inductive index.
        """

        for loop in el.loops:

            self.circ.add_loop(loop)
            loop.add_index(b_id)
            loop.add_to_k_mat(self.w)

            if get_optim_mode():
                self.circ._add_to_parameters(loop)

    def process_edge_and_update_circ(
        self,
        b_id: int,
        w_id: int,
        k_mat: list,
        c_edge_mat: list,
    ) -> Tuple[int, list, list]:
        """Process the edge and update the related circuit properties.

        Parameters
        ----------
            b_id:
                Point to each row of B matrix of the circuit.
            w_id:
                Point to each row of W matrix of the circuit.
            k_mat:
                Matrix related to loop calculation
            c_edge_mat:
                edge capacitance matrix
        """

        for el in self.circ.elements[self.edge]:

            self.edge_elems_by_type[el.type].append(el)

            # Case of inductive element
            if hasattr(el, 'loops'):

                self.edge_elems_by_type[Capacitor].append(el.cap)

                if get_optim_mode():
                    self.circ._add_to_parameters(el.cap)
                    if hasattr(el.cap, 'partial_mat'):
                        self.circ.partial_mats[el.cap] += (
                            el.cap.partial_mat(self.mat_rep)
                        )

                self.circ.elem_keys[el.type].append(
                    el.get_key(self.edge, b_id, w_id)
                )

                self.update_circuit_loop_from_element(el, b_id)

                b_id += 1

                k_mat.append(self.w)

                c_edge_mat.append(
                    el.get_cap_for_flux_dist(self.circ.flux_dist)
                )

            # Case of L and C
            if hasattr(el, 'partial_mat'):

                self.circ.partial_mats[el] += el.partial_mat(self.mat_rep)

            if get_optim_mode():

                self.circ._add_to_parameters(el)

        self.processed = True

        return b_id, k_mat, c_edge_mat

    def _check_if_edge_is_processed(self) -> None:

        assert self.processed, 'Edge is not processed yet!'

    def get_eff_cap_value(self) -> float:
        """Return effective capacitor value of the edge."""

        self._check_if_edge_is_processed()

        return sum(list(map(
            lambda c: c.get_value(),
            self.edge_elems_by_type[Capacitor]
        )))

    def get_eff_ind_value(self) -> float:
        """Return effective inductor value of the edge."""

        self._check_if_edge_is_processed()

        return sum(list(map(
            lambda l: 1/l.get_value(),
            self.edge_elems_by_type[Inductor]
        )))

    def is_JJ_in_this_edge(self) -> bool:
        """Check if the edge contains any JJ."""

        self._check_if_edge_is_processed()

        return len(self.edge_elems_by_type[Junction]) != 0

    def is_JJ_without_ind(self) -> bool:
        """Check if the edge only has JJ and no inductor."""

        self._check_if_edge_is_processed()

        flag = (
            len(self.edge_elems_by_type[Junction]) != 0
            and len(self.edge_elems_by_type[Inductor]) == 0
        )

        return flag


class Circuit:
    """Class that contains circuit properties and builds the Hamiltonian using
    the theory discussed in the original SQcircuit paper. Provides methods to
    calculate:

        * Eigenvalues and eigenvectors
        * Phase coordinate representation of eigenvectors
        * Coupling operators
        * Matrix elements
        * Decoherence rates
        * Gradients of Hamiltonian, eigenvalues/vectors, and decoherence rates

    Parameters
    ----------
        elements:
            A dictionary that contains the circuit's elements at each edge
            of the circuit.
        flux_dist:
            Provide the method of distributing the external fluxes. If
            ``flux_dist`` is ``"all"``, SQcircuit assign the external fluxes
            based on the capacitor of each inductive element (This option is
            necessary for time-dependent external fluxes). If ``flux_dist`` is
            ``"inductor"`` SQcircuit finds the external flux distribution by
            assuming the capacitor of the inductors are much smaller than the
            junction capacitors, If ``flux_dist`` is ``"junction"`` it is the
            other way around.
    """

    def __init__(
        self,
        elements: Dict[Tuple[int, int], List[Element]],
        flux_dist: str = 'junctions',
    ) -> None:

        #######################################################################
        # General circuit attributes
        #######################################################################

        self.elements = OrderedDict(
            [(key, elements[key]) for key in elements.keys()]
        )

        if flux_dist not in ['junctions', 'inductors', 'all']:
            raise ValueError("flux_dist option must either be 'junctions', "
                             "'inductors', or 'all'.")
        self.flux_dist = flux_dist

        # circuit inductive loops
        self.loops: List[Loop] = []

        # number of nodes without ground
        self.n: int = max(max(self.elements))

        # number of branches that contain junctions without a parallel inductor.
        self.num_jun_without_ind: int = 0

        self.elem_keys = {
            # inductor element keys: (edge, el, b_id) b_id point to
            # each row of B matrix (external flux distribution of that element)
            Inductor: [],
            # junction element keys: (edge, el, b_id, w_id) b_id point to
            # each row of B matrix (external flux distribution of that element)
            # and w_id point to each row of W matrix
            Junction: [],
        }

        # contains the parameters that we want to optimize.
        self._parameters: OrderedDict[Tuple[Element, Tensor]] = OrderedDict()

        #######################################################################
        # Transformation related attributes
        #######################################################################

        # get the capacitance matrix, sudo-inductance matrix, W matrix,
        # and B matrix (loop distribution over inductive elements)
        self.C, self.L, self.W, self.B = self._get_LCWB()

        # initialize the transformation matrices for charge and flux operators.
        self.R, self.S = np.eye(self.n), np.eye(self.n)

        # initialize transformed susceptance, inverse capacitance,
        # and W matrices.
        self.cInvTrans, self.lTrans, self.wTrans = (
            np.linalg.inv(sqf.to_numpy(self.C)),
            sqf.to_numpy(self.L).copy(),
            self.W.copy()
        )

        # natural angular frequencies of the circuit for each mode as a numpy
        # array (zero for charge modes)
        self.omega = np.zeros(self.n)

        # transform the Hamiltonian of the circuit
        self._transform_hamil()

        # charge islands of the circuit
        self.charge_islands = {
            i: Charge() for i in range(self.n) if
            self._is_charge_mode(i)
        }

        #######################################################################
        # Operator and diagonalization related attributes
        #######################################################################

        # truncation numbers for each mode
        self.m = []
        # squeezed truncation numbers (eliminating the modes with truncation
        # number equals 1)
        self.ms = []

        self._memory_ops: Dict[str, Union[List[Qobj],
                                          List[List[Qobj]], dict]] = {
            'Q': [],  # list of charge operators (normalized by 1/sqrt(hbar))
            'QQ': [[]],  # list of charge times charge operators
            'phi': [],  # list of flux operators (normalized by 1/sqrt(hbar))
            'N': [],  # list of number operators
            'exp': [],  # List of exponential operators
            'root_exp': [],  # List of square root of exponential operators
            'cos': {},  # List of cosine operators
            'sin': {},  # List of sine operators
            'sin_half': {},  # list of sin(phi/2)
            'ind_hamil': {},  # list of w^T*phi that appears in Hamiltonian
        }

        # TODO: fix typing; add comments etc.
        self.descrip_vars: Dict[str, Union[List[float], np.ndarray]] = {
            'omega': [],
            'phi_zp': [],
            'ng': [],
            'EC': None,
            'EJ': None
        }

        # LC part of the Hamiltonian
        self._LC_hamil = 0

        # eigenvalues of the circuit
        self._efreqs = sqf.array([])
        # eigenvectors of the circuit
        self._evecs = []

        # Toggle whether we need to copy all elements (namely the
        # _memory_ops and _LC_hamil; see .__getstate__)
        self._toggle_fullcopy = True

    def update(self) -> None:
        """Update the circuit Hamiltonian to reflect in-place changes made to
        the scalar values used for circuit elements (ex. C, L, J...).
        """

        self.elem_keys = {
            Inductor: [],
            Junction: [],
        }
        self.loops: List[Loop] = []

        self._parameters = OrderedDict()

        self.C, self.L, self.W, self.B = self._get_LCWB()
        self.R, self.S = np.eye(self.n), np.eye(self.n)
        self.cInvTrans, self.lTrans, self.wTrans = (
            np.linalg.inv(sqf.to_numpy(self.C)),
            sqf.to_numpy(self.L).copy(),
            self.W.copy()
        )
        self.omega = np.zeros(self.n)
        self._transform_hamil()

        self._memory_ops: Dict[
            str, Union[List[Qobj], List[List[Qobj]], dict]
        ] = {
            'Q': [],  # list of charge operators (normalized by 1/sqrt(hbar))
            'QQ': [[]],  # list of charge times charge operators
            'phi': [],  # list of flux operators (normalized by 1/sqrt(hbar))
            'N': [],  # list of number operators
            'exp': [],  # List of exponential operators
            'root_exp': [],  # List of square root of exponential operators
            'cos': {},  # List of cosine operators
            'sin': {},  # List of sine operators
            'sin_half': {},  # list of sin(phi/2)
            'ind_hamil': {},  # list of w^T*phi that appears in Hamiltonian
        }

        self._build_op_memory()
        self._LC_hamil = self._get_LC_hamil()
        self._build_exp_ops()

        self._efreqs = sqf.array([])
        self._evecs = []

    def __getstate__(self) -> dict[str, Any]:
        attrs = self.__dict__

        # When ``_toggle_fullcopy`` is ``False``, remove attributes of the
        # circuit which cost a lot of memory but can be reconstructed by calling
        # ``.update()``. Useful when pickling a copy of the circuit.
        if self._toggle_fullcopy:
            avoid_attrs = []
        else:
            avoid_attrs = ['_memory_ops', '_LC_hamil']

        self_dict = {k: attrs[k] for k in attrs if k not in avoid_attrs}

        return self_dict

    def __setstate__(self, state) -> None:
        self.__dict__ = state
        # Set ``_toggle_fullcopy`` back to ``True`` because this is the default
        # state when initializing a circuit, but is often set to ``False`` when
        # pickling the circuit.
        self._toggle_fullcopy = True

    @property
    def efreqs(self) -> Union[ndarray, Tensor]:
        """Eigenfrequencies in the chosen frequency unit for SQcircuit. If the
        SQcircuit engine is ``PyTorch``, the efreqs will be in ``Tensor``
        format; otherwise, they will be in ``ndarray`` format."""
        if len(self._efreqs) == 0:
            raise CircuitStateError('Please diagonalize the circuit first.')

        return self._efreqs / (2 * np.pi * unt.get_unit_freq())

    @property
    def evecs(self) -> Union[List[Qobj], Tensor]:
        """List of circuit eigenvectors. If the SQcircuit engine is ``PyTorch``,
        each eigenvector will be in ``Tensor`` format; otherwise, they will be
        in ``Qutip.Qobj`` format."""
        if len(self._evecs) == 0:
            raise CircuitStateError('Please diagonalize the circuit first.')

        return self._evecs

    @property
    def trunc_nums(self) -> List[int]:
        """List of truncation numbers of the circuit. For harmonic modes, these
        are N where the Hilbert space is 0, 1, …, (N-1) and for charge modes
        these are N where the Hilbert space is -(N-1), …, 0, …, (N-1).
        """
        trunc_nums = []
        for i in range(self.n):
            if self._is_charge_mode(i):
                trunc_nums.append(int((self.m[i] + 1)/2))
            else:
                trunc_nums.append(self.m[i])
        return trunc_nums

    @property
    def parameters(self) -> List[Tensor]:
        """The values of the elements in the circuit which require gradients
        to be computed (either leaf tensors wtih ``requires_grad == True``, or
        non-leaf tensors). The parameters can be set by a list of new values
        for each of the elements.
    
        Only available when using the ``PyTorch`` engine of ``SQcircuit``.
        """
        raise_optim_error_if_needed()

        return list(self._parameters.values())

    @parameters.setter
    def parameters(self, new_params: Union[Tensor, List[Tensor]]) -> None:
        raise_optim_error_if_needed()

        try:
            for i, element in enumerate(self._parameters.keys()):
                element.internal_value = new_params[i]
        except IndexError as e:
            raise ValueError('Shape of new parameters does not match.') from e

        self.update()

    @property
    def parameters_grad(self) -> Union[List[Optional[Tensor]], Tensor]:
        """Return the gradients of the tensors in ``.parameters``. If all values
        are not ``None``, it is returned as a stacked ``Tensor``, otherwise as a
        list of individual values.
        """
        raise_optim_error_if_needed()

        grad_list = []
        for val in self.parameters:
            grad_list.append(val.grad)

        if None in grad_list:
            return grad_list

        return torch.stack(grad_list).detach().clone()

    @property
    def parameters_dict(
            self
    ) -> OrderedDict[Tuple[Union[Element, Loop], Tensor]]:
        """The dictionary of (element, value) pairs for the elements in
        the circuit which require gradient.
        """
        raise_optim_error_if_needed()

        return self._parameters

    @property
    def parameters_elems(self) -> List[Union[Element, Loop]]:
        """The elements in the circuit which require gradient.
        """
        raise_optim_error_if_needed()

        return list(self._parameters.keys())

    def get_params_type(self) -> List[Union[Type[Element], Type[Loop]]]:
        """List of the types for each element in the circuit's parameters.
        """
        raise_optim_error_if_needed()

        elements_flattened = list(self._parameters.keys())

        params_type = [type(element) for element in elements_flattened]

        return params_type

    def zero_grad(self) -> None:
        """Set the gradient of all values in ``self.parameters`` to ``None``.
        """
        raise_optim_error_if_needed()

        for val in self.parameters:
            val.grad=None

    def _add_to_parameters(self, el: Element) -> None:
        """Add an element which requires gradient computation to ``.parameters``.
        Either
            - ``requires_grad`` is ``True``; or
            - the element is not a leaf tensor.

        Parameters
        ----------
            el:
                An element to add to ``.parameters``, if the element requires
                gradient and is not already present.
        """
        if (el.requires_grad or not el.is_leaf) and el not in self._parameters:
            self._parameters[el] = el.internal_value

    def add_loop(self, loop: Loop) -> None:
        """Add loop to the circuit loops. Should only be called when
        initializing the circuit.

        Parameters
        ----------
            loop:
                Loop in the circuit to add to ``.loops``.
        """
        if loop not in self.loops:
            loop.reset()
            self.loops.append(loop)

    def safecopy(self, save_eigs=False) -> Self:
        """Return a copy of ``self``, explicitly detaching and cloning all
        tensor values in the circuit (which are element and loop values).
        Eigenvalues/vectors are either discarded or detached and cloned based
        on the value of ``save_eigs``.

        Parameters
        ----------
            save_eigs:
                Whether to retain the eigenvalues/vectors in the copied version
                of the circuit.

        Returns
        ----------
            Deepcopy of `self`.
        """
        # Instantiate new container
        new_circuit = copy(self)

        # When using the PyTorch engine, SQcircuit contains many tensor values,
        # which may not be leafs. These don't implement a deepcopy method, so
        # need to be explicitly detached/cloned.
        if get_optim_mode():
            # Capacitance and inductance matrices are constructed from
            # element values
            new_circuit.C = new_circuit.C.detach().clone()
            new_circuit.L = new_circuit.L.detach().clone()

            # Replace all loops in circuit with identically-valued copies
            # whose `.internal_value`s are cloned.
            new_loops: List[Loop] = []
            replacement_dict: Dict[Union[Loop, Element], Union[Loop, Element]] = {}
            for loop in self.loops:
                new_loop = copy(loop)
                new_loop.internal_value = loop.internal_value.detach().clone()
                new_loops.append(new_loop)
                replacement_dict[loop] = new_loop
            new_circuit.loops = new_loops

            # Replace all elements in circuit with identically-valued copies
            # whose `.internal_value`s are cloned.
            new_elements = defaultdict(list)
            for edge in self.elements:
                for el in self.elements[edge]:
                    new_el = copy(el)
                    new_el.internal_value = el.internal_value.detach().clone()
                    # Need to also replace loops associated with circuit
                    # with the new copies.
                    if hasattr(el, 'loops'):
                        new_loops = []
                        for l in el.loops:
                            new_loops.append(replacement_dict[l])
                        new_el.loops = new_loops
                    new_elements[edge].append(new_el)

                    replacement_dict[el] = new_el
            new_circuit.elements = new_elements

            # Replace the parameters dict with the copied elements
            new_circuit._parameters = OrderedDict()
            for el in self._parameters:
                new_el = replacement_dict[el]
                new_circuit._parameters[new_el] = new_el.internal_value

            # Several operators in SQcircuit are saved in dictionaries
            # indexed by element. These keys all need to be updated to use
            # the copied elements.

            # Update the `.elem_keys` dictionary
            new_circuit.elem_keys = {
                Inductor: [],
                Junction: [],
            }
            for edge, el, b_id, w_id in self.elem_keys[Junction]:
                new_el = replacement_dict[el]
                new_circuit.elem_keys[Junction].append((edge, new_el, b_id, w_id))
            for edge, el, b_id in self.elem_keys[Inductor]:
                new_el = replacement_dict[el]
                new_circuit.elem_keys[Inductor].append((edge, new_el, b_id))

            # Update the `.partial_mats` dictionary
            new_circuit.partial_mats = defaultdict(lambda: 0)
            for el, partial_mat in self.partial_mats.items():
                try:
                    new_circuit.partial_mats[replacement_dict[el]] = partial_mat
                except KeyError:
                    new_circuit.partial_mats[el] = partial_mat

            # Update the `.memory_ops` dictionary.
            new_circuit._memory_ops = {}
            # Some operators contain dictionaries indexed by elements; the
            # others are fine.
            problem_types = ['cos', 'sin', 'sin_half', 'ind_hamil']
            for op_type in self._memory_ops:
                if op_type not in problem_types:
                    new_circuit._memory_ops[op_type] = self._memory_ops[op_type]
                else:
                    new_circuit._memory_ops[op_type] = {}
                    for el, b_id in self._memory_ops[op_type].keys():
                        new_circuit._memory_ops[op_type][(replacement_dict[el], b_id)] = (
                            self._memory_ops[op_type][(el, b_id)]
                        )

        # Remove old eigenvectors/values, if desired
        if save_eigs:
            new_circuit._efreqs = self._efreqs.detach().clone()
            new_circuit._evecs = self._evecs.detach().clone()
        else:
            new_circuit._efreqs = sqf.array([])
            new_circuit._evecs = []

        # Delete the `.descrip_vars`, because bug in the `ExplicitSymbol` class
        # prevents native `deepcopy`ing.
        new_circuit.descrip_vars = None

        # Deepcopy the whole thing, now that problematic attributes have been
        # explicitly taken care of.
        return deepcopy(new_circuit)

    def picklecopy(self) -> Self:
        """Helper function which returns a shallow copy of ``self`` with
        ``._toggle_fullcopy = False``. Use for pickling circuit to save memory.

        Returns
        ----------
            Copy of self with ``._toggle_fullcopy = False``.
        """
        # Instantiate new container
        new_circuit = copy(self)

        # Remove large objects when saving
        new_circuit._toggle_fullcopy = False

        return new_circuit

    def _get_LCWB(self) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """Calculate the capacitance matrix, sustenance matrix, W matrix,
        and the flux distribution over inductive elements B matrix.

        Returns
        ----------
            A tuple of the (capacitance, susceptance, W, B) matrices.
        """

        c_mat = sqf.zeros((self.n, self.n), dtype=torch.float64)
        l_mat = sqf.zeros((self.n, self.n), dtype=torch.float64)
        w_mat = []
        b_mat = sqf.array([])

        self.partial_mats = defaultdict(lambda: 0)

        # point to each row of B matrix (external flux distribution of that
        # element) or count the number of inductive elements.
        b_id = 0

        # w_id point to each row of W matrix for junctions or count the
        # number of edges contain JJ
        w_id = 0

        # number of branches that contain JJ without a parallel inductor.
        num_jun_without_ind = 0

        # k_mat is a matrix that transfers node coordinates to the edge phase
        # drop for inductive elements (In the paper Appendix D we referred to it
        # as W matrix)
        k_mat = []

        # capacitor at each inductive element
        c_edge_mat = []

        for edge in self.elements.keys():

            circ_edge = CircuitEdge(self, edge)

            b_id, k_mat, c_edge_mat = circ_edge.process_edge_and_update_circ(
                b_id,
                w_id,
                k_mat,
                c_edge_mat
            )

            if circ_edge.is_JJ_without_ind():
                num_jun_without_ind += 1

            c_mat += sqf.array(circ_edge.get_eff_cap_value()) * sqf.array(
                circ_edge.mat_rep)

            l_mat += sqf.array(circ_edge.get_eff_ind_value()) * sqf.array(
                circ_edge.mat_rep)

            if circ_edge.is_JJ_in_this_edge():
                w_mat.append(circ_edge.w)
                w_id += 1

        w_mat = np.array(w_mat)

        try:
            k_mat = np.array(k_mat)

            k_mat = sqf.remove_dependent_columns(k_mat)

            x_mat = k_mat.T @ np.diag([sqf.to_numpy(c) for c in c_edge_mat])

            for loop in self.loops:
                g = np.zeros((1, b_id))
                g[0, loop.indices] = loop.get_g()
                x_mat = np.concatenate((x_mat, g), axis=0)

            # number of inductive loops of the circuit
            n_loops = len(self.loops)

            if n_loops != 0:
                i_mat = np.concatenate(
                    (np.zeros((b_id - n_loops, n_loops)), np.eye(n_loops)),
                    axis=0
                )
                b_mat = np.linalg.inv(x_mat) @ i_mat
                b_mat = np.around(b_mat, 5)

        except ValueError:
            raise ValueError('The edge list does not specify a connected graph '
                             'or all inductive loops of the circuit are not '
                             'specified.') from None

        self.num_jun_without_ind = num_jun_without_ind

        return c_mat, l_mat, w_mat, b_mat

    def _is_charge_mode(self, i: int) -> bool:
        """Check if the mode is a charge mode.

        Parameters
        ----------
            i:
                index of the mode. (starts from zero for the first mode)

        Returns
        ----------
            Whether the ``i``th mode is a charge mode.
        """
        if i >= self.n:
            raise ValueError(f'The circuit only has {self.n} modes!')

        return self.omega[i] == 0

    def _apply_transformation(self, S: ndarray, R: ndarray) -> None:
        """Apply S and R transformation on transformed C, L, and W matrix.

        Parameters
        ----------
            S:
                Transformation matrices related to flux operators.
            R:
                Transformation matrices related to charge operators.
        """

        self.cInvTrans = R.T @ self.cInvTrans @ R
        self.lTrans = S.T @ self.lTrans @ S

        if len(self.W) != 0:
            self.wTrans = self.wTrans @ S

        self.S = self.S @ S
        self.R = self.R @ R

    def _get_and_apply_transformation_1(self) -> Tuple[ndarray, ndarray]:
        """Get and apply the first transformation of the coordinates that
        simultaneously diagonalizes the capacitance and susceptance matrices.

        Returns
        ----------
            A tuple of the (S1, R1) matrices.
        """

        c_mat_root = sqrtm(sqf.to_numpy(self.C))
        c_mat_root_inv = np.linalg.inv(c_mat_root)
        l_mat_root = sqrtm(sqf.to_numpy(self.L))

        _, D, U = np.linalg.svd(l_mat_root @ c_mat_root_inv)

        # the case that there is not any inductor in the circuit
        if np.max(D) == 0:
            D = np.diag(np.eye(self.n))
            sing_locs = list(range(0, self.n))
        else:
            # find the number of singularities in the circuit

            l_mat_eigs, _ = np.linalg.eig(sqf.to_numpy(self.L))
            num_sings = len(l_mat_eigs[
                    l_mat_eigs / np.max(l_mat_eigs) < ACC['sing_mode_detect']
            ])
            sing_locs = list(range(self.n - num_sings, self.n))
            D[sing_locs] = np.max(D)

        # build S1 and R1 matrix
        s_1 = c_mat_root_inv @ U.T @ np.diag(np.sqrt(D))
        r_1 = np.linalg.inv(s_1).T

        self._apply_transformation(s_1, r_1)

        self.lTrans[sing_locs, sing_locs] = 0

        return s_1, r_1

    @staticmethod
    def _independent_rows(A: ndarray) -> Tuple[List[int], List[ndarray]]:
        """Use the Gram-Schmidt process to find the linear independent rows of 
        matrix A and return the list of row indices of A and list of the rows.

        Parameters
        ----------
            A:
                ``np.ndarray`` matrix that we try to find its independent
                rows.

        Returns
        ----------
            The list of indices of linearly independent rows of ``A`` and a
            basis for the row space of ``A``.
        """

        # normalize the rows of matrix A
        A_norm = A / (np.linalg.norm(A, axis=1).reshape(A.shape[0], 1) + 1e-9)

        # Get the row of each A that has the highest norm in descending order.
        # This is important for the case of JJ capacitively coupled to JL.
        sorted_index = np.argsort(-np.linalg.norm(A, axis=1))

        basis = []
        idx_list = []

        for i in sorted_index:
            a = A_norm[i, :]
            a_prime = a - sum([np.dot(a, e) * e for e in basis])
            if (np.abs(a_prime) > ACC['Gram-Schmidt']).any():
                idx_list.append(i)
                basis.append(a_prime / np.linalg.norm(a_prime))

        return idx_list, basis

    def _round_to_zero_one(self, W: ndarray) -> ndarray:
        """Round the charge mode elements of W or transformed W matrix that
        are close to 0, -1, and 1 to the exact value of 0, -1, and 1
        respectively.

        Parameters
        ----------
            W:
                ``np.ndarray`` that can be either W or transformed W matrix.

        Returns
        ----------
            A rounded copy of ``W``.
        """

        rounded_W = W.copy()

        if self.num_jun_without_ind == 0:
            rounded_W[:, self.omega == 0] = 0

        charge_only_W = rounded_W[:, self.omega == 0]

        charge_only_W[np.abs(charge_only_W) < ACC['Gram-Schmidt']] = 0
        charge_only_W[np.abs(charge_only_W - 1) < ACC['Gram-Schmidt']] = 1
        charge_only_W[np.abs(charge_only_W + 1) < ACC['Gram-Schmidt']] = -1

        rounded_W[:, self.omega == 0] = charge_only_W

        return rounded_W

    def _is_gram_schmidt_successful(self, S) -> bool:
        """Check if the Gram_Schmidt process has the sufficient accuracy for
        the ``S`` transformation matrix.

        Parameters
        ----------
            S:
                Transformation matrix related to flux operators.

        Returns
        ----------
            True if the Gram-Schmidt succeeded
        """

        is_successful = True

        # absolute value of the current wTrans
        cur_w_trans = self._round_to_zero_one(self.wTrans @ S)

        for j in range(self.n):
            if self._is_charge_mode(j):
                for abs_w in np.abs(cur_w_trans[:, j]):
                    if abs_w != 0 and abs_w != 1:
                        is_successful = False

        return is_successful

    def _is_junction_in_circuit(self) -> bool:
        """Check if there are any Josephson junctions in the circuit.
        
        Returns
        ----------
            True if the circuit contains Josephson junctions, false otherwise.
        """

        return len(self.W) != 0

    def _get_and_apply_transformation_2(self) -> Tuple[ndarray, ndarray]:
        """Get and apply the second transformation of the coordinates that
        transforms the subspace of the charge operators in order to have the
        reciprocal primitive vectors in Cartesian direction.

        Returns
        ----------
            A tuple of the (S2, R2) transformation matrices.
        """

        if len(self.W) != 0:
            # remove independent rows
            w_t = sqf.remove_dependent_columns(self.W.T)
            w_reduced = w_t.T
            w_reduced_transformed = w_reduced @ self.S1

            # charge part of the reduced w_mat
            w_charge = w_reduced_transformed[:, self.omega == 0].copy()

            # get the charge basis part of the wTrans matrix
            # w_charge = self.wTrans[:, self.omega == 0].copy()

            # number of operators represented in charge bases
            nq = w_charge.shape[1]
        else:
            nq = 0
            w_charge = np.array([])

        # if we need to represent an operator in the charge basis
        if nq != 0 and self.num_jun_without_ind != 0:

            # list of indices of w vectors that are independent
            ind_lst = []

            # random vectors to complete the basis
            rnd_vecs = []

            # use Gram–Schmidt to find the linear independent rows of
            # normalized w_charge (w_charge_norm)
            basis = []

            while len(basis) != nq:
                if len(basis) == 0:
                    ind_lst, basis = self._independent_rows(w_charge)
                else:
                    # random vectors to complete the basis
                    rnd_vecs = list(
                        np.linalg.norm(w_charge, 'fro')
                        * np.random.randn(nq - len(basis), nq)
                    )
                    completed_basis = np.array(basis + rnd_vecs)
                    _, basis = self._independent_rows(completed_basis)

            # the second S and R matrix are:
            f_mat = np.array(list(w_charge[ind_lst, :]) + rnd_vecs)
            s_2 = block_diag(np.eye(self.n - nq), np.linalg.inv(f_mat))

            r_2 = np.linalg.inv(s_2.T)

        else:
            s_2 = np.eye(self.n, self.n)
            r_2 = s_2

        if self._is_gram_schmidt_successful(s_2):

            self._apply_transformation(s_2, r_2)

            self.wTrans = self._round_to_zero_one(self.wTrans)

            return s_2, r_2

        else:
            logger.warning('Gram_Schmidt process failed. Retrying...')

            return self._get_and_apply_transformation_2()

    def _get_and_apply_transformation_3(self) -> Tuple[ndarray, ndarray]:
        """ Get and apply the third transformation of the coordinates 
        that scales the modes.

        Returns
        ----------
            A tuple of the (S3, R3) transformation matrices.
        """

        s_3 = np.eye(self.n)

        for j in range(self.n):

            if self._is_charge_mode(j):
                # already scaled by second transformation
                continue

            # for harmonic modes
            elif self._is_junction_in_circuit():

                # note: alpha here is absolute value of alpha (alpha is pure
                # imaginary)
                # get alpha for j-th mode
                jth_alphas = np.abs(self.alpha(range(self.wTrans.shape[0]), j))
                self.wTrans[:, j][jth_alphas < ACC['har_mode_elim']] = 0

                if np.max(jth_alphas) > ACC['har_mode_elim']:
                    # find the coefficient in wTrans for j-th mode that
                    # has maximum alpha
                    s = np.abs(self.wTrans[np.argmax(jth_alphas), j])
                    s_3[j, j] = 1 / s
                else:
                    # scale the uncoupled mode
                    s = np.max(np.abs(self.S[:, j]))
                    s_3[j, j] = 1 / s

            else:
                # scale the uncoupled mode
                s = np.max(np.abs(self.S[:, j]))
                s_3[j, j] = 1 / s

        r_3 = np.linalg.inv(s_3.T)

        self._apply_transformation(s_3, r_3)

        return s_3, r_3

    def _transform_hamil(self) -> None:
        """Transform the Hamiltonian of the circuit into the charge and Fock
        bases.
        """

        # get the first transformation
        self.S1, self.R1 = self._get_and_apply_transformation_1()

        # natural frequencies of the circuit(zero for modes in charge basis)
        self.omega = np.sqrt(np.diag(self.cInvTrans) * np.diag(self.lTrans))

        if self._is_junction_in_circuit():
            # get the second transformation
            self.S2, self.R2 = self._get_and_apply_transformation_2()

        # scaling the modes by third transformation
        self.S3, self.R3 = self._get_and_apply_transformation_3()

    def _compute_descrip_vars(self) -> None:
        """Compute parameters of transformed Hamiltonian to reduced number
        of significant figures and in the frequency units of ``SQcircuit``;
        store in the ``.descrip_vars`` dictionary for easy access.
        """

        # Dimensions of modes of circuit
        self.descrip_vars['n_modes'] = self.n
        self.descrip_vars['har_dim'] = np.sum(self.omega != 0)
        self.descrip_vars['charge_dim'] = np.sum(self.omega == 0)
        self.descrip_vars['n_loops'] = len(self.loops)

        # Rounded matrices describing circuit layout
        self.descrip_vars['W'] = np.round(self.wTrans, 6)
        self.descrip_vars['S'] = np.round(self.S, 3)
        if self.loops:
            self.descrip_vars['B'] = np.round(self.B, 2)
        else:
            self.descrip_vars['B'] = np.zeros((len(self.elem_keys[Junction])
                          + len(self.elem_keys[Inductor]), 1))

        # Harmonic mode variables
        self.descrip_vars['omega'] = (
            self.omega / (2 * np.pi * unt.get_unit_freq())
        )
        self.descrip_vars['phi_zp'] = [
            self._phi_zp(i) for i in range(self.descrip_vars['har_dim'])
        ]

        # Compute the element values in the correct units
        ## Junction energies
        self.descrip_vars['EJ'] = []
        for _, el, _, _ in self.elem_keys[Junction]:
            self.descrip_vars['EJ'].append(
                sqf.to_numpy(el.get_value('Hz'))
                / (2 * np.pi * unt.get_unit_freq())
            )

        ## Inductive energies
        self.descrip_vars['EL'] = []
        self.descrip_vars['EL_has_ext_flux'] = []
        for _, el, b_id in self.elem_keys[Inductor]:
            self.descrip_vars['EL'].append(
                sqf.to_numpy(el.get_value('Hz')) / unt.get_unit_freq()
            )
            # Check if the element has any external flux associated with it
            # (corresponding row of B is nonzero)
            self.descrip_vars['EL_has_ext_flux'].append(
                np.sum(np.abs(self.descrip_vars['B'][b_id, :])) != 0
            )

        ## Charging energies
        self.descrip_vars['EC'] = (
            (2 * unt.e) **2
            / (unt.hbar * 2 * np.pi * unt.get_unit_freq())
            * self._get_quadratic_cInvTrans()
        )

        # Values of external flux and gate charge
        self.descrip_vars['loops'] = [
            sqf.to_numpy(self.loops[i].value()) / 2 / np.pi
            for i in range(self.descrip_vars['n_loops'])
        ]
        self.descrip_vars['ng'] = [
            self.charge_islands[i].value()
            for i in range(self.descrip_vars['har_dim'], self.n)
        ]

    def description(
        self,
        tp: Optional[str] = None,
        _test: bool = False,
    ) -> Optional[str]:
        """Print out Hamiltonian and a listing of the modes (whether they are
        harmonic or charge modes, with the frequency for each harmonic mode),
        Hamiltonian parameters, and external flux values.

        Values printed in the description can be accessed using the
        ``.descrip_vars`` dictionary, including a symbolic Hamiltonian in
        ``.descrip_vars['H']``.

        Parameters
        ----------
            tp:
                If ``None`` prints out the output as LaTeX if SQcircuit is
                running in a Jupyter notebook and as text if SQcircuit is
                running in Python terminal. If ``tp`` is ``'ltx'``,
                the output is in LaTeX format, and if ``tp`` is ``'txt'`` the
                output is in text format.
            _test:
                if True, return the entire description as string
                text (use only for testing the function).

        Returns
        ----------
            The text of the description as a string, if ``_test`` is ``True``.
        """
        if tp is None:
            if is_notebook():
                txt = HamilTxt('ltx', _test=_test)
            else:
                txt = HamilTxt('txt', _test=_test)
        else:
            txt = HamilTxt(tp, _test=_test)

        self._compute_descrip_vars()
        self.descrip_vars['H'] = symbolic.construct_hamiltonian(self)

        final_txt = txt.print_circuit_description(self.descrip_vars)

        if _test:
            return final_txt

    def loop_description(self, _test: bool = False) -> Optional[str]:
        """
        Print out the external flux distribution over inductive elements.

        Parameters
        ----------
            _test:
                if True, return the entire description as string
                text. (use only for testing the function)

        Returns
        ----------
            The text of the external flux distribution, if ``_test`` is
            ``True``.
        """
        loop_description_txt = HamilTxt.print_loop_description(self)

        if _test:
            return loop_description_txt

    def set_trunc_nums(self, nums: List[int]) -> None:
        """Set the truncation numbers for each mode.

        Parameters
        ----------
            nums:
                A list that contains the truncation numbers for each mode.
                Harmonic modes with truncation number N are 0, 1 , ...,
                (N-1), and charge modes with truncation number N are -(N-1),
                ..., 0, ..., (N-1).
        """

        logger.info('set_trunc_nums called')

        if not isinstance(nums, list):
            raise ValueError('The input must be be a python list')
        if len(nums) != self.n:
            raise ValueError('The number of modes (length of the input) must be '
                             'equal to the number of nodes.')

        self.m = self.n*[1]

        for i in range(self.n):
            # for charge modes:
            if self._is_charge_mode(i):
                self.m[i] = 2 * nums[i] - 1
            # for harmonic modes
            else:
                self.m[i] = nums[i]

        # squeeze the mode with truncation number equal to 1.
        self.ms = list(filter(lambda x: x != 1, self.m))

        self._build_op_memory()

        self._LC_hamil = self._get_LC_hamil()

        self._build_exp_ops()

    def set_charge_offset(self, mode: int, ng: float) -> None:
        """Set the charge offset for each charge mode.

        Parameters
        ----------
            mode:
                An integer that specifies the charge mode. To see, which mode
                is a charge mode, one can use ``description()`` method.
            ng:
                The charge offset.
        """
        if not isinstance(mode, int):
            raise ValueError('Mode number should be an integer.')
        if mode - 1 not in self.charge_islands:
            raise ValueError('The specified mode is not a charge mode.')

        if len(self.m) == 0:
            self.charge_islands[mode - 1].set_offset(ng)
        else:
            self.charge_islands[mode - 1].set_offset(ng)

            self._build_op_memory()

            self._LC_hamil = self._get_LC_hamil()

    def set_charge_noise(self, mode: int, A: float) -> None:
        """Set the charge noise for each charge mode.

        Parameters
        ----------
            mode:
                An integer that specifies the charge mode. To see which mode
                is a charge mode, we can use ``description()`` method.
            A:
                The charge noise.
        """
        if not isinstance(mode, int):
            raise ValueError('The mode number should be an integer.')
        if mode - 1 not in self.charge_islands:
            raise ValueError('The specified mode is not a charge mode.')

        self.charge_islands[mode - 1].setNoise(A)

    def _get_op_dims(self) -> List[list]:
        """Return the operator dims related to ``Qutip.Qobj``."""

        return [self.ms, self.ms]

    def _get_state_dims(self) -> List[list]:
        """Return the state dims related to ``Qutip.Qobj``."""

        return [self.ms, len(self.ms) * [1]]

    def _squeeze_op(self, op: Qobj) -> Qobj:
        """
        Return the same Quantum operator with squeezed dimensions.

        Parameters
        ----------
            op:
                Any quantum operator in qutip.Qobj format

        Returns
        ----------
            A squeezed copy of ``op``.
        """

        if isinstance(op, Qobj):
            return op.contract()

        op_sq = sqf.copy(op)
        op_sq.dims = self._get_op_dims()
        return op_sq

    def _impedance(self, i: int) -> float:
        """Compute the impedance of the ``i``th mode.

        Parameters
        ----------
            i:
                Index of the mode (starts from zero for the first mode).
        
        Returns
        ----------
            The impedance Z of mode ``i``.
        """
        return np.sqrt(
            self.cInvTrans[i, i] / self.lTrans[i, i]
        )

    def _charge_op_isolated(self, i: int) -> Qobj:
        """Return charge operator for each isolated mode normalized by
        square root of hbar. By isolated, we mean that the operator is not in
        the general tensor product states of the overall system.

        Parameters
        ----------
            i:
                Index of the mode (starts from zero for the first mode).

        Returns
        ----------
            The isolated charge operator for the ``i``th mode.
        """

        if self._is_charge_mode(i):
            ng = self.charge_islands[i].value()
            op = (2*unt.e/np.sqrt(unt.hbar)) * (qt.charge((self.m[i]-1)/2)-ng)
        else:
            Q_zp = -1j * np.sqrt(0.5 / self._impedance(i))
            if self.m[i] == 1:
                # QuTiP 5.0.x no longer supports create/destroy with N = 1
                op = Qobj([[0.]]).to('csr')
            else:
                op = Q_zp * (qt.destroy(self.m[i]) - qt.create(self.m[i]))

        return op

    def _flux_op_isolated(self, i: int) -> Qobj:
        """Return flux operator for each isolated mode normalized by
        square root of hbar. By isolated, we mean that the operator is not in
        the general tensor product states of the overall system.

        Parameters
        ----------
            i:
                Index of the mode. (starts from zero for the first mode)

        Returns
        ----------
            The isolated flux operator for the ``i``th mode.
        """

        if self._is_charge_mode(i):
            op = qt.qeye(self.m[i])
        else:
            if self.m[i] == 1:
                op = Qobj([[0.]]).to('csr')
            else:
                op = np.sqrt(0.5 * self._impedance(i)) * (
                    qt.destroy(self.m[i]) + qt.create(self.m[i])
                )

        return op

    def _num_op_isolated(self, i: int) -> Qobj:
        """Return number operator for each isolated mode. By isolated,
        we mean that the operator is not in the general tensor product states
        of the overall system.

        Parameters
        ----------
            i:
                Index of the mode. (starts from zero for the first mode)
    
        Returns
        ----------
            The isolated number operator for the ``i``th mode.
        """

        if self._is_charge_mode(i):
            op = qt.charge((self.m[i] - 1) / 2)

        else:
            op = qt.num(self.m[i])

        return op

    def _d_op_isolated(self, i: int, w: float) -> Qobj:
        """Return charge displacement operator for each isolated mode. By
        isolated, we mean that the operator is not in the general tensor
        product states of the overall system.

        Parameters
        ----------
            i:
                Index of the mode. (starts from zero for the first mode)
            w:
                Represent the power of the displacement operator, d^w. Right
                now w should be only 0, 1, and -1.

        Returns
        ----------
            The isolated charge displacement operator for the ``i``th mode, to 
            the power of ``w``.
        """

        if w == 0:
            return qt.qeye(self.m[i])

        d = np.zeros((self.m[i], self.m[i]))

        for k in range(self.m[i]):
            for j in range(self.m[i]):
                if j - 1 == k:
                    d[k, j] = 1
        d = Qobj(d).to('csr')

        if w < 0:
            return d

        elif w > 0:
            return d.dag()

    def _phi_zp(self, i: int) -> float:
        """Return the zero-point fluctuations for the ``i``th mode's
        flux operator.

        Parameters
        ----------
            i:
                Index of the mode (starts from zero for the first mode).

        Returns
        ----------
            The zero-point fluctuations for mode ``i``.
        """
        return (
            (2 * np.pi / unt.Phi0)
            * np.sqrt(unt.hbar / 2 * self._impedance(i))
        )

    def alpha(self, i: int, j: int) -> float:
        """Return the alpha, amount of displacement, for the bosonic
        displacement operator for junction i and mode j.

        Parameters
        ----------
            i:
                Index of the Junction (starts from zero for the first mode)
            j:
                Index of the mode (starts from zero for the first mode).

        Returns
        ----------
            The value of alpha for the junction ``i`` and mode ``j``.
        """

        return 1j * self._phi_zp(j) * self.wTrans[i, j]

    def _build_op_memory(self) -> None:
        """Build the charge, flux, number, and cross multiplication of charge
        operators and store them in memory related to operators.
        """

        charge_ops: List[Qobj] = []
        flux_ops: List[Qobj] = []
        num_ops: List[Qobj] = []
        charge_by_charge_ops: List[List[Qobj]] = []

        for i in range(self.n):

            Q = []
            charges_row = []
            num = []
            flux = []
            for j in range(self.n):
                if i == j:
                    Q_iso = self._charge_op_isolated(j)
                    if Q_iso.shape == (1, 1):
                        # QuTip 5.0.x converts (1x1 matrix)^2 to scalar
                        Q2 = Q + [Qobj(Q_iso * Q_iso).to('csr')]
                    else:
                        Q2 = Q + [Q_iso * Q_iso]
                    # append the rest with qeye.
                    Q2 += [qt.qeye(self.m[k]) for k in range(j+1, self.n)]
                    charges_row.append(self._squeeze_op(qt.tensor(*Q2)))

                    Q.append(Q_iso)
                    num.append(self._num_op_isolated(j))
                    flux.append(self._flux_op_isolated(j))
                else:
                    if j > i:
                        QQ = Q + [self._charge_op_isolated(j)]
                        # append the rest with qeye.
                        QQ += [qt.qeye(self.m[k]) for k in range(j+1, self.n)]
                        charges_row.append(self._squeeze_op(qt.tensor(*QQ)))

                    Q.append(qt.qeye(self.m[j]))
                    num.append(qt.qeye(self.m[j]))
                    flux.append(qt.qeye(self.m[j]))

            charge_ops.append(self._squeeze_op(qt.tensor(*Q)))
            num_ops.append(self._squeeze_op(qt.tensor(*num)))
            flux_ops.append(self._squeeze_op(qt.tensor(*flux)))
            charge_by_charge_ops.append(charges_row)

        self._memory_ops['Q'] = charge_ops
        self._memory_ops['QQ'] = charge_by_charge_ops
        self._memory_ops['phi'] = flux_ops
        self._memory_ops['N'] = num_ops

    def _build_exp_ops(self) -> None:
        """Build exponential operators needed to construct cosine potential of
        the Josephson Junctions and store them in memory related to operators.
        Note that cosine of JJs can be written as summation of two
        exponential terms,cos(x)=(exp(ix)+exp(-ix))/2. This function builds
        the quantum operators for only one exponential terms.
        """

        # list of exp operators
        exp_ops = []
        # list of square root of exp operators
        root_exp_ops = []

        # number of Josephson Junctions
        nJ = self.wTrans.shape[0]

        for i in range(nJ):

            # list of isolated exp operators
            exp = []
            # list of isolated square root of exp operators
            exp_h = []

            # tensor multiplication of displacement operator for JJ Hamiltonian
            for j in range(self.n):

                if self._is_charge_mode(j):
                    exp.append(self._d_op_isolated(j, self.wTrans[i, j]))
                    exp_h.append(qt.qeye(self.m[j]))
                else:
                    if self.m[j] == 1:
                        exp.append(qt.qeye(1).to('csr'))
                        exp_h.append(qt.qeye(1).to('csr'))
                    else:
                        exp.append(qt.displace(self.m[j], self.alpha(i, j)).to('csr'))
                        exp_h.append(qt.displace(self.m[j], self.alpha(i, j) / 2).to('csr'))

            exp_ops.append(self._squeeze_op(qt.tensor(*exp)))
            root_exp_ops.append(self._squeeze_op(qt.tensor(*exp_h)))

        self._memory_ops['exp'] = exp_ops
        self._memory_ops['root_exp'] = root_exp_ops

    def _get_quadratic_cInvTrans(self) -> ndarray:
        """Compute a modified ``cInvTrans`` for use when computing a
        quadratic form like
            ``0.5 Q^T @ cInvTrans @ Q``
        by summing over the upper/lower triangle.

        Returns
        ----------
            ``self.cInvTrans`` with the diagonal divided by 2.
        """
        q_cInvTrans = self.cInvTrans.copy()
        q_cInvTrans[np.diag_indices_from(q_cInvTrans)] /= 2

        return q_cInvTrans

    def _get_LC_hamil(self) -> Qobj:
        """Construct the LC part of the circuit Hamiltonian.

        Returns
        ----------
            The LC part of the Hamiltonian.
        """
        LC_hamil = 0

        q_cInvTrans = self._get_quadratic_cInvTrans()

        for i in range(self.n):
            # we write j in this form because of "_memory_ops["QQ"]" shape
            for j in range(self.n - i):
                if j == 0:
                    if self._is_charge_mode(i):
                        LC_hamil += (q_cInvTrans[i, i]
                                     * self._memory_ops['QQ'][i][j])
                    else:
                        LC_hamil += self.omega[i] * self._memory_ops['N'][i]

                elif j > 0:
                    if self.cInvTrans[i, i + j] != 0:
                        LC_hamil += (q_cInvTrans[i, i + j]
                                     * self._memory_ops['QQ'][i][j])

        return LC_hamil

    def _get_external_flux_at_element(
        self,
        b_id: int,
        torch = False
    ) -> Union[float, Tensor]:
        """
        Return the external flux at an inductive element.

        Parameters
        ----------
            b_id:
                An integer point to each row of B matrix (external flux
                distribution of that element)
            torch:
                If ``True``, cast loop values to floats always

        Returns
        ----------
            The external flux at the element referenced by ``b_id.``
        """
        phi_ext = sqf.zero()
        for i, loop in enumerate(self.loops):
            phi_ext += loop.value() * self.B[b_id, i]

        if isinstance(phi_ext, Tensor) and not torch:
            return sqf.to_numpy(phi_ext)
        else:
            return phi_ext

    def _get_inductive_hamil(self) -> Qobj:
        """Construct the inductive part of the circuit Hamiltonian.

        Returns
        ----------
            The inductive part of the Hamiltonian.
        """

        H = 0
        for edge, el, b_id in self.elem_keys[Inductor]:
            # phi = 0
            # if b_id is not None:
            phi = self._get_external_flux_at_element(b_id)

            # summation of the 1 over inductor values.
            x = np.squeeze(sqf.to_numpy(1 / el.get_value()))
            op = self.coupling_op('inductive', edge, force_use_qutip=True)
            H += x * phi * (unt.Phi0 / 2 / np.pi) * op / np.sqrt(unt.hbar)

            # save the operators for loss calculation
            self._memory_ops['ind_hamil'][(el, b_id)] = op
        for _, el, b_id, w_id in self.elem_keys[Junction]:
            # phi = 0
            # if b_id is not None:
            phi = self._get_external_flux_at_element(b_id)

            EJ = sqf.to_numpy(el.get_value())

            exp = np.exp(1j * phi) * self._memory_ops['exp'][w_id]
            root_exp = np.exp(1j * phi / 2) * self._memory_ops['root_exp'][
                w_id]
            cos = (exp + exp.dag()) / 2
            sin = (exp - exp.dag()) / 2j
            sin_half = (root_exp - root_exp.dag()) / 2j

            self._memory_ops['cos'][el, b_id] = self._squeeze_op(cos)
            self._memory_ops['sin'][el, b_id] = self._squeeze_op(sin)
            self._memory_ops['sin_half'][el, b_id] = self._squeeze_op(sin_half)

            H += -EJ * cos

        return H

    def charge_op(self, mode: int, basis: str = 'original') -> Qobj:
        """Return charge operator for specific mode in the Fock/Charge basis or
        the eigenbasis.

        Parameters
        ----------
            mode:
                Integer that specifies the mode number.
            basis:
                String that specifies the basis. It can be either ``'original'``
                for original Fock/Charge basis or ``'eig'`` for eigenbasis.

        Returns
        ----------
            The charge operator for the ``i``th mode in the basis specified by
            ``basis``.
        """
        if len(self.m) == 0:
            raise CircuitStateError('Please specify the truncation number for each mode.')
        if basis not in ['original', 'eig']:
            raise ValueError(f'Invalid basis \'{basis}\' passed. Permitted '
                             'bases are \'original\' and \'eig\'.')

        # charge operator in Fock/Charge basis
        Q_FC = self._memory_ops['Q'][mode-1]

        if basis == 'original':
            if get_optim_mode():
                return sqf.qobj_to_tensor(Q_FC)
            else:
                return Q_FC

        elif basis == 'eig':
            if get_optim_mode():
                mat_evecs = self._evecs.T
                Q = sqf.qobj_to_tensor(Q_FC)
                Q_eig = torch.conj(mat_evecs.T) @ Q @ mat_evecs

                return Q_eig
            else:
                mat_evecs = np.concatenate(list(map(
                    lambda v: v.full(), self._evecs)), axis=1)
                Q_eig = np.conj(mat_evecs.T) @ Q_FC.full() @ mat_evecs

                return Qobj(Q_eig)

    def _get_w_id(self, el: Junction, b_id: int) -> Optional[int]:
        """"
        Find the corresponding ``W`` matrix index given an junction and its
        ``B`` matrix index.
        
        Parameters
        ----------
            el:
                Josephson junction in circuit.
            b_id:
                Index of B matrix corresponding to ``el``.

        Returns
        ----------
            The corresponding ``W`` matrix index, if it exists.
        """
        for _, o_el, o_b_id, w_id in self.elem_keys[Junction]:
            if o_el == el and o_b_id == b_id:
                return w_id

        return None

    def op(self, typ: str, keywords: Dict) -> Union[Qobj, Tensor]:
        """Get a saved circuit operator of type ``typ``, specified by keywords
        given in the ``keywords`` dict, as a backpropagatable ``Tensor`` object 
        when using the ``'PyTorch'`` engine. Currently supports the
        following operators:

        * ``'sin_half'``

        Parameters
        ----------
            typ:
                Type of saved operator.
            keywords:
                Dictionary specifying which operator of type ``typ`` to return.

        Returns
        ----------
            The `typ` operator of the circuit specified by ``keywords``.
        """
        if typ == 'sin_half':
            b_id = keywords['b_id']
            el = keywords['el']
            if get_optim_mode():
                w_id = self._get_w_id(el, b_id)

                phi = self._get_external_flux_at_element(b_id, torch=True)
                root_exp = (
                    torch.exp(1j * phi / 2)
                    * sqf.qobj_to_tensor(self._memory_ops['root_exp'][w_id])
                )

                sin_half = (root_exp - sqf.dag(root_exp)) / 2j
                # ToDo: need to squeeze?
                return sin_half
            else:
                return self._memory_ops['sin_half'][el, b_id]
        else:
            raise ValueError('The operator \'{typ}\' is not supported.')

    def _diag_np(
        self,
        n_eig: int
    ) -> Tuple[Union[ndarray, Tensor], List[Union[Qobj, Tensor]]]:
        """Perform the diagonalization of the circuit Hailtonian using SciPy's
        sparse eigensolver. 
        
        Parameters
        ----------
            n_eig:
                Number of eigenvalues to compute.

        Returns
        ----------
            efreqs:
                Array of eigenfrequencies in frequency unit of SQcircuit.
            evecs:
                List of eigenvectors in qutip.Qobj or Tensor format, depending
                on numerical engine.
        """
        hamil = self.hamiltonian()

        # get the data out of qutip variable and use sparse SciPy 
        # eigensolver which is faster.
        try:
            efreqs, evecs = scipy.sparse.linalg.eigs(
                hamil.data_as('csr_matrix'), k=n_eig, which='SR'
            )
        except ArpackNoConvergence:
            efreqs, evecs = scipy.sparse.linalg.eigs(
                hamil.data_as('csr_matrix'), k=n_eig, ncv=10*n_eig, which='SR'
            )
        # the output of eigen solver is not sorted
        efreqs_sorted = np.sort(efreqs.real)

        sort_arg = np.argsort(efreqs)
        if isinstance(sort_arg, int):
            sort_arg = [sort_arg]

        evecs_sorted = [
            Qobj(evecs[:, ind], dims=self._get_state_dims())
            for ind in sort_arg
        ]

        # store the eigenvalues and eigenvectors of the circuit Hamiltonian
        self._efreqs = efreqs_sorted
        self._evecs = evecs_sorted

        return efreqs_sorted / (2 * np.pi * unt.get_unit_freq()), evecs_sorted

    def _diag_torch(self, n_eig: int) -> Tuple[Tensor, Tensor]:
        """Diagonalize the circuit using a Torch Function, so that the 
        calculated eigenvalues/vectors are backpropagatable.

        To restrict the number ``n`` of eigenvectors for which the gradient is
        computed, call ``set_max_eigenvector_grad(n)`` before
        diagonalizing.
        
        Parameters
        ----------
            n_eig:
                Number of eigenvalues to compute.

        Returns
        ----------
            efreqs:
                Tensor of eigenfrequencies.
            evecs:
                Tensor of eigenvectors.
        """
        eigen_solution = sqtorch.eigencircuit(self, n_eig)
        eigenvalues = torch.real(eigen_solution[:, 0])
        eigenvectors = eigen_solution[:, 1:]
        self._efreqs = eigenvalues
        self._evecs = eigenvectors

        return eigenvalues / (2 * np.pi * unt.get_unit_freq()), eigenvectors

    def diag(
        self,
        n_eig: int
    ) -> Tuple[Union[ndarray, Tensor], List[Union[Qobj, Tensor]]]:
        """
        Diagonalize the Hamiltonian of the circuit and return the
        eigenfrequencies and eigenvectors of the circuit up to specified
        number of eigenvalues.

        Parameters
        ----------
            n_eig:
                Number of eigenvalues to output. The lower ``n_eig``, the
                faster ``SQcircuit`` finds the eigenvalues.

        Returns
        ----------
            efreqs:
                ndarray of eigenfrequencies in frequency unit of SQcircuit
                (gigahertz by default).
            evecs:
                List of eigenvectors in qutip.Qobj or Tensor format, depending
                on numerical engine.
        """
        if len(self.m) == 0:
            raise CircuitStateError('Please specify the truncation number for each mode.')
        if not isinstance(n_eig, int):
            raise ValueError('n_eig (number of eigenvalues) should be an integer.')

        logger.info('diag called')
        if get_optim_mode():
            return self._diag_torch(n_eig)
        else:
            return self._diag_np(n_eig)

    def truncate_circuit(self, K: int) -> List[int]:
        """Set Hilbert space dimensionality of circuit to ``k=floor(K^{1/n})``
        for all modes, where ``n`` is the number of modes in the circuit.
        Note that charge modes have dimensionality 2*m+1, where m is the
        assigned truncation number. Consequently, to ensure equal Hilbert
        space sizing among charge and harmonic modes, a truncation number
        of ``(1 / 2) * (k + 1)`` is assigned to charge modes.


        Parameters
        ----------
            K:
                Total truncation number

        Returns
        ----------
            trunc_nums:
                List of truncation numbers for each mode of the circuit
        """
        if not isinstance(K, int):
            raise ValueError('The total truncation number must be an integer.')
        if K < 1:
            raise ValueError('The total truncation number must be >=1.')

        trunc_num_average = K ** (1 / len(self.omega))
        charge_cutoff = (1 / 2) * (trunc_num_average + 1)

        trunc_nums = []
        for mode_idx in range(len(self.omega)):
            # Harmonic mode
            if self.omega[mode_idx] != 0:
                trunc_nums.append(int(np.floor(trunc_num_average)))
            # Charge mode
            else:
                trunc_nums.append(int(np.floor(charge_cutoff)))

        self.set_trunc_nums(trunc_nums)
        return trunc_nums

    def check_convergence(self, eig_vec_idx=1, t=10, threshold=1e-5):
        """
        Check whether the diagonalization of the circuit has converged.

        Parameters
        ----------
            eig_vec_idx:
                Index of eigenvector to use to test convergence.
            t:
                Number of entries of eigenvector to use to test convergence.
            threshold:
                Cutoff for convergence.

        Returns
        ----------
            convergence_succeeded:
                Truthy value of whether the circuit converged
            epsilon:
                Calculated value for convergence test
        """
        if (self._efreqs.shape[0] == 0) or (len(self._evecs) == 0):
            raise CircuitStateError('Must call circuit.diag before testing convergence.')

        reshaped_evec = sqf.to_numpy(self.evecs[eig_vec_idx]).reshape(self.m)
        restricted_evec = reshaped_evec[(slice(-t),)*len(self.m)]

        epsilon = 1 - np.sum(np.abs(restricted_evec)**2)

        return (epsilon < threshold), epsilon

    ###########################################################################
    # Methods that calculate circuit properties
    ###########################################################################

    def coord_transform(self, var_type: str) -> ndarray:
        """
        Return the transformation of the coordinates as ndarray for each type
        of variables, either charge or flux.

        Parameters
        ----------
            var_type:
                The type of the variables that can be either ``"charge"`` or
                ``"flux"``.

        Returns
        ----------
            Matrix giving coordinate transformation for ``var_type`` coordinates.
        """
        var_type = var_type.lower()
        if var_type == 'charge':
            return np.linalg.inv(self.R)
        elif var_type == 'flux':
            return np.linalg.inv(self.S)
        else:
            raise ValueError("The input must be either 'charge' or 'flux'.")

    def hamiltonian(self) -> Qobj:
        """
        Returns the transformed hamiltonian of the circuit as
        ``qutip.Qobj`` format.

        Returns
        ----------
            Circuit Hamiltonian.
        """
        if len(self.m) == 0:
            raise CircuitStateError('Please specify the truncation number for each mode.')

        Hind = self._get_inductive_hamil()

        H = Hind + self._LC_hamil

        # TODO: if diagonal matrix, could diagonalize instantantly
        return H.to('csr')

    def _tensor_to_modes(self, tensorIndex: int) -> List[int]:
        """
        Decomposes the tensor product space index to each mode indices. For
        example index 5 of the tensor product space can be decomposed to [1,
        0,1] modes if the truncation number for each mode is 2.

        Parameters
        ----------
            tensorIndex:
                Index of tensor product space

        Returns
        ----------
            ind_lst:
                A list of mode indices (self.n)
        """

        # i-th mP element is the multiplication of the self.m elements until
        # its i-th element
        mP = []
        for i in range(self.n - 1):
            if i == 0:
                mP.append(self.m[-1])
            else:
                mP = [mP[0] * self.m[-1 - i]] + mP

        ind_lst = []
        indexP = tensorIndex
        for i in range(self.n):
            if i == self.n - 1:
                ind_lst.append(indexP)
                continue
            ind_lst.append(int(indexP / mP[i]))
            indexP = indexP % mP[i]

        return ind_lst

    def eig_phase_coord(self, k: int, grid: Sequence[ndarray]) -> ndarray:
        """
        Return the phase coordinate representations of the eigenvectors as
        ndarray.

        Parameters
        ----------
            k:
                The eigenvector index. For example, we set it to 0 for the
                ground state and 1 for the first excited state.
            grid:
                A list that contains the range of values of phase φ for which
                we want to evaluate the wavefunction.

        Returns
        ----------
            Phase coordinate representation of the ``k``th eigenvector over
            the values of φ provided in ``grid``.
        """
        if len(self._evecs) == 0:
            raise CircuitStateError('Please diagonalize the circuit first.')
        if not isinstance(k, int):
            raise ValueError('The eigenstate index must be an integer.')

        phi_list = [*np.meshgrid(*grid, indexing='ij')]

        # The total dimension of the circuit Hilbert Space
        netDimension = np.prod(self.m)

        state = 0

        for i in range(netDimension):

            # decomposes the tensor product space index (i) to each mode
            # indices as a list
            ind_lst = self._tensor_to_modes(i)

            if get_optim_mode():
                term = self._evecs[k][i].item()
            else:
                term = self._evecs[k][i][0]

            for mode in range(self.n):

                # mode number related to that node
                n = ind_lst[mode]

                # For charge basis
                if self._is_charge_mode(mode):
                    term *= 1 / np.sqrt(2 * np.pi) * np.exp(
                        1j * phi_list[mode] * n)
                # For harmonic basis
                else:
                    # compute in log-space due to large magnitude variation
                    x0 = np.sqrt(unt.hbar * np.sqrt(
                        self.cInvTrans[mode, mode] / self.lTrans[mode, mode]))

                    coeff_log = (
                        - 0.25 * np.log(np.pi)
                        - 0.5 * sum(np.log(np.arange(1, n + 1)))
                        - 0.5 * np.log(x0)
                        + 0.5 * np.log(unt.Phi0)
                    )

                    if n < 250:
                        term_hermitenorm = eval_hermitenorm(n, np.sqrt(2) * phi_list[mode] * unt.Phi0 / x0)
                        term_hermite_signs = np.where(term_hermitenorm != 0, np.sign(term_hermitenorm), 0)
                        term_hermitenorm_log = np.where(term_hermitenorm != 0, np.log(np.abs(term_hermitenorm)), 0)
                    else:
                        term_hyper = hyperu(-0.5 * n,
                                            -0.5,
                                            (phi_list[mode] * unt.Phi0 / x0)**2)
                        term_hermite_signs = np.where(term_hyper != 0, np.power(np.sign(phi_list[mode]), n), 0)
                        term_hermitenorm_log = np.where(term_hyper != 0, -(n/2) * np.log(2) * np.log(np.abs(term_hyper)), 0)

                    # Resort to mpmath library if vectorized SciPy code fails
                    if not np.all(np.isfinite(term_hermitenorm_log)):
                        bad_pos = ~np.isfinite(term_hermitenorm_log)
                        it = np.nditer(term_hermitenorm_log, flags=['multi_index'])
                        for _ in it:
                            idx = it.multi_index
                            if bad_pos[idx]:
                                hermite_val = mpmath.hermite(n, phi_list[mode][idx] * unt.Phi0 / x0)
                                if hermite_val == 0:
                                    term_hermite_signs[idx] = 0
                                    term_hermitenorm_log[idx] = 0
                                else:
                                    term_hermite_signs[idx] = mpmath.sign(hermite_val)
                                    term_hermitenorm_log[idx] = mpmath.log(mpmath.fabs(hermite_val)) - (n/2) * np.log(2)

                    term_log = (
                        coeff_log
                        + (-(phi_list[mode]*unt.Phi0/x0)**2/2)
                        + term_hermitenorm_log
                    )

                    term *= term_hermite_signs * np.exp(term_log)

            state += term

        state = np.squeeze(state)

        # transposing the first two modes
        if len(state.shape) > 1:
            indModes = list(range(len(state.shape)))
            indModes[0] = 1
            indModes[1] = 0
            state = state.transpose(*indModes)

        return state

    def coupling_op(
        self,
        ctype: str,
        nodes: Tuple[int, int],
        force_use_qutip = False,
    ) -> Union[Qobj, Tensor]:
        """Return the capacitive or inductive coupling operator related to the
        specified nodes. The output is in ``qutip.Qobj`` when using the
        ``'NumPy'`` engine and as a ``torch.Tensor`` in the ``PyTorch`` engine.

        Parameters
        ----------
            ctype:
                Coupling type which is either ``"capacitive"`` or
                ``"inductive"``.
            nodes:
                A tuple of circuit nodes to which we want to couple.

        Returns
        ----------
            Coupling operator of type ``ctype`` between nodes in ``nodes``.
        """

        if ctype not in ['capacitive', 'inductive']:
            raise ValueError("The coupling type must be either 'capacitive' or "
                             "'inductive'.")
        if not (isinstance(nodes, tuple) or isinstance(nodes, list)):
            raise ValueError('Nodes must be a tuple of integers.')

        def conditional_cast(op):
            if get_optim_mode() and not force_use_qutip:
                return sqf.qobj_to_tensor(op)
            return op

        def sp_add(a, b):
            if a == 0:
                return b
            return a + b

        node1 = nodes[0]
        node2 = nodes[1]

        if ctype == 'capacitive':
            K = sqf.mat_mul(sqf.mat_inv(self.C), self.R)
        elif ctype == 'inductive':
            K = self.S

        if force_use_qutip:
            K = sqf.to_numpy(K)

        op = 0
        # for the case that we have ground in the edge
        if 0 in nodes:
            node = node1 + node2
            if ctype == 'capacitive':
                for i in range(self.n):
                    op = sp_add(
                        op, 
                        K[node - 1, i] * conditional_cast(self._memory_ops['Q'][i])
                    )
            if ctype == 'inductive':
                for i in range(self.n):
                    op = sp_add(
                        op,
                        K[node - 1, i] * conditional_cast(self._memory_ops['phi'][i])
                    )

        else:
            if ctype == 'capacitive':
                for i in range(self.n):
                    op = sp_add(
                        op,
                        (K[node2 - 1, i] - K[node1 - 1, i]) * conditional_cast(self._memory_ops['Q'][i])
                    )
            if ctype == 'inductive':
                for i in range(self.n):
                    op = sp_add(
                        op,
                        (K[node1 - 1, i] - K[node2 - 1, i])* conditional_cast(self._memory_ops['phi'][i])
                    )

        return self._squeeze_op(op)

    def matrix_elements(
        self,
        ctype: str,
        nodes: Tuple[int, int],
        states: Tuple[int, int],
    ) -> float:
        """
        Return the matrix element of two eigenstates for either capacitive
        or inductive coupling.

        Parameters
        ----------
            ctype:
                Coupling type which is either ``"capacitive"`` or
                ``"inductive"``.
            nodes:
                A tuple of circuit nodes to which we want to couple.
            states:
                A tuple of indices of eigenstates for which we want to
                calculate the matrix element.

        Returns
        ----------
            Matrix element between eigenstates in ``states`` for coupling
            operator of type ``ctype`` between nodes in ``nodes``.
        """
        if len(self._evecs) == 0:
            raise CircuitStateError('Please diagonalize the circuit first.')

        state1 = self._evecs[states[0]]
        state2 = self._evecs[states[1]]

        # get the coupling operator
        op = self.coupling_op(ctype, nodes)

        # return (state1.dag() * op * state2).data[0, 0] (original)
        return sqf.operator_inner_product(state1, op, state2) # (modified)

    @staticmethod
    def _dephasing(A: float, partial_omega: float) -> float:
        """
        Calculate a dephasing rate of arbitrary type given a noise amplitude
        and eigenfrequency derivative.

        Parameters
        ----------
            A:
                Noise Amplitude
            partial_omega:
                The derivatives of angular frequency with respect to the
                noisy parameter

        Returns
        ----------
            Dephasing rate specified by ``A`` and ``partial_omega``.
        """
        return (sqf.abs(partial_omega * A)
                * np.sqrt(2 * np.abs(np.log(ENV['omega_low'] * ENV['t_exp']))))

    def _dec_rate_flux_np(self, states: Tuple[int, int]) -> float:
        """
        Calculate dephasing rate due to flux noise.

        Parameters
        ----------
            states:
                A tuple of state to calculate the decoherence rate

        Returns
        ----------
            Flux dephasing rate between ``states``.
        """
        decay = 0
        for loop in self.loops:
            partial_omega = self._get_partial_omega_mn(loop, states=states)
            decay = decay + self._dephasing(loop.A, partial_omega)

        return decay

    def _dec_rate_charge_np(self, states: Tuple[int, int]) -> float:
        """
        Calculate dephasing rate due to charge noise.

        Parameters
        ----------
            states:
                A tuple of state to calculate the decoherence rate

        Returns
        ----------
            Charge dephasing rate between ``states``.
        """
        state_m= self._evecs[states[0]]
        state_n = self._evecs[states[1]]

        decay = 0
        for i in range(self.n):
            op = 0
            if self._is_charge_mode(i):
                for j in range(self.n):
                    op += (self.cInvTrans[i, j] * self._memory_ops['Q'][j] / np.sqrt(unt.hbar))

                partial_omega = sqf.abs(sqf.operator_inner_product(state_m, op, state_m)
                                        - sqf.operator_inner_product(state_n, op, state_n))
                A = self.charge_islands[i].A * 2 * unt.e
                decay = decay + self._dephasing(A, partial_omega)

        return decay

    def _dec_rate_cc_np(self, states: Tuple[int, int]) -> float:
        """
        Calculate dephasing rate due to critical current noise.

        Parameters
        ----------
            states:
                A tuple of state to calculate the decoherence rate

        Returns
        ----------
            Critical current dephasing rate between ``states``.
        """
        decay = 0
        for el, b_id in self._memory_ops['cos']:
            partial_omega = self._get_partial_omega_mn(
                el, states=states, _b_id=b_id
            )
            A = el.A * el.get_value()
            decay = decay + self._dephasing(A, partial_omega)

        return decay

    def dec_rate(
        self,
        dec_type: str,
        states: Tuple[int, int],
        total: bool = True
    ) -> float:
        """ Return the decoherence rate in [1/s] between each two eigenstates
        for different types of depolarization and dephasing.

        Parameters
        ----------
            dec_type:
                decoherence type that can be: ``"capacitive"`` for capacitive
                loss; ``"inductive"`` for inductive loss; `"quasiparticle"` for
                quasiparticle loss; ``"charge"`` for charge noise, ``"flux"``
                for flux noise; and ``"cc"`` for critical current noise.
            states:
                A tuple of eigenstate indices, for which we want to
                calculate the decoherence rate. For example, for ``states=(0,
                1)``, we calculate the decoherence rate between the ground
                state and the first excited state.
            total:
                if ``False`` return a decoherence rate associated with a
                transition from state m to state n for ``states=(m, n)``. If
                ``True`` return a decoherence rate associated with both m to n
                and n to m transitions.

        Returns
        ----------
            Decoherence/dephasing rate between ``states`` specified by 
            ``dec_type``.
        """
        if len(self._efreqs) == 0:
            raise CircuitStateError('Please diagonalize the circuit first.')

        omega1 = self._efreqs[states[0]]
        omega2 = self._efreqs[states[1]]

        state1 = self._evecs[states[0]]
        state2 = self._evecs[states[1]]

        omega = sqf.abs(omega2 - omega1)

        decay = sqf.zero(dtype=torch.float64, requires_grad=True)

        # prevent the exponential overflow (exp(709) is the biggest number
        # that numpy can calculate)
        alpha = unt.hbar * omega / (unt.k_B * ENV['T'])
        if alpha > 709:
            logger.info('Omega=%.2e exceeded threshold; approximating '
                        + 'spectral density function.', sqf.to_numpy(omega))
            down = 2
            up = 0
        else:
            down = (1 + 1 / sqf.tanh(alpha / 2))
            up = down * sqf.exp(-alpha)

        # for temperature-dependent loss
        if not total:
            if states[0] > states[1]:
                tempS = down
            else:
                tempS = up
        else:
            tempS = down + up

        if dec_type == 'capacitive':
            for edge in self.elements.keys():
                for el in self.elements[edge]:
                    if isinstance(el, Capacitor):
                        cap = el
                    else:
                        cap = el.cap
                    if cap.Q:
                        decay = decay + tempS * cap.get_value() / cap.Q(omega) * sqf.abs(
                            self.matrix_elements(
                                'capacitive', edge, states)) ** 2
        elif dec_type == 'inductive':
            for el, _ in self._memory_ops['ind_hamil']:
                op = self._memory_ops['ind_hamil'][(el, _)]
                Q = el.Q(omega, ENV['T'])
                if np.isnan(sqf.to_numpy(Q)):
                    logger.warning('Calculated Q for %s was NaN', el)
                    decay = decay + 0
                else:
                    decay = decay + tempS / Q / el.get_value() * sqf.abs(
                        sqf.operator_inner_product(state1, op, state2)) ** 2
        elif dec_type == 'quasiparticle':
            for el, b_id in self._memory_ops['sin_half']:
                op = self.op('sin_half', {'el': el, 'b_id': b_id})
                Y = el.Y(omega, ENV['T'])
                if np.isnan(sqf.to_numpy(Y)):
                    logger.warning('Calculated Y for %s was NaN', el)
                    decay = decay + 0
                else:
                    decay = decay + (
                        tempS * Y * omega * el.get_value()
                        * unt.hbar
                        * sqf.abs(sqf.operator_inner_product(state1, op, state2)) ** 2
                    )
        elif dec_type == 'charge':
            if get_optim_mode():
                decay = decay + sqtorch.dec_rate_charge_torch(self, states)
            else:
                decay = decay + self._dec_rate_charge_np(states)
        elif dec_type == 'cc':
            if get_optim_mode():
                decay = decay + sqtorch.dec_rate_cc_torch(self, states)
            else:
                decay = decay + self._dec_rate_cc_np(states)
        elif dec_type == 'flux':
            if get_optim_mode():
                decay = decay + sqtorch.dec_rate_flux_torch(self, states)
            else:
                decay = decay + self._dec_rate_flux_np(states)
        else:
            raise ValueError(f'The decoherence type {dec_type} is not supported.')

        return decay

    def _get_quadratic_Q(self, A: ndarray) -> Qobj:
        """Return quadratic form of 1/2 * Q^T * A * Q

        Parameters
        ----------
            A:
                ndarray matrix that specifies the coefficient for
                quadratic expression.

        Returns
        ----------
            Quadratic form with charge operators using ``A``.
        """

        op = 0

        for i in range(self.n):
            for j in range(self.n-i):
                if j == 0:
                    op += 0.5 * A[i, i+j] * self._memory_ops['QQ'][i][j]
                elif j > 0:
                    op += A[i, i+j] * self._memory_ops['QQ'][i][j]

        return op

    def _get_quadratic_phi(self, A: ndarray) -> Qobj:
        """Get quadratic form of 1/2 * phi^T * A * phi

        Parameters
        ----------
            A:
                ndarray matrix that specifies the coefficient for
                quadratic expression.

        Returns
        ----------
            Quadratic form with flux operators using ``A``.
        """

        op = 0

        # number of harmonic modes
        n_H = len(self.omega != 0)

        for i in range(n_H):
            for j in range(n_H):
                phi_i = self._memory_ops['phi'][i].copy()
                phi_j = self._memory_ops['phi'][j].copy()
                if i == j:
                    op += 0.5 * A[i, i] * phi_i**2
                elif j > i:
                    op += A[i, j] * phi_i * phi_j

        return op

    def _get_partial_H(
        self,
        el: Union[Capacitor, Inductor, Junction, Loop],
        _b_id: Optional[int] = None,
    ) -> Qobj:
        """
        Compute the gradient of the Hamiltonian with respect to elements or
        loop as ``qutip.Qobj`` format.
        Parameters
        ----------
            el:
                Element of a circuit that can be either ``Capacitor``,
                ``Inductor``, ``Junction``, or ``Loop``.
            _b_id:
                Optional integer to indicate which row of the B matrix
                (per-element external flux distribution) to use. This specifies
                which JJ of the circuit to consider specifically (ex. for
                critical current noise calculation).

        Returns
        ----------
            Partial derivative of Hamiltonian with respect to ``el``.
        """

        partial_H = 0

        if isinstance(el, Capacitor):
            cInv = np.linalg.inv(sqf.to_numpy(self.C))
            A = -self.R.T @ cInv @ self.partial_mats[el] @ cInv @ self.R
            partial_H += self._get_quadratic_Q(A)

        elif isinstance(el, Inductor):

            A = -self.S.T @ self.partial_mats[el]  @ self.S
            partial_H += self._get_quadratic_phi(A)

            for edge, el_ind, b_id in self.elem_keys[Inductor]:
                if el == el_ind:

                    phi = self._get_external_flux_at_element(b_id)

                    partial_H += -(self._memory_ops['ind_hamil'][(el, b_id)]
                                   / np.squeeze(sqf.to_numpy(el.get_value()))**2 / np.sqrt(unt.hbar)
                                   * (unt.Phi0/2/np.pi) * phi)

        elif isinstance(el, Loop):

            loop_idx = self.loops.index(el)

            for edge, el_ind, b_id in self.elem_keys[Inductor]:
                partial_H += (
                    self.B[b_id, loop_idx]
                    * self._memory_ops["ind_hamil"][(el_ind, b_id)]
                    / sqf.to_numpy(el_ind.get_value())
                    * unt.Phi0 / np.sqrt(unt.hbar) / 2 / np.pi
                )

            for edge, el_JJ, b_id, w_id in self.elem_keys[Junction]:
                partial_H += (self.B[b_id, loop_idx] * sqf.to_numpy(el_JJ.get_value())
                              * self._memory_ops['sin'][(el_JJ, b_id)])

        elif isinstance(el, Junction):

            for _, el_JJ, b_id, w_id in self.elem_keys[Junction]:

                if el == el_JJ and _b_id is None:
                    partial_H += -self._memory_ops['cos'][(el, b_id)]

                elif el == el_JJ and _b_id == b_id:
                    partial_H += -self._memory_ops['cos'][(el, b_id)]

        return partial_H

    def get_partial_omega(
        self,
        el: Union[Capacitor, Inductor, Junction, Loop],
        m: int,
        subtract_ground: bool = True,
        _b_id: Optional[int] = None,
    ) -> float:
        """Return the gradient of the eigen angular frequency with respect to
        elements or loop as ``qutip.Qobj`` format.

        Parameters
        ----------
            el:
                Element of a circuit that can be either ``Capacitor``,
                ``Inductor``, ``Junction``, or ``Loop``.
            m:
                Integer specifies the eigenvalue. for example ``m=0`` specifies
                the ground state and ``m=1`` specifies the first excited state.
            subtract_ground:
                If ``True``, it subtracts the ground state frequency from the
                desired frequency.
            _b_id:
                Optional integer to indicate which row of the B matrix
                (per-element external flux distribution) to use. This specifies
                which JJ of the circuit to consider specifically (ex. for
                critical current noise calculation).

        Returns
        ----------
            Partial derivative of eigenfrequency ``m`` with respect to ``el``,
            in units of angular frequency.
        """
        if len(self._evecs) == 0:
            raise CircuitStateError('Please diagonalize the circuit first.')

        state_m = self._evecs[m]
        partial_H = self._get_partial_H(el, _b_id)
        partial_omega_m = sqf.operator_inner_product(
            state_m, partial_H, state_m
        )

        if subtract_ground:
            state_0 = self._evecs[0]
            partial_omega_0 = sqf.operator_inner_product(
                state_0, partial_H, state_0
            )

            return sqf.real(partial_omega_m - partial_omega_0)
        else:
            return sqf.real(partial_omega_m)

    def _get_partial_omega_mn(
        self,
        el: Union[Capacitor, Inductor, Junction, Loop],
        states: Tuple[int, int],
        _b_id: Optional[int] = None,
    ) -> float:
        """Return the gradient of the eigen angular frequency with respect to
        elements or loop as ``qutip.Qobj`` format. Note that if
        ``states=(m, n)``, it returns ``partial_omega_m - partial_omega_n``.

        Parameters
        ----------
            el:
                Element of a circuit that can be either ``Capacitor``,
                ``Inductor``, ``Junction``, or ``Loop``.
            states:
                Integers indicating indices of eigenenergies to calculate
                the difference of.
            _b_id:
                Optional integer to indicate which row of the B matrix (external
                flux distribution of that element) to use. This specifies which
                JJ of the circuit to consider specifically (ex. for critical
                current noise calculation).

        Returns
        ----------
            Partial derivative of the energy difference between ``states``
            with respect to ``el``, in units of angular frequency.
        """

        state_m = self._evecs[states[0]]
        state_n = self._evecs[states[1]]

        partial_H = self._get_partial_H(el, _b_id)

        partial_omega_m = sqf.operator_inner_product(
            state_m, partial_H, state_m
        )
        partial_omega_n = sqf.operator_inner_product(
            state_n, partial_H, state_n
        )

        partial_omega_mn = partial_omega_m - partial_omega_n
        # assert sqf.imag(partial_omega_mn)/sqf.real(partial_omega_mn) < 1e-6

        return sqf.real(partial_omega_mn)

    def get_partial_vec(
        self,
        el: Union[Element, Loop],
        m: int,
        epsilon=1e-12
    ) -> Qobj:
        """Return the gradient of the eigenvectors with respect to
        elements or loop as ``qutip.Qobj`` format.

        Parameters
        ----------
            el:
                Element of a circuit that can be either ``Capacitor``,
                ``Inductor``, ``Junction``, or ``Loop``.
            m:
                Integer specifies the eigenvalue. for example ``m=0`` specifies
                the ground state and ``m=1`` specifies the first excited state.

        Returns
        ----------
            Partial derivative of the ``m``th eigenvector, with respect to
            ``el``.
        """
        if len(self._evecs) == 0:
            raise CircuitStateError('Please diagonalize the circuit first.')

        state_m = self._evecs[m]

        n_eig = len(self._evecs)

        partial_H = self._get_partial_H(el)
        partial_state = sqf.zeros(state_m.shape)

        for n in range(n_eig):
            if n == m:
                continue
            state_n = self._evecs[n]

            delta_omega = sqf.to_numpy(self._efreqs[m] - self._efreqs[n])

            partial_state += (
                sqf.operator_inner_product(state_n, partial_H, state_m)
                * state_n
                / (delta_omega + epsilon)
            )

        return partial_state
