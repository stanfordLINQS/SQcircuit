from typing import List, Optional, Tuple, Union

import numpy as np
import qutip as qt
import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from SQcircuit import Capacitor, Element, Inductor, Junction, Loop
from SQcircuit.noise import ENV
from SQcircuit import functions as sqf
from SQcircuit import units as unt

SupportedGradElements = Union[Capacitor, Inductor, Junction, Loop]
EIGENVECTOR_MAX_GRAD = None

###############################################################################
# Interfaces to custom Torch functions
###############################################################################

def eigencircuit(circuit: 'Circuit', n_eig: int) -> Tensor:
    """Given a circuit, computes the concatenated tensor including both 
    eigenvalues and eigenvectors of a circuit using the ``EigenSolver`` class.

    Parameters
    ----------
        circuit:
            A circuit for which the eigensystem will be solved.
        n_eig:
            Number of eigenvalues to output. The lower `n_eig`, the
                faster `SQcircuit` finds the eigenvalues.

    Returns
    ----------
        A concatenated tensor of the eigenfrequencies and eigenvectors
    """
    return EigenSolver.apply(
        torch.stack(circuit.parameters) if circuit.parameters else torch.tensor([]),
        circuit,
        n_eig,
        EIGENVECTOR_MAX_GRAD
    )


def set_max_eigenvector_grad(n: Optional[int]) -> None:
    """Sets the maximum number of eigenvectors to compute the gradient for when
    calling ``diag``. Setting this to ``None`` will compute the gradient for
    all eigenvectors. 
    
    Parameters
    ----------
        n:
            The maximum number of eigenvectors to compute the gradient for, or
            ``None``.
    """
    global EIGENVECTOR_MAX_GRAD
    EIGENVECTOR_MAX_GRAD = n


def dec_rate_cc_torch(circuit: 'Circuit', states: Tuple[int, int]) -> Tensor:
    """Given a circuit, computes critical current dephasing rate using the
    ``DecRateCC`` class.

    Parameters
    ----------
        circuit:
            A circuit to compute the dephasing rate of.
        states:
            A tuple of states to compute the dephasing between.

    Returns
    ----------
        The critical current dephasing rate of ``circuit`` between ``states``.
    """
    return DecRateCC.apply(
        torch.stack(circuit.parameters) if circuit.parameters else torch.tensor([]),
        circuit,
        states
    )


def dec_rate_charge_torch(
        circuit: 'Circuit',
        states: Tuple[int, int]
) -> Tensor:
    """Given a circuit, computes charge dephasing rate using the
    ``DecRateCharge`` class.

    Parameters
    ----------
        circuit:
            A circuit to compute the dephasing rate of.
        states:
            A tuple of states to compute the dephasing between.

    Returns
    ----------
        The charge dephasing rate of ``circuit`` between ``states``.
    """
    return DecRateCharge.apply(
        torch.stack(circuit.parameters) if circuit.parameters else torch.tensor([]),
        circuit,
        states
    )


def dec_rate_flux_torch(circuit: 'Circuit', states: Tuple[int, int]) -> Tensor:
    """Given a circuit, computes charge dephasing rate using the
    ``DecRateFlux`` class.

    Parameters
    ----------
        circuit:
            A circuit to compute the dephasing rate of.
        states:
            A tuple of states to compute the dephasing between.

    Returns
    ----------
        The flux rate of ``circuit`` between ``states``.
    """
    return DecRateFlux.apply(
        torch.stack(circuit.parameters) if circuit.parameters else torch.tensor([]),
        circuit,
        states
    )


###############################################################################
# Eigensolver
###############################################################################

class EigenSolver(Function):
    """
    Subclass of ``torch.Function`` which implements (once-differentiable)
    forward and backwards pass for diagonalizing a ``SQcircuit.Circuit.``
    object.
    """

    @staticmethod
    def forward(ctx,
                element_tensors: Tensor,
                circuit: 'Circuit',
                n_eig: int,
                eigenvector_max_grad: Optional[int]) -> Tensor:
        """
        Forward pass for diagonalizing a `Circuit` object.

        Parameters
        ----------
            ctx:
                The Torch context argument
            element_tensors:
                The elements of ``circuit`` which require gradient, as a tensor
                given by ``circuit.parameters``. This argument is not strictly
                used by the forward pass, but the gradient is computed for it,
                it must be an input by convention.
            circuit:
                The ``SQcircuit`` circuit to diagonalize.
            n_eig:
                The number of eigenvalues/vectors to compute.
            eigenvector_max_grad:
                The maximum number of eigenvectors to compute the gradient
                for.

        Returns
        ----------
            A concatenated tensor of the eigenvalues and eigenvectors.
        """

        # Compute forward pass for eigenvalues/vectors
        eigenvalues, eigenvectors = circuit._diag_np(n_eig=n_eig)

        # Construct eigenvalues tensor
        eigenvalues = [eigenvalue * 2 * np.pi * unt.get_unit_freq() for
                       eigenvalue in eigenvalues]
        eigenvalue_tensors = [torch.as_tensor(eigenvalue) for
                              eigenvalue in eigenvalues]
        eigenvalue_tensor = torch.stack(eigenvalue_tensors)
        eigenvalue_tensor = torch.unsqueeze(eigenvalue_tensor, dim=-1)
        # Construct eigenvectors tensor
        eigenvector_tensors = [torch.as_tensor(eigenvector.full()) for
                               eigenvector in eigenvectors]
        eigenvector_tensor = torch.squeeze(torch.stack(eigenvector_tensors))


        # Setup context -- needs to be done after diagonalization so that
        # memory ops are filled
        eigenvalues = torch.real(eigenvalue_tensor)
        ctx.circuit = circuit.safecopy()

        ## Save eigenvalues, vectors into `ctx` circuit
        ctx.circuit._efreqs = eigenvalues
        ctx.circuit._evecs = eigenvector_tensor

        ## Number of eigenvalues
        ctx.n_eig = n_eig
        ## Number of eigenvectors to use in computation of partial_omega
        if eigenvector_max_grad is None:
            ctx.eigenvector_max_grad = ctx.n_eig
        else:
            ctx.eigenvector_max_grad = eigenvector_max_grad
        ## Output shape
        ctx.out_shape = element_tensors.shape

        # Return concatenated Tensor
        return torch.cat([eigenvalue_tensor, eigenvector_tensor], dim=-1)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output) -> Tuple[Tensor]:
        """
        Backward pass for diagonalizing a `Circuit` object.

        Parameters
        ----------
            ctx:
                The Torch context argument
            grad_output:
                The Torch grad output

        Returns
        ----------
            The gradient for the ``element_tensors`` input for the forward pass,
            and ``None`` for all other inputs.         
        """

        # Extract key parts of `ctx`
        elements = ctx.circuit.parameters_elems
        m, n, l = (
            len(ctx.circuit.parameters), # number of parameters
            ctx.n_eig,                   # number of eigenvalues
            (grad_output.shape[1] - 1),  # length of eigenvectors
        )
        # Break grad_output into eigenvalue sub-tensor and eigenvector
        # sub-tensor
        grad_output_eigenvalue = grad_output[:, 0]
        grad_output_eigenvector = grad_output[:, 1:]

        partial_omega = torch.zeros([m, n], dtype=float)
        partial_eigenvec = torch.zeros([m, n, l], dtype=torch.complex128)
        for el_idx in range(m):
            for eigen_idx in range(n):
                # Compute backward pass for eigenvalues
                partial_omega[el_idx, eigen_idx] = ctx.circuit.get_partial_omega(
                        el=elements[el_idx],
                        m=eigen_idx,
                        subtract_ground=False
                )
                # Compute backwards pass for only _some_ of the eigenvectors.
                # This computation is expensive, and the gradient of higher
                # eigenvectors is infrequently used. (However, _computing_)
                # many eigenvectors is necessary for accurate first-order
                # gradients of _any_ eigenvector.)
                if eigen_idx < ctx.eigenvector_max_grad:
                    partial_tensor = torch.squeeze(torch.as_tensor(
                        ctx.circuit.get_partial_vec(
                            el=elements[el_idx],
                            m=eigen_idx
                        )
                    ))
                    partial_eigenvec[el_idx, eigen_idx, :] = partial_tensor
        eigenvalue_grad = torch.sum(
            partial_omega * torch.conj(grad_output_eigenvalue), axis=-1)
        eigenvector_grad = torch.sum(
            partial_eigenvec * torch.conj(grad_output_eigenvector),
            axis=(-1, -2))

        return torch.real(eigenvalue_grad + eigenvector_grad).view(ctx.out_shape), None, None, None

###############################################################################
# Decoherence rate helper functions
###############################################################################

def partial_squared_omega(
    cr: 'Circuit',
    grad_el: SupportedGradElements,
    partial_H: qt.Qobj,
    partial_H_squared: qt.Qobj,
    states: Tuple[int, int]
) -> float:
    """ Calculates the second derivative of the difference between the ``m``th 
    and ``n``th eigenfrequencies with respect to an arbitrary parameter ``x``
    and ``grad_el``.

    Parameters
    ----------
        cr:
            The ``Circuit`` object to differentiate the eigenfrequencies of
        grad_el:
            The element to take the gradient of the frequencies with respect to
        partial_H:
            The derivative of the circuit Hamiltonian with respect to the
            arbitrary parameter.
        partial_H_squared:
            The second derivative of the circuit Hamiltonian with respect to 
            the arbitrary parameter and ``grad_el``.
        states:
            The indices ``(m, n)`` of the eigenfrequencies to differentiate.
    
    Returns
    ----------
        The second derivative of the eigenfrequency difference.         
    """

    m, n = states

    state_m = cr.evecs[m]
    partial_state_m = cr.get_partial_vec(grad_el, m)
    state_n = cr.evecs[n]
    partial_state_n = cr.get_partial_vec(grad_el, n)

    # Compute the first term of the second-order derivative, from differentiating
    # the state
    if partial_H == 0:
        # operator_inner_product behaves badly if passed a float
        p2_omega_1 = 0
    else:
        p2_omega_1 = 2 * np.real(
            sqf.operator_inner_product(partial_state_m, partial_H, state_m)
            - sqf.operator_inner_product(partial_state_n, partial_H, state_n)
        )

    # Compute the second term of the second-order derivative, from differentating
    # partial_H
    if partial_H_squared == 0:
        p2_omega_2 = 0
    else:
        p2_omega_2 = (
            sqf.operator_inner_product(state_m, partial_H_squared, state_m)
            - sqf.operator_inner_product(state_n, partial_H_squared, state_n)
        )

    # Return sum
    p2_omega = p2_omega_1 + p2_omega_2
    ## The eigenfrequencies (and hence derivatives) should be real since H
    ## is Hermitian, but numerical imprecision can result in small complex
    ## components.
    return sqf.to_numpy(p2_omega.real)


def partial_dephasing_rate(
    A: float,
    partial_A: float,
    partial_omega_mn: float,
    partial_squared_omega_mn: float
) -> float:
    """Calculate the derivative of the dephasing rate with noise amplitude
    ``A`` that depends on ``partial_omega_mn``.

    Parameters
    ----------
        A:
            The noise amplitude of the dephasing
        partial_A:
            The derivative of ``A`` with respect to an external parameter
        partial_omega_mn:
            The derivative of the difference between eigenfrequencies 
            used to calculate the dephasing rate
        partial_squared_omega_mn:
            The derivative of ``partial_omega_mn`` with respect to an external
            parameter

    Returns
    ----------
        The derivative of dephasing rate based on the input parameters.
    """
    return (
        np.sign(partial_omega_mn)
        * np.sqrt(2 * np.abs(np.log(ENV["omega_low"] * ENV["t_exp"])))
        * (partial_A * partial_omega_mn + A * partial_squared_omega_mn)
    )


def get_B_indices(
    cr: 'Circuit',
    el: Union[Junction, Inductor]
) -> List[int]:
    """
    Return the list of ``B_idx``'s with the element ``el`` (the same element
    could be placed at multiple branches).

    Parameters
    ----------
        cr:
            The ``Circuit`` object.
        el:
            The element to find in ``cr``.

    Returns
    ----------
        A list of rows of the ``B`` matrix the element is associated with.
    """
    B_indices = []
    if isinstance(el, Junction):
        for _, el_JJ, B_idx, _ in cr.elem_keys[Junction]:
            if el_JJ is el:
                B_indices.append(B_idx)
    elif isinstance(el, Inductor):
        for _, el_ind, B_idx in cr.elem_keys[Inductor]:
            if el_ind is el:
                B_indices.append(B_idx)

    return B_indices


###############################################################################
# Charge noise
###############################################################################

def partial_H_ng(
    cr: 'Circuit',
    charge_idx: int
) -> qt.Qobj:
    """Calculates the  derivative of the Hamiltonian of ``cr`` with 
    respect to the gate charge on the ``charge_idx`` charge mode.

    Parameters
    ----------
        cr:
            The ``Circuit`` object to differentiate the Hamiltonian of
        charge_idx:
            The charge mode whose gate charge to differentiate with respect to

    Returns
    ----------
        The partial derivative of ``cr``'s Hamiltonian with respect to gate
        charge.
    """
    if not cr._is_charge_mode(charge_idx):
        raise ValueError('The mode index passed is not a charge mode!')

    op = 0
    for j in range(cr.n):
        op += (
            cr.cInvTrans[charge_idx, j]
            * cr._memory_ops["Q"][j]
            / np.sqrt(unt.hbar)
        )
    return op


def partial_squared_H_ng(
    cr: 'Circuit',
    charge_idx: int,
    grad_el: SupportedGradElements
) -> qt.Qobj:
    """Calculates the second derivative of the Hamiltonian of ``cr`` with 
    respect to the gate charge on the ``charge_idx`` charge mode and ``grad_el``.

    Parameters
    ----------
        cr:
            The ``Circuit`` object to differentiate the Hamiltonian of
        charge_idx:
            The charge mode whose gate charge to differentiate with respect to
        grad_el:
            The circuit element to differentiate with respect to

    Returns
    ----------
        The second derivative of the Hamiltonian with respect to gate charge and
        ``grad_el``.
    """
    if not cr._is_charge_mode(charge_idx):
        raise ValueError('The mode index passed is not a charge mode!')

    # The charge operators only multiply the capacitance matrix
    if not isinstance(grad_el, Capacitor):
        return 0

    cInv = np.linalg.inv(sqf.to_numpy(cr.C))
    A = cInv @ cr.partial_mats[grad_el] @ cInv

    op = 0
    for j in range(cr.n):
        op += A[charge_idx, j] * cr._memory_ops["Q"][j] / np.sqrt(unt.hbar)
    return -op


def partial_omega_ng(
    cr: 'Circuit',
    charge_idx: int,
    states: Tuple[int, int]
) -> float:
    """Calculates the derivative of the difference between the ``m``th
    and ``n``th eigenfrequencies of ``cr`` with respect to the gate charge of
    the ``charge_idx`` charge mode.

    Parameters
    ----------
        cr:
            The `Circuit`` object to differentiate the eigenfrequencies of
        charge_idx:
            The charge mode whose gate charge to differentiate with respect to
        states:
            The numbers ``(m, n)`` of the eigenfrequencies to differentiate.

    Returns
    ----------
        The first derivative of the eigenfrequency difference with respect to
        gate charge.   
    """
    if not cr._is_charge_mode(charge_idx):
        raise ValueError('The mode index passed is not a charge mode!')

    state_m = cr.evecs[states[0]]
    state_n = cr.evecs[states[1]]
    op = partial_H_ng(cr, charge_idx)

    partial_omega_mn = (
        sqf.operator_inner_product(state_m, op, state_m)
        - sqf.operator_inner_product(state_n, op, state_n)
    )

    # The eigenfrequencies (and hence derivatives) should be real since H
    # is Hermitian, but numerical imprecision can result in small complex
    # components.
    return sqf.to_numpy(partial_omega_mn.real)


def partial_squared_omega_mn_ng(
    cr: 'Circuit',
    charge_idx: int,
    grad_el: SupportedGradElements,
    states: Tuple[int, int]
) -> float:
    """Calculates the second derivative of the difference between the ``m``th 
    and ``n``th eigenfrequencies with respect to the gate charge on the
    ```charge_idx`` mode and ``grad_el``.

    Parameters
    ----------
        cr:
            The ``Circuit`` object to differentiate the eigenfrequencies of
        charge_idx:
            The charge mode whose gate charge to differentiate with respect to
        grad_el:
            The circuit element to differentiate with respect to
        states:
            The numbers ``(m, n)`` of the eigenfrequencies to differentiate.

    Returns
    ----------
        The second derivative of the eigenfrequency difference with respect to
        gate charge and ``grad_el``.  
    """
    if not cr._is_charge_mode(charge_idx):
        raise ValueError('The mode index passed is not a charge mode!')

    partial_H = partial_H_ng(cr, charge_idx)
    partial_H_squared = partial_squared_H_ng(cr, charge_idx, grad_el)

    return partial_squared_omega(
        cr,
        grad_el,
        partial_H,
        partial_H_squared,
        states
    )


def partial_charge_dec(
    cr: 'Circuit',
    grad_el: SupportedGradElements,
    states: Tuple[int, int]
) -> float:
    """
    Calculate the derivative of the charge dephasing rate between the states
    given in ``states`` with respect to ``grad_el``.

    Parameters
    ----------
        cr:
            The ``Circuit`` object.
        grad_el:
            The circuit element to differentiate with respect to
        states:
            A tuple ``(m, n)`` of states to consider charge decoherence between.
    
    Returns
    ----------
        The first partial derivative of the charge dephasing rate.
    """
    dec_rate_grad = 0
    for i in range(cr.n):
        if cr._is_charge_mode(i):
            partial_omega_mn = partial_omega_ng(cr, i, states)
            partial_squared_omega_mn = partial_squared_omega_mn_ng(
                cr=cr,
                charge_idx=i,
                grad_el=grad_el,
                states=states
            )
            A = cr.charge_islands[i].A * 2 * unt.e
            partial_A = 0

            dec_rate_grad += partial_dephasing_rate(
                A,
                partial_A,
                partial_omega_mn,
                partial_squared_omega_mn
            )

    return dec_rate_grad


class DecRateCharge(Function):
    """
    Torch ``Function`` wrapper to compute forward and backwards pass for the
    charge dephasing rate. 

    The forward pass is computed using an internal method of the ``Circuit``
    class, and the gradient is computed using helper functions in the
    ``torch_extensions.py`` module.
    """
    @staticmethod
    def forward(
        circuit_parameters: Tensor,
        circuit: 'Circuit',
        states: Tuple[int, int]
    ) -> Tensor:
        return torch.as_tensor(circuit._dec_rate_charge_np(states))

    @staticmethod
    def setup_context(ctx, inputs, output):
        circuit_parameters, circuit, states = inputs

        ctx.circuit = circuit.safecopy(save_eigs=True)
        ctx.states = states

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output) -> Tuple[Tensor, None, None]:
        output_grad = torch.zeros(len(ctx.circuit._parameters))

        for idx, elem in enumerate(ctx.circuit._parameters.keys()):
            output_grad[idx] = grad_output * partial_charge_dec(
                ctx.circuit,
                elem,
                ctx.states
            )

        return output_grad, None, None


###############################################################################
# Critical current noise
###############################################################################

def partial_squared_H_EJ(
    cr: 'Circuit',
    EJ_el: Junction,
    B_idx: int,
    grad_el: SupportedGradElements
) -> qt.Qobj:
    """ Calculates the second derivative of the Hamiltonian of ``cr`` with 
    respect to the Josephson energy of ``EJ_el`` charge mode and ``grad_el``.

    Parameters
    ----------
        cr:
            The ``Circuit`` object to differentiate the Hamiltonian of.
        EJ_el:
            The Josephson junction to differentiate with respect to.
        B_idx:
            The index of the ``cr.B`` matrix identifying which branch the 
            ``EJ_el`` is on.
        grad_el:
            The circuit element to differentiate with respect to.

    Returns
    ----------
        The second deritative of the Hamiltonain with respec to ``EJ_el`` and
        ``grad_el``.
    """
    if not isinstance(grad_el, Loop):
        return 0

    loop_idx = cr.loops.index(grad_el)
    return cr.B[B_idx, loop_idx] * cr._memory_ops['sin'][(EJ_el, B_idx)]


def partial_squared_omega_mn_EJ(
    cr: 'Circuit',
    EJ_el: Junction,
    B_idx: int,
    grad_el: SupportedGradElements,
    states: Tuple[int, int]
) -> float:
    """ Calculates the second derivative of the difference between the ``m``th 
    and ``n``th eigenfrequencies with respect to ``EJ_el`` and ``grad_el``.

    Parameters
    ----------
        cr:
            The ``Circuit`` object to differentiate the eigenfrequencies of.
        EJ_el:
            A Josephson junction to differentiate with respect to.
        B_idx:
            A number
        grad_el:
            A circuit element to differentiate with respect to.
        states:
            The numbers ``(m, n)`` of the eigenfrequencies to differentiate.

    Returns
    ----------
        The second derivative of the eigenfrequency difference with respect to
        ``EJ_el`` and ``grad_el``.
    """
    partial_H = cr._get_partial_H(EJ_el, _B_idx = B_idx)
    partial_H_squared = partial_squared_H_EJ(cr, EJ_el, B_idx, grad_el)

    return partial_squared_omega(
        cr,
        grad_el,
        partial_H,
        partial_H_squared,
        states
    )


def partial_cc_dec(
    cr: 'Circuit',
    grad_el: SupportedGradElements,
    states: Tuple[int, int]
) -> float:
    """Calculates the derivative of the critical current dephasing rate between
    the states given  in ``states`` with respect to ``grad_el``.

    Parameters
    ----------
        cr:
            The ``Circuit`` object.
        states:
            A tuple ``(m, n)`` of states to consider charge decoherence between.
        grad_el:
            The circuit element to differentiate with respect to.
    
    Returns
    ----------
        The derivative of the critical current dephasing rate between ``states``
        with respect to ``grad_el``.
    """

    dec_rate_grad = 0
    # Sum over all cosine operators because each Josephson junction is
    # associated with exactly one.
    for EJ_el, B_idx in cr._memory_ops['cos']:
        partial_omega_mn = sqf.to_numpy(cr._get_partial_omega_mn(
            EJ_el,
            states=states,
            _B_idx=B_idx
        ))
        partial_squared_omega_mn = partial_squared_omega_mn_EJ(
            cr,
            EJ_el,
            B_idx,
            grad_el,
            states
        )
        A = sqf.to_numpy(EJ_el.A * EJ_el.get_value())
        partial_A = EJ_el.A if grad_el is EJ_el else 0

        dec_rate_grad += partial_dephasing_rate(
            A,
            partial_A,
            partial_omega_mn,
            partial_squared_omega_mn
        )

    return dec_rate_grad


class DecRateCC(Function):
    """
    Torch ``Function`` wrapper to compute forward and backwards pass for the
    critical current dephasing rate. 

    The forward pass is computed using an internal method of the ``Circuit``
    class, and the gradient is computed using helper functions in the
    ``torch_extensions.py`` module.
    """
    @staticmethod
    def forward(
        element_tensors: Tensor,
        circuit: 'Circuit',
        states: Tuple[int, int]
    ) -> Tensor:

        return torch.as_tensor(circuit._dec_rate_cc_np(states))

    @staticmethod
    def setup_context(ctx, inputs, output):
        circuit_parameters, circuit, states = inputs

        ctx.circuit = circuit.safecopy(save_eigs=True)
        ctx.states = states

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output) -> Tuple[Tensor, None, None]:
        output_grad = torch.zeros(len(ctx.circuit._parameters))

        for idx, elem in enumerate(ctx.circuit._parameters.keys()):
            output_grad[idx] = grad_output * partial_cc_dec(
                ctx.circuit,
                elem,
                ctx.states
            )

        return output_grad, None, None


###############################################################################
# Flux noise
###############################################################################

def partial_squared_H_phi(
    cr: 'Circuit',
    loop: Loop,
    grad_el: SupportedGradElements
) -> qt.Qobj:
    """Calculates the second derivative of the Hamiltonian of ``cr`` with 
    respect to the external flux through ``loop`` and ``grad_el``.

    Parameters
    ----------
        cr:
            The ``Circuit`` object to differentiate the Hamiltonian of.
        loop:
            The loop with external flux to differentiate with respect to.
        grad_el:
            The circuit element to differentiate with respect to.
    
    Returns
    ----------
        The second derivative of the Hamiltonian with respect to external flux
        and ``grad_el``.
    """
    # Only inductors and junctions (and loops) are associated with external
    # flux.
    if isinstance(grad_el, Capacitor):
        return 0

    loop_idx = cr.loops.index(loop)
    B_indices = get_B_indices(cr, grad_el)

    H_squared = 0
    if isinstance(grad_el, Junction):
        for B_idx in B_indices:
            H_squared += cr.B[B_idx, loop_idx] * cr._memory_ops['sin'][(grad_el, B_idx)]
        return H_squared
    elif isinstance(grad_el, Inductor):
        for B_idx in B_indices:
            H_squared += (
                cr.B[B_idx, loop_idx]
                / -sqf.to_numpy(grad_el.get_value()**2)
                * unt.Phi0 / np.sqrt(unt.hbar) / 2 / np.pi
                * cr._memory_ops["ind_hamil"][(grad_el, B_idx)]
            )
        return H_squared
    elif isinstance(grad_el, Loop):
        loop_idx_1 = cr.loops.index(loop)
        loop_idx_2 = cr.loops.index(grad_el)
        for edge, el_JJ, B_idx, W_idx in cr.elem_keys[Junction]:
            H_squared += (
                sqf.to_numpy(el_JJ.get_value())
                * cr.B[B_idx, loop_idx_2]
                * cr.B[B_idx, loop_idx_1]
                * cr._memory_ops['cos'][(el_JJ, B_idx)]
            )
        return H_squared
    else:
        raise NotImplementedError


def partial_squared_omega_mn_phi(
    cr: 'Circuit',
    loop: Loop,
    grad_el: SupportedGradElements,
    states: Tuple[int, int]
) -> float:
    """Calculate the second derivative of the difference between the ``m``th 
    and `n`th eigenfrequencies with respect to ``loop`` and ``grad_el``.

    Parameters
    ----------
        cr:
            The ``Circuit`` object to differentiate the eigenfrequencies of.
        loop:
            A loop in ``cr`` to differentiate with respect to.
        grad_el:
            A circuit element to differentiate with respect to.
        states:
            The numbers ``(m, n)`` of the eigenfrequencies to differentiate.

    Returns
    ----------
        The gradient of the eigenfrequency difference with respect to
        ``grad_el`` and the flux through ``loop``.
    """
    partial_H = cr._get_partial_H(loop)
    partial_H_squared = partial_squared_H_phi(cr, loop, grad_el)

    return partial_squared_omega(
        cr,
        grad_el,
        partial_H,
        partial_H_squared,
        states
    )


def partial_flux_dec(
    cr: 'Circuit',
    grad_el: SupportedGradElements,
    states: Tuple[int, int]
) -> float:
    """Calculate the derivative of the flux dephasing rate between the states 
    given in `states` with respect to `grad_el`.

    Parameters
    ----------
        cr:
            The ``Circuit`` object.
        states:
            A tuple ``(m, n)`` of states to consider charge decoherence between.
        grad_el:
            The circuit element to differentiate with respect to.

    Returns
    ----------
        The derivative of the flux dephasing rate with respect to ``grad_el``.
    """
    dec_rate_grad = 0
    for loop in cr.loops:
        partial_omega_mn = sqf.to_numpy(cr._get_partial_omega_mn(
            loop,
            states=states
        ))
        partial_squared_omega_mn = partial_squared_omega_mn_phi(
            cr,
            loop,
            grad_el,
            states
        )

        A = loop.A
        partial_A = 0

        dec_rate_grad += partial_dephasing_rate(
            A,
            partial_A,
            partial_omega_mn,
            partial_squared_omega_mn
        )
    return dec_rate_grad


class DecRateFlux(Function):
    """
    Torch ``Function`` wrapper to compute forward and backwards pass for the
    flux dephasing rate. 

    The forward pass is computed using an internal method of the ``Circuit``
    class, and the gradient is computed using helper functions in the
    ``torch_extensions.py`` module.
    """
    @staticmethod
    def forward(
        circuit_parameters: Tensor,
        circuit: 'Circuit',
        states: Tuple[int, int]
    ) -> Tensor:
        return torch.as_tensor(circuit._dec_rate_flux_np(states))

    @staticmethod
    def setup_context(ctx, inputs, output):
        circuit_parameters, circuit, states = inputs

        ctx.circuit = circuit.safecopy(save_eigs=True)
        ctx.states = states

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output) -> Tuple[Tensor, None, None]:
        output_grad = torch.zeros(len(ctx.circuit._parameters))

        for idx, elem in enumerate(ctx.circuit._parameters.keys()):
            output_grad[idx] = grad_output * partial_flux_dec(
                ctx.circuit,
                elem,
                ctx.states
            )

        return output_grad, None, None
