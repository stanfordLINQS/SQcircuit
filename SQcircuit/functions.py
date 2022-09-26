"""utils.py module with functions implemented in both PyTorch and numpy,
depending on optimization mode."""

import numpy as np
import qutip as qt
import scipy
import torch

from collections.abc import Iterable

from SQcircuit.settings import get_optim_mode

def _vectorize(circuit) -> torch.Tensor:
    """Converts an ordered dictionary of element values for a given circuit into Tensor format.
    Parameters
    ----------
        circuit:
            A circuit to vectorize.
    """
    elements = list(circuit.elements.values())[0]
    element_values = [element.get_value() for element in elements]
    element_tensors = torch.stack(element_values)
    return element_tensors

def eigencircuit(circuit, num_eigen):
    """Given a circuit, returns Torch functions that compute the
    eigenvalues and eigenvectors of a circuit.
    Parameters
    ----------
        circuit:
            A circuit for which the eigensystem will be solved.
    """

    elements = list(circuit.elements.values())[0]
    initial_element_vals = [element.get_value() for element in elements]

    tensor_list = _vectorize(circuit)

    # TODO: Combine following two methods into one (to avoid calling diag_np twice)
    class EigenvalueSolver(torch.autograd.Function):
        @staticmethod
        def forward(ctx, element_tensors):
            elements = list(circuit.elements.values())[0]
            values_units = [(element_tensors[idx].numpy(), elements[idx].unit)
                            for idx in range(len(element_tensors))]
            circuit.update_elements(elements, values_units=values_units)
            eigenvalues, _ = circuit.diag_np(n_eig=num_eigen)
            eigenvalues = [eigenvalue * 2 * np.pi for eigenvalue in eigenvalues]
            eigenvalue_tensors = [torch.as_tensor(eigenvalue) for eigenvalue in eigenvalues]
            return torch.stack(eigenvalue_tensors)

        @staticmethod
        def backward(ctx, grad_output):
            cr_elements = list(circuit.elements.values())[0]
            m, n = tensor_list.shape[0], num_eigen
            partial_omega = torch.zeros([m, n], dtype=float)
            for el_idx in range(m):
                for eigen_idx in range(n):
                    partial_omega[el_idx, eigen_idx] = circuit._get_partial_omega(el=cr_elements[el_idx],
                                                                                 m=eigen_idx, subtract_ground=True) * \
                                                       initial_element_vals[el_idx]
            return torch.sum(partial_omega * grad_output, axis=-1)

    class EigenvectorSolver(torch.autograd.Function):
        @staticmethod
        def forward(ctx, element_tensors):
            elements = list(circuit.elements.values())[0]
            values_units = [(element_tensors[idx].numpy(), elements[idx].unit)
                            for idx in range(len(element_tensors))]
            circuit.update_elements(elements, values_units=values_units)
            _, eigenvectors = circuit.diag_np(n_eig=num_eigen)
            eigenvector_tensors = [torch.as_tensor(eigenvector.full()) for eigenvector in eigenvectors]
            return torch.squeeze(torch.stack(eigenvector_tensors))

        @staticmethod
        def backward(ctx, grad_output):
            cr_elements = list(circuit.elements.values())[0]
            m, n, l = tensor_list.shape[0], *grad_output.shape
            partial_eigenvec = torch.zeros([m, n, l])
            for el_idx in range(m):
                for eigen_idx in range(n):
                    partial_tensor = torch.squeeze(
                        torch.as_tensor(circuit._get_partial_vec(el=cr_elements[el_idx], m=eigen_idx).full()))
                    partial_eigenvec[el_idx, eigen_idx, :] = partial_tensor * initial_element_vals[
                        el_idx]  # rescale gradient based on initial value
            return torch.real(torch.sum(partial_eigenvec * torch.conj(grad_output), axis=(-1, -2)) +
                              torch.sum(torch.conj(partial_eigenvec) * grad_output, axis=(-1, -2)))

    return tensor_list, EigenvalueSolver, EigenvectorSolver

def abs(x):
    if get_optim_mode():
        return torch.abs(x)
    return np.abs(x)

def tanh(x):
    if get_optim_mode():
        return torch.tanh(x)
    return np.tanh(x)

def exp(x):
    if get_optim_mode():
        return torch.exp(x)
    return np.exp(x)

def sqrt(x):
    if get_optim_mode():
        return torch.sqrt(x)
    return np.sqrt(x)

def mat_inv(A):
    if get_optim_mode():
        return torch.linalg.inv(A)
    return np.linalg.inv(A)

def init_sparse(shape):
    if get_optim_mode():
        return torch.sparse_coo_tensor(size=shape)
    return qt.Qobj()

def init_op():
    if get_optim_mode():
        return torch.sparse_coo_tensor()
    return qt.Qobj()

def zeros(shape):
    if get_optim_mode():
        return torch.zeros(shape)
    return np.zeros(shape)

def array(object):
    if get_optim_mode():
        return torch.as_tensor(object)
    return np.array(object)

def sum(a):
    if get_optim_mode():
        return torch.sum(a)
    return np.sum(a)

def sort(a):
    if get_optim_mode():
        return torch.sort(a)
    return np.sort(a)

def cast(value):
    if get_optim_mode():
        if isinstance(value, qt.Qobj):
            return sparse_csr_to_tensor(value.data)
        return torch.tensor(value, requires_grad = True)
    return value

def normal(mean, var):
    if get_optim_mode():
        return torch.normal(mean, var)
    return np.random.normal(mean, var, 1)[0]

def numpy(input):
    if get_optim_mode():
        if isinstance(input, list):
            return [value.detach().numpy() for value in input]
        else:
            return input.detach().numpy()
    return input

def copy(x):
    if get_optim_mode():
        return x.clone()
    return x.copy()

def mat_mul(A, B):
    if get_optim_mode():
        return torch.mm(torch.as_tensor(A, dtype=torch.float32), torch.as_tensor(B, dtype=torch.float32))
    return A @ B

def sparse_csr_to_tensor(S):
    S = S.tocoo()
    values = S.data
    indices = np.vstack((S.row, S.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = S.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def sparse_mul(S, T):
    if get_optim_mode():
        if isinstance(S, qt.Qobj):
            S = sparse_csr_to_tensor(S.data)
        if isinstance(T, qt.Qobj):
            T = sparse_csr_to_tensor(T.data)
        return torch.sparse.mm(S, T)
    return S * T