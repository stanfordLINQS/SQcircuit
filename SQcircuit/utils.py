"""utils.py module with functions implemented in both PyTorch and numpy,
depending on optimization mode."""

import numpy as np
import qutip as qt
import scipy
import torch

from SQcircuit.settings import OPTIM_MODE

def _vectorize(circuit) -> torch.Tensor:
    """Converts an ordered dictionary of element values for a given circuit into Tensor format.
    Parameters
    ----------
        circuit:
            A circuit to vectorize.
    """
    elements = list(circuit.elements.values())[0]
    element_values = [element.get_value(element_units=False) for element in elements]
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
    initial_element_vals = [element.get_value(element_units=False) for element in elements]

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
            values_units = [(element_tensors[idx].numpy() / elements[idx].get_unit_scale(), elements[idx].unit)
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

def qabs(x):
    if OPTIM_MODE:
        return torch.abs(x)
    return np.abs(x)

def qtanh(x):
    if OPTIM_MODE:
        return torch.tanh(x)
    return np.tanh(x)

def qexp(x):
    if OPTIM_MODE:
        return torch.exp(x)
    return np.exp(x)

def qsqrt(x):
    if OPTIM_MODE:
        return torch.sqrt(x)
    return np.sqrt(x)

def qmat_inv(A):
    if OPTIM_MODE:
        return torch.linalg.inv(A)
    return np.linalg.inv(A)

def qinit_op(shape):
    if OPTIM_MODE:
        return torch.zeros(shape)
    return qt.Qobj()

def qzeros(shape):
    if OPTIM_MODE:
        return torch.zeros(shape)
    return np.zeros(shape)

def qarray(object):
    if OPTIM_MODE:
        return torch.array(object)
    return np.array(object)

def qsum(a):
    if OPTIM_MODE:
        return torch.sum(a)
    return np.sum(a)

def qsort(a):
    if OPTIM_MODE:
        return torch.sort(a)
    return np.sort(a)

def qcast(value):
    if OPTIM_MODE:
        return torch.Tensor(value, requires_grad = True)
    return value

def qnormal(mean, var):
    if OPTIM_MODE:
        return torch.normal(mean, var)
    return np.random.normal(mean, var, 1)[0]