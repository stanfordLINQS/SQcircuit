"""
utils_backup.py contains backup utility classes
"""

from typing import List, Tuple

from SQcircuit.circuit import Circuit
from SQcircuit.elements import Capacitor
from SQcircuit.noise import ENV
import SQcircuit.units as unt

import torch
import numpy as np


def _vectorize(circuit: Circuit) -> torch.Tensor:
    """Converts a circuit to an ordered set of PyTorch tensors, corresponding
    to the values of each element in the same order.

    Parameters
    ----------
        circuit:
            A circuit to vectorize.
    """
    elements = list(circuit.elements.values())[0]
    element_values = [element.get_value(element_units=True) for element in elements]
    element_tensors = [torch.as_tensor(element_value)
                       for element_value in element_values]
    element_tensors = torch.stack(element_tensors)
    return element_tensors

def eigencircuit(circuit: Circuit, num_eigen):
    """Solves for the eigenvalues of a circuit, returning them
    in PyTorch tensor format.

    Parameters
    ----------
        circuit:
            A circuit for which the eigensystem will be solved.
    """

    elements = list(circuit.elements.values())[0]
    initial_element_vals = [element.get_value(element_units=False) for element in elements]

    element_tensors = _vectorize(circuit)
    element_tensors.requires_grad = True

    class EigenvalueSolver(torch.autograd.Function):
        @staticmethod
        def forward(ctx, element_tensors):
            elements = list(circuit.elements.values())[0]
            values_units = [(element_tensors[idx].numpy(), elements[idx].unit)
                            for idx in range(len(element_tensors))]
            circuit.update_elements(elements, values_units=values_units)
            eigenvalues, _ = circuit.diag(n_eig=num_eigen)
            eigenvalues = [eigenvalue * 1e9 * 2 * np.pi for eigenvalue in eigenvalues]
            eigenvalue_tensors = [torch.as_tensor(eigenvalue) for eigenvalue in eigenvalues]
            return torch.stack(eigenvalue_tensors)

        @staticmethod
        def backward(ctx, grad_output):
            cr_elements = list(circuit.elements.values())[0]
            m, n = element_tensors.shape[0], num_eigen
            partial_omega = torch.zeros([m, n], dtype=float)
            for el_idx in range(m):
                for eigen_idx in range(n):
                    partial_omega[el_idx, eigen_idx] = circuit.get_partial_omega(el=cr_elements[el_idx],
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
            _, eigenvectors = circuit.diag(n_eig=num_eigen)
            eigenvector_tensors = [torch.as_tensor(eigenvector.full()) for eigenvector in eigenvectors]
            return torch.squeeze(torch.stack(eigenvector_tensors))

        @staticmethod
        def backward(ctx, grad_output):
            cr_elements = list(circuit.elements.values())[0]
            m, n, l = element_tensors.shape[0], *grad_output.shape
            partial_eigenvec = torch.zeros([m, n, l])
            for el_idx in range(m):
                for eigen_idx in range(n):
                    partial_tensor = torch.squeeze(
                        torch.as_tensor(circuit._get_partial_vec(el=cr_elements[el_idx], m=eigen_idx).full()))
                    partial_eigenvec[el_idx, eigen_idx, :] = partial_tensor * initial_element_vals[
                        el_idx]  # rescale gradient based on initial value
            return torch.real(torch.sum(partial_eigenvec * torch.conj(grad_output), axis=(-1, -2)) +
                              torch.sum(torch.conj(partial_eigenvec) * grad_output, axis=(-1, -2)))

    return element_tensors, EigenvalueSolver, EigenvectorSolver