"""
utils_backup.py contains backup utility classes
"""

from typing import List

from SQcircuit.circuit import Circuit

import torch


def vectorize(circuit: Circuit) -> List[torch.Tensor]:
    """Converts a circuit to an ordered set of PyTorch tensors, corresponding
    to the values of each element in the same order.

    Parameters
    ----------
        circuit:
            A circuit to vectorize.
    """
    elements = circuit.elements.values()
    element_values = [element.get_value(element_units=True) for element in elements]
    element_tensors = [torch.as_tensor(element_value)
                       for element_value in element_values]
    return torch.stack(element_tensors)


def eigencircuit(circuit: Circuit, num_eigen):
    """Solves for the eigenvalues of a circuit, returning them
    in PyTorch tensor format.

    Parameters
    ----------
        circuit:
            A circuit for which the eigensystem will be solved.
    """
    element_tensors = vectorize(circuit)
    element_tensors.requires_grad = True

    class EigenvalueSolver(torch.autograd.Function):
        @staticmethod
        def forward(ctx, element_tensors):
            elements = circuit.elements.values()
            values_units = [(element_tensors[idx].numpy(), element_tensors[idx].unit)
                            for idx in range(len(element_tensors))]
            circuit.update_elements(elements, values_units=values_units)
            H = circuit.hamiltonian()
            eigenvalues, _ = H.diag(num_eigen = num_eigen)
            eigenvalue_tensors = [torch.as_tensor(eigenvalue) for eigenvalue in eigenvalues]
            return torch.stack(eigenvalue_tensors)

        @staticmethod
        def backward(ctx, grad_output):
            cr_elements = circuit.elements.values()
            m, n = element_tensors.shape[0], num_eigen
            partial_omega = torch.FloatTensor((m, n))
            for el_idx in range(m):
                for eigen_idx in range(n):
                    partial_omega[el_idx, eigen_idx] = Circuit.get_partial_omega(el = cr_elements[el_idx],
                                                              m = eigen_idx)

            return partial_omega @ grad_output

    return element_tensors, EigenvalueSolver(element_tensors)
