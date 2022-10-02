"""utils.py module with functions implemented in both PyTorch and numpy,
depending on optimization mode."""


import numpy as np
import qutip as qt
import scipy
import torch


from SQcircuit.settings import get_optim_mode
import SQcircuit.units as unt


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
            eigenvalues, _ = circuit.diag_np(n_eig=num_eigen)
            eigenvalues = [eigenvalue * 2 * np.pi * unt.get_unit_freq() for eigenvalue in eigenvalues]
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
            _, eigenvectors = circuit.diag_np(n_eig=num_eigen)
            eigenvector_tensors = [torch.as_tensor(eigenvector.full()) for eigenvector in eigenvectors]
            return torch.squeeze(torch.stack(eigenvector_tensors))

        @staticmethod
        def backward(ctx, grad_output):
            cr_elements = list(circuit.elements.values())[0]
            m, n, l = tensor_list.shape[0], *grad_output.shape
            partial_eigenvec = torch.zeros([m, n, l], dtype=torch.complex128)
            for el_idx in range(m):
                for eigen_idx in range(n):
                    partial_tensor = torch.squeeze(
                        torch.as_tensor(circuit._get_partial_vec(el=cr_elements[el_idx], m=eigen_idx).full()))
                    partial_eigenvec[el_idx, eigen_idx, :] = partial_tensor * initial_element_vals[
                        el_idx]  # rescale gradient based on initial value
            return torch.real(torch.sum(partial_eigenvec * torch.conj(grad_output), axis=(-1, -2)) +
                              torch.sum(torch.conj(partial_eigenvec) * grad_output, axis=(-1, -2)))

    return tensor_list, EigenvalueSolver, EigenvectorSolver

# func_names = ['abs', 'tanh', 'exp', 'sqrt']
#
# func_constructor = """def {0}(x):
#     if flag:
#         return torch.{0}(x)
#     else:
#         return np.{0}(x)"""
#
# for func_name in func_names:
#     exec(func_constructor.format(func_name))


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
        return torch.sparse_coo_tensor(size=shape, dtype=torch.complex128)
    return qt.Qobj()


def init_op(size):
    if get_optim_mode():
        return torch.zeros(size = size, dtype = torch.complex128)
        # return torch.sparse_coo_tensor(size = size, dtype=torch.complex128)
    return qt.Qobj()


def zeros(shape):
    if get_optim_mode():
        return torch.zeros(shape, dtype=torch.complex128)
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
            return qobj_to_tensor(value)
        return torch.tensor(
            value,
            requires_grad=True,
            dtype=torch.complex128
        )
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


def dag(state):
    if get_optim_mode():
        if isinstance(state, np.ndarray) and state.ndim == 1:
            return np.conj(state)
        if isinstance(state, torch.Tensor) and state.dim() == 1:
            return torch.conj(state)
        return torch.conj(torch.transpose(state))
    return state.dag()


def copy(x):
    if get_optim_mode():
        if isinstance(x, qt.Qobj):
            return x.copy()
        return x.clone()
    return x.copy()


def unwrap(x):
    if get_optim_mode():
        return x
    return x.data[0, 0]


def dense(obj):
    if isinstance(obj, qt.Qobj):
        return obj.data.todense()
    if isinstance(obj, torch.Tensor) and obj.layout != torch.strided:
        return obj.to_dense()
    return obj


def mat_mul(A, B):
    if get_optim_mode():
        A = dense(A)
        B = dense(B)
        return torch.matmul(torch.as_tensor(A, dtype=torch.complex128), torch.as_tensor(B, dtype=torch.complex128))
    if isinstance(A, qt.Qobj) and isinstance(B, qt.Qobj):
        return A * B
    return A @ B


'''def sparse_csr_to_tensor(S):
    S = S.tocoo()
    values = S.data
    indices = np.vstack((S.row, S.col))

    i = torch.LongTensor(indices)
    v = torch.as_tensor(values, dtype=torch.complex128)
    shape = S.shape

    return torch.sparse.Tensor(i, v, torch.Size(shape), dtype=torch.complex128)'''

# Perhaps there is a way to do this without converting to dense
# Ex. https://github.com/pytorch/pytorch/pull/57125/commits
'''def sparse_csr_to_tensor(S):
    D = torch.as_tensor(S.todense(), dtype=torch.complex128)
    r = torch.real(D)
    i = torch.imag(D)
    return torch.complex(r, i).to_sparse()'''


# Currently, use dense form as PyTorch doesn't seem to support complex sparse tensors
def qobj_to_tensor(S):
    return torch.as_tensor(S.full(), dtype=torch.complex128)


def mul(S, T):
    if get_optim_mode():
        if isinstance(S, qt.Qobj):
            S = qobj_to_tensor(S)
        if isinstance(T, qt.Qobj):
            T = qobj_to_tensor(T)
        # return torch.sparse.mm(S, T)
        return torch.matmul(S, T)
    return S * T


def qutip(input):
    if get_optim_mode():
        if isinstance(input, torch.Tensor):
            return qt.Qobj(inpt=input.detach().numpy())
        return input
    return input