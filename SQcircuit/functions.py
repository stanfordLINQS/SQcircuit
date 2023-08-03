"""utils.py module with functions implemented in both PyTorch and numpy,
depending on optimization mode."""

from typing import List, Union

import scipy
import torch

import numpy as np
import qutip as qt

from torch import Tensor
from numpy import ndarray
from qutip import Qobj

import SQcircuit.units as unt

from SQcircuit.settings import get_optim_mode


def eigencircuit(circuit, n_eig: int):
    """Given a circuit, returns Torch functions that compute the concatenated 
    tensor including both eigenvalues and eigenvectors of a circuit.

    Parameters
    ----------
        circuit:
            A circuit for which the eigensystem will be solved.
        n_eig:
            Number of eigenvalues to output. The lower ``n_eig``, the
                faster ``SQcircuit`` finds the eigenvalues.
    """
    return EigenSolver.apply(torch.stack(circuit.parameters) if circuit.parameters else torch.tensor([]), 
                             circuit, 
                             n_eig)

class EigenSolver(torch.autograd.Function):

    @staticmethod
    def forward(ctx, element_tensors, circuit, n_eig):
        ctx.save_for_backward(element_tensors)
        ctx.circuit=circuit
        ctx.n_eig=n_eig
        # Compute forward pass for eigenvalues
        eigenvalues, eigenvectors = circuit.diag_np(n_eig=n_eig)
        eigenvalues = [eigenvalue * 2 * np.pi * unt.get_unit_freq() for
                    eigenvalue in eigenvalues]
        eigenvalue_tensors = [torch.as_tensor(eigenvalue) for
                              eigenvalue in eigenvalues]
        eigenvalue_tensor = torch.stack(eigenvalue_tensors)
        eigenvalue_tensor = torch.unsqueeze(eigenvalue_tensor, dim=-1)
        # Compute forward pass for eigenvectors
        eigenvector_tensors = [torch.as_tensor(eigenvector.full()) for
                               eigenvector in eigenvectors]
        eigenvector_tensor = torch.squeeze(torch.stack(eigenvector_tensors))
        return torch.cat([eigenvalue_tensor, eigenvector_tensor], dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        # Break grad_output into eigenvalue sub-tensor and eigenvector 
        # sub-tensor
        elements = list(ctx.circuit._parameters.keys())
        m, n, l = (
            len(ctx.circuit.parameters), 
            ctx.n_eig,
            (grad_output.shape[1] - 1),  # grad_output.shape[1] ==  n + 1
        ) 
        grad_output_eigenvalue = grad_output[:, 0]
        grad_output_eigenvector = grad_output[:, 1:]

        partial_omega = torch.zeros([m, n], dtype=float)
        partial_eigenvec = torch.zeros([m, n, l], dtype=torch.complex128)
        for el_idx in range(m):
            for eigen_idx in range(n):
                # Compute backward pass for eigenvalues
                partial_omega[
                    el_idx, eigen_idx] = ctx.circuit.get_partial_omega(
                    el=elements[el_idx],
                    m=eigen_idx, subtract_ground=False)
                # Compute backward pass for eigenvectors
                partial_tensor = torch.squeeze(
                    torch.as_tensor(
                        ctx.circuit.get_partial_vec(el=elements[el_idx],
                                                m=eigen_idx).full()))
                partial_eigenvec[el_idx, eigen_idx, :] = partial_tensor
        eigenvalue_grad = torch.sum(
            partial_omega * torch.conj(grad_output_eigenvalue), axis=-1)
        eigenvector_grad = torch.sum(
            partial_eigenvec * torch.conj(grad_output_eigenvector),
            axis=(-1, -2))

        input_tensor, = ctx.saved_tensors
        return torch.real(eigenvalue_grad + eigenvector_grad).view(input_tensor.shape), None, None


def get_kn_solver(n: int):
    class kn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            x = numpy(x)
            return torch.as_tensor(scipy.special.kn(n, x))

        @staticmethod
        def backward(ctx, grad_output):
            z, = ctx.saved_tensors
            return grad_output * scipy.special.kvp(n, z)

    return kn


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


def block_diag(*args: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
    if get_optim_mode():
        return torch.block_diag(*args)

    return scipy.linalg.block_diag(*args)


def tensor_product_torch(*args: Tensor) -> Tensor:
    """ Pytorch function similar to ``qutip.tensor``."""
    op_list = args

    out = torch.tensor([])

    for n, op in enumerate(op_list):
        if n == 0:
            out = op
        else:
            out = torch.kron(out, op)

    return out


def tensor_product(*args: Union[Qobj, Tensor]) -> Union[Qobj, Tensor]:
    if get_optim_mode():
        return tensor_product_torch(*args)

    return qt.tensor(*args)


def eye(N: int) -> Union[Qobj, Tensor]:
    """Return identity operator in qutip or torch.

    parameters
    -----------
        N:
            Size of the operator.
    """
    if get_optim_mode():
        return torch.eye(N)

    return qt.qeye(N)


def diag(v):
    if get_optim_mode():
        return torch.diag(v)

    return np.diag(v)


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
        return torch.zeros(size=size, dtype=torch.complex128)
        # return torch.sparse_coo_tensor(size = size, dtype=torch.complex128)
    return qt.Qobj()


def zeros(shape, dtype=torch.complex128):
    if get_optim_mode():
        return torch.zeros(shape, dtype=dtype)
    return np.zeros(shape)


def log(x):
    if get_optim_mode():
        return torch.log(x)
    return np.log(x)


def array(object):
    if get_optim_mode():
        return torch.as_tensor(object)
    return np.array(object)


def sum(a):
    if get_optim_mode():
        return torch.sum(a)
    return np.sum(a)


def sinh(x):
    if get_optim_mode():
        return torch.sinh(x)
    return np.sinh(x)


def cosh(x):
    if get_optim_mode():
        return torch.cosh(x)
    return np.cosh(x)


def tanh(x):
    if get_optim_mode():
        return torch.tanh(x)
    return np.tanh(x)


def sort(a):
    if get_optim_mode():
        return torch.sort(a)
    return np.sort(a)

# TODO: Fix this function name
def cast(value, dtype=torch.complex128, requires_grad = True):
    if get_optim_mode():
        if isinstance(value, qt.Qobj):
            return qobj_to_tensor(value)
        return torch.tensor(
            value,
            requires_grad=requires_grad,
            dtype=dtype
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
    else:
        if type(input) is qt.Qobj:
            return input.full()
    return input


def dag(state):
    assert len(state.shape) <= 2
    if get_optim_mode():
        if isinstance(state, np.ndarray) and state.ndim == 1:
            return np.conj(state)
        if isinstance(state, torch.Tensor) and state.dim() == 1:
            return torch.conj(state)
        return torch.conj(torch.transpose(state, 0, 1))
    return state.dag()


def copy(x):
    if get_optim_mode():
        if isinstance(x, qt.Qobj):
            return x.copy()
        return x.clone()
    return x.copy()


def unwrap(x):
    if get_optim_mode():
        return x[0, 0]
    return x.data[0, 0]


def dense(obj):
    if isinstance(obj, qt.Qobj):
        return obj.data.todense()
    if isinstance(obj, torch.Tensor) and obj.layout != torch.strided:
        return obj.to_dense()
    return obj


def mat_mul(A, B):
    if get_optim_mode():
        if type(A) is Tensor and type(B) is Tensor:
            if A.is_sparse and B.is_sparse:
                return torch.sparse.mm(A, B)
            if (A.is_sparse and not B.is_sparse) or (not A.is_sparse and B.is_sparse):
                return torch.sparse.mm(A, B)
            return torch.mm(A, B)
        elif type(A) is Tensor and type(B) is Qobj:
            B = qobj_to_tensor(B)
            # TODO: Add additional check on input dimensions, to make sure this works
            return torch.transpose(torch.sparse.mm(B, A), 0, 1)
        elif type(B) is Tensor and type(A) is Qobj:
            A = qobj_to_tensor(A)
            return torch.sparse.mm(A, B)
        A = dense(A)
        B = dense(B)
        return torch.matmul(torch.as_tensor(A, dtype=torch.complex128),
                            torch.as_tensor(B, dtype=torch.complex128))
    if isinstance(A, qt.Qobj) and isinstance(B, qt.Qobj):
        return A * B
    return A @ B


def qobj_to_tensor(qobj, dtype=torch.complex128):
    '''return torch.as_tensor(S.full(), dtype=torch.complex128)'''
    # Convert to coo, as there does not seem to be a way to convert directly
    # from csr format to PyTorch sparse tensor
    coo_scipy = qobj.data.tocoo()
    indices = np.vstack((coo_scipy.row, coo_scipy.col))
    values = coo_scipy.data
    shape = coo_scipy.shape

    indices_tensor = torch.LongTensor(indices)
    values_tensor = torch.tensor(values, dtype=dtype)
    coo_tensor = torch.sparse_coo_tensor(indices_tensor, values_tensor, torch.Size(shape),
                            dtype=dtype)

    return coo_tensor

def mul(S, T):
    if get_optim_mode():
        if isinstance(S, qt.Qobj):
            S = qobj_to_tensor(S)
        if isinstance(T, qt.Qobj):
            T = qobj_to_tensor(T)
        return torch.matmul(S, T)
    return S * T


def qutip(A: Union[Qobj, Tensor], dims=List[list]) -> Qobj:
    if get_optim_mode():
        if isinstance(A, torch.Tensor):
            if A.is_sparse:
                input = A.detach().coalesce()
                indices = input.indices().numpy()
                row_indices = indices[0, :]
                col_indices = indices[1, :]
                values = input.values().numpy()
                shape = tuple(input.shape)
                coo_sparse = scipy.sparse.coo_matrix((values, (row_indices, col_indices)), shape=shape)
                csr_sparse = coo_sparse.tocsr()
                qobj = qt.Qobj(inpt=csr_sparse)
                qobj.dims = dims
                return qobj

            qobj = qt.Qobj(inpt=A.detach().numpy())
            qobj.dims = dims
            return qobj
        return A
    return A


def mat_to_op(A: Union[ndarray, Tensor]):
    """Change matrix format to ``qutip.Qobj`` operator."""

    if get_optim_mode():
        return A

    return qt.Qobj(A)


def pow(x, a):
    if get_optim_mode():
        return torch.pow(x, a)
    return x ** a

def minimum(a, b):
    if get_optim_mode():
        if a > b:
            return 0 * a + b
        else:
            return a
        #return torch.minimum(a, b)
    return np.minimum(a, b)

def maximum(a, b):
    if get_optim_mode():
        if a > b:
            return 0 * b + a
        else:
            return b
        # return torch.maximum(a, b)
    return np.maximum(a, b)

def round(x, a=3):
    if get_optim_mode():
        return torch.round(x * 10**a) / (10**a)
    return np.round(x, a)

def operator_inner_product(state1, op, state2):
    if get_optim_mode():
        state1 = torch.unsqueeze(state1, 1)
        state2 = torch.unsqueeze(state2, 1)

    # torch.sparse.mm requires sparse in first arg, dense in second
    A = mat_mul(op, state2)
    B = mat_mul(dag(state1), A)
    return unwrap(B)

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