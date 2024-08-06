"""Utility module with functions implemented in both PyTorch and numpy,
depending on optimization mode."""
from typing import List, Union

import numpy as np
from numpy import ndarray
import qutip as qt
from qutip import Qobj
import scipy
import torch
from torch import Tensor

from SQcircuit.settings import get_optim_mode


def remove_zero_columns(matrix):
    """Removes columns from the matrix that contain only zero values."""

    # Identify columns that are not all zeros
    non_zero_columns = np.any(matrix != 0, axis=0)

    # Filter the matrix to keep only non-zero columns
    filtered_matrix = matrix[:, non_zero_columns]

    return filtered_matrix


def remove_dependent_columns(matrix, tol=1e-5):
    """Remove dependent columns from the given matrix."""

    # first remove the zero columns from the matrix
    no_zero_col_matrix = remove_zero_columns(matrix)

    # Perform SVD
    _, s, _ = np.linalg.svd(no_zero_col_matrix)

    # Identify columns with singular values above tolerance
    independent_columns = np.where(np.abs(s) > tol)[0]

    # Reconstruct matrix with only independent columns
    reduced_matrix = no_zero_col_matrix[:, independent_columns]

    return reduced_matrix


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

def zero(dtype=torch.complex128):
    if get_optim_mode():
        return torch.tensor(0, dtype=dtype)
    return 0


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
    def tensor_to_numpy(tensor):
        if not list(tensor.size()):
            # Convert single-value tensors directly to numpy scalar
            return tensor.item()
        else:
            return tensor.detach().numpy()

    if get_optim_mode():
        if isinstance(input, list):
            return [tensor_to_numpy(value) for value in input]
        else:
            return tensor_to_numpy(input)
    else:
        if isinstance(input, qt.Qobj):
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
        if isinstance(A, Tensor) and isinstance(B, Tensor):
            if A.is_sparse and B.is_sparse:
                return torch.sparse.mm(A, B)
            if (A.is_sparse and not B.is_sparse) or (not A.is_sparse and B.is_sparse):
                return torch.sparse.mm(A, B)
            return torch.mm(A, B)
        elif isinstance(A, Tensor) and isinstance(B, Qobj):
            B = qobj_to_tensor(B)
            # TODO: Add additional check on input dimensions, to make sure this works
            return torch.transpose(torch.sparse.mm(B, A), 0, 1)
        elif isinstance(A, Qobj) and isinstance(B, Tensor):
            A = qobj_to_tensor(A)
            return torch.sparse.mm(A, B)
        A = dense(A)
        B = dense(B)
        return torch.matmul(torch.as_tensor(A, dtype=torch.complex128),
                            torch.as_tensor(B, dtype=torch.complex128))
    if isinstance(A, Qobj) and isinstance(B, Qobj):
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
    coo_tensor = torch.sparse_coo_tensor(
        indices_tensor, values_tensor, torch.Size(shape), dtype=dtype
    )

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
                coo_sparse = scipy.sparse.coo_matrix(
                    (values, (row_indices, col_indices)), shape=shape
                )
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


def real(x):
    if isinstance(x, Tensor):
        return torch.real(x)
    else:
        return np.real(x)
 

def imag(x):
    if isinstance(x, Tensor):
        return torch.imag(x)
    else:
        return np.imag(x)


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
