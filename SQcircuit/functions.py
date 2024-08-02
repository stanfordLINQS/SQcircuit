"""Utility module with functions implemented in both PyTorch and numpy,
depending on optimization mode."""
from typing import List, Union

import numpy as np
from numpy import ndarray
import qutip as qt
from qutip import Qobj
import scipy
import scipy.special as sp
import torch
from torch.autograd.function import once_differentiable
from torch import Tensor

from SQcircuit.settings import get_optim_mode

###############################################################################
# Mathematical functions
###############################################################################

def abs(x):
    if get_optim_mode():
        return torch.abs(x)
    return np.abs(x)


def cosh(x):
    if get_optim_mode():
        return torch.cosh(x)
    return np.cosh(x)


def exp(x):
    if get_optim_mode():
        return torch.exp(x)
    return np.exp(x)


def imag(x):
    if isinstance(x, Tensor):
        return torch.imag(x)
    else:
        return np.imag(x)


def log(x):
    if get_optim_mode():
        return torch.log(x)
    return np.log(x)


def log_sinh(x):
    if get_optim_mode():
        return x + torch.log(1 + torch.exp(-2 * x)) - torch.log(torch.tensor(2))
    else:
        return x + np.log(1 + np.exp(-2 * x)) - np.log(2)


def maximum(a, b):
    if get_optim_mode():
        if a > b:
            return 0 * b + a
        else:
            return b
        # return torch.maximum(a, b)
    return np.maximum(a, b)


def minimum(a, b):
    if get_optim_mode():
        if a > b:
            return 0 * a + b
        else:
            return a
        #return torch.minimum(a, b)
    return np.minimum(a, b)


def normal(mean, var):
    if get_optim_mode():
        return torch.normal(mean, var)
    return np.random.normal(mean, var)


def pow(x, a):
    if get_optim_mode():
        return torch.pow(x, a)
    return x ** a


def real(x):
    if isinstance(x, Tensor):
        return torch.real(x)
    else:
        return np.real(x)


def round(x, a=3):
    if get_optim_mode():
        return torch.round(x * 10**a) / (10**a)
    return np.round(x, a)


def sqrt(x):
    if get_optim_mode():
        return torch.sqrt(x)
    return np.sqrt(x)


def tanh(x):
    if get_optim_mode():
        return torch.tanh(x)
    return np.tanh(x)


def sum(a):
    if get_optim_mode():
        return torch.sum(a)
    return np.sum(a)


def sinh(x):
    if get_optim_mode():
        return torch.sinh(x)
    return np.sinh(x)


###############################################################################
# Special functions
###############################################################################


def kn(n, x):
    if get_optim_mode():
        return KnSolver.apply(n, x)

    return sp.kn(n, x)


def log_k0(x):
    if get_optim_mode():
        return LogK0Solver.apply(x)

    return LogK0Solver.log_kv(0, x)


class KnSolver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, n: int, x):
        ctx.save_for_backward(x)
        ctx.order = n
        x = numpy(x)
        return torch.as_tensor(sp.kn(n, x))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        z, = ctx.saved_tensors
        return None, grad_output * sp.kvp(ctx.order, z)


class LogK0Solver(torch.autograd.Function):
    """Computes the logarithm of the modified Bessel function of the second
    kind for n = 0.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = numpy(x)
        return torch.as_tensor(LogK0Solver.log_kv(0, x))

    @staticmethod
    def log_kv(n, x):
        return np.log(sp.kve(n, x)) - x

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        z, = ctx.saved_tensors
        z = numpy(z)
        # K0'(z) = -K1(z); see Eq. 10.29.3 of DLMF
        return grad_output * -1 * torch.exp(
            torch.tensor(LogK0Solver.log_kv(1, z) - LogK0Solver.log_kv(0, z))
        )


###############################################################################
# Creation routines
###############################################################################

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


def init_op(size):
    if get_optim_mode():
        return torch.zeros(size=size, dtype=torch.complex128)
        # return torch.sparse_coo_tensor(size = size, dtype=torch.complex128)
    return qt.Qobj()


def init_sparse(shape):
    if get_optim_mode():
        return torch.sparse_coo_tensor(size=shape, dtype=torch.complex128)
    return qt.Qobj()


def zero(dtype=torch.complex128):
    if get_optim_mode():
        return torch.tensor(0, dtype=dtype)
    return 0


def zeros(shape, dtype=torch.complex128):
    if get_optim_mode():
        return torch.zeros(shape, dtype=dtype)
    return np.zeros(shape)


###############################################################################
# Matrix operations
###############################################################################

def block_diag(*args: Union[ndarray, Tensor]) -> Union[ndarray, Tensor]:
    if get_optim_mode():
        return torch.block_diag(*args)

    return scipy.linalg.block_diag(*args)


def dag(state):
    assert len(state.shape) <= 2
    if get_optim_mode():
        if isinstance(state, np.ndarray) and state.ndim == 1:
            return np.conj(state)
        if isinstance(state, torch.Tensor) and state.dim() == 1:
            return torch.conj(state)
        return torch.conj(torch.transpose(state, 0, 1))
    return state.dag()


def diag(v):
    if get_optim_mode():
        return torch.diag(v)

    return np.diag(v)


def tensor_product(*args: Union[Qobj, Tensor]) -> Union[Qobj, Tensor]:
    if get_optim_mode():
        return tensor_product_torch(*args)

    return qt.tensor(*args)


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


def mat_inv(A):
    if get_optim_mode():
        return torch.linalg.inv(A)
    return np.linalg.inv(A)



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


def mul(S, T):
    if get_optim_mode():
        if isinstance(S, qt.Qobj):
            S = qobj_to_tensor(S)
        if isinstance(T, qt.Qobj):
            T = qobj_to_tensor(T)
        return torch.matmul(S, T)
    return S * T


def operator_inner_product(state1, op, state2):
    if get_optim_mode():
        state1 = torch.unsqueeze(state1, 1)
        state2 = torch.unsqueeze(state2, 1)

    # torch.sparse.mm requires sparse in first arg, dense in second
    A = mat_mul(op, state2)
    B = mat_mul(dag(state1), A)
    return unwrap(B)


def sort(a):
    if get_optim_mode():
        return torch.sort(a)
    return np.sort(a)


###############################################################################
# Type-casting
###############################################################################

def array(object):
    if get_optim_mode():
        return torch.as_tensor(object)
    return np.array(object)


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


def copy(x):
    if get_optim_mode():
        if isinstance(x, qt.Qobj):
            return x.copy()
        return x.clone()
    return x.copy()


def dense(obj):
    if isinstance(obj, qt.Qobj):
        return obj.data.todense()
    if isinstance(obj, torch.Tensor) and obj.layout != torch.strided:
        return obj.to_dense()
    return obj


def mat_to_op(A: Union[ndarray, Tensor]):
    """Change matrix format to ``qutip.Qobj`` operator."""

    if get_optim_mode():
        return A

    return qt.Qobj(A)


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


def unwrap(x):
    if get_optim_mode():
        return x[0, 0]
    return x.data[0, 0]
