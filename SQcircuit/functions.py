"""Utility module with functions implemented in both PyTorch and numpy,
depending on optimization mode."""
from typing import Union

import numpy as np
from numpy import ndarray
import qutip as qt
from qutip import Qobj
from qutip.core.data import Dia, CSR, Dense
import scipy
from scipy.special import kve
import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from SQcircuit.settings import get_optim_mode

###############################################################################
# Mathematical functions
###############################################################################

def abs(x):
    if isinstance(x, Tensor):
        return torch.abs(x)
    return np.abs(x)


def cosh(x):
    if isinstance(x, Tensor):
        return torch.cosh(x)
    return np.cosh(x)


def exp(x):
    if isinstance(x, Tensor):
        return torch.exp(x)
    return np.exp(x)


def imag(x):
    if isinstance(x, Tensor):
        return torch.imag(x)
    return np.imag(x)


def log(x):
    if isinstance(x, Tensor):
        return torch.log(x)
    return np.log(x)


def log_sinh(x):
    if isinstance(x, Tensor):
        return x + torch.log(1 - torch.exp(-2 * x)) - torch.log(torch.tensor(2))
    else:
        return x + np.log(1 - np.exp(-2 * x)) - np.log(2)


def maximum(a, b):
    if isinstance(a, Tensor):
        if a > b:
            return 0 * b + a
        else:
            return b
    return np.maximum(a, b)


def minimum(a, b):
    if isinstance(a, Tensor):
        if a > b:
            return 0 * a + b
        else:
            return a
    return np.minimum(a, b)


def normal(mean, var):
    if isinstance(mean, Tensor):
        return torch.normal(mean, var)
    return np.random.normal(mean, var)


def pow(x, a):
    if isinstance(x, Tensor):
        return torch.pow(x, a)
    return x ** a


def real(x):
    if isinstance(x, Tensor):
        return torch.real(x)
    return np.real(x)


def round(x, a=3):
    if isinstance(x, Tensor):
        return torch.round(x * 10**a) / (10**a)
    return np.round(x, a)


def sqrt(x):
    if isinstance(x, Tensor):
        return torch.sqrt(x)
    return np.sqrt(x)


def tanh(x):
    if isinstance(x, Tensor):
        return torch.tanh(x)
    return np.tanh(x)


def sum(a):
    if isinstance(a, Tensor):
        return torch.sum(a)
    return np.sum(a)


def sinh(x):
    if isinstance(x, Tensor):
        return torch.sinh(x)
    return np.sinh(x)


###############################################################################
# Special functions
###############################################################################

def k0e(x):
    if isinstance(x, Tensor):
        return K0eSolver.apply(x)
    return kve(0, x)


def log_k0(x):
    if isinstance(x, Tensor):
        return LogK0Solver.apply(x)
    return log(k0e(x)) - x


class K0eSolver(torch.autograd.Function):
    """Computes ``torch.special.scaled_modified_bessel_k0`` with gradient.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.special.scaled_modified_bessel_k0(x)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        x = to_numpy(x)

        return grad_output * torch.tensor(kve(0, x) - kve(1, x))


class LogK0Solver(torch.autograd.Function):
    """Computes the logarithm of the modified Bessel function of the second
    kind for n = 0.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = to_numpy(x)
        return torch.as_tensor(LogK0Solver.log_kv(0, x))

    @staticmethod
    def log_kv(n, x):
        return np.log(kve(n, x)) - x

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        x = to_numpy(x)
        # K0'(z) = -K1(z); see Eq. 10.29.3 of DLMF
        return grad_output * -1 * torch.exp(
            torch.tensor(LogK0Solver.log_kv(1, x) - LogK0Solver.log_kv(0, x))
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


def zero(dtype=torch.complex128, requires_grad=False):
    if get_optim_mode():
        return torch.tensor(0, dtype=dtype, requires_grad=requires_grad)
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

    if isinstance(state, np.ndarray) and state.ndim == 1:
        return np.conj(state)
    elif isinstance(state, torch.Tensor):
        if state.dim() == 1:
            return torch.conj(state)
        else:
            return torch.conj(torch.transpose(state, 0, 1))
    elif isinstance(state, Qobj):
        return state.dag()
    else:
        raise ValueError(f'Object type {type(state)} is not supported.')


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
    if isinstance(A, Tensor):
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


def copy(x):
    if get_optim_mode():
        if isinstance(x, Qobj):
            return x.copy()
        return x.clone()
    return x.copy()


def dense(obj):
    if isinstance(obj, Qobj):
        return obj.full()
    if isinstance(obj, torch.Tensor) and obj.layout != torch.strided:
        return obj.to_dense()
    return obj


def mat_to_op(A: Union[ndarray, Tensor]):
    if isinstance(A, Tensor):
        return A

    return Qobj(A).to('csr')


def to_numpy(A: Union[ndarray, float, Tensor]):
    if isinstance(A, Tensor):
        return A.detach().numpy()
    elif isinstance(A, Qobj):
        return A.full()
    return A


def qobj_to_tensor(qobj, dtype=torch.complex128):
    """Convert ``qobj`` to tensor (sparse if ``qobj`` is not dense). """

    if qobj.dtype == CSR:
        # Convert to coo, as there does not seem to be a way to convert directly
        # from csr format to PyTorch sparse tensor
        coo_scipy = qobj.data_as('csr_matrix').tocoo()
    elif qobj.dtype == Dia:
        coo_scipy = qobj.data_as('dia_matrix').tocoo()
    elif qobj.dtype == Dense:
        return torch.as_tensor(qobj.full(), dtype=dtype)
    else:
        raise ValueError(f'The datatype {qobj.dtype} is not supported.') 

    # Now deal with COO
    indices = np.vstack((coo_scipy.row, coo_scipy.col))
    values = coo_scipy.data
    shape = coo_scipy.shape

    indices_tensor = torch.LongTensor(indices)
    values_tensor = torch.tensor(values, dtype=dtype)
    coo_tensor = torch.sparse_coo_tensor(
        indices_tensor, values_tensor, torch.Size(shape), dtype=dtype
    )

    return coo_tensor


def unwrap(x):
    if get_optim_mode():
        return x[0, 0]
    return x
