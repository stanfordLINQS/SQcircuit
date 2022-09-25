"""utils.py module with functions implemented in both PyTorch and numpy,
depending on optimization mode."""

import numpy as np
import qutip as qt
import torch

from SQcircuit.settings import OPTIM_MODE

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