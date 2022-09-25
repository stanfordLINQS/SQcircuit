"""utils.py module with functions implemented in both PyTorch and numpy,
depending on optimization mode."""

import numpy as np
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