import torch
from torch.autograd import gradcheck

import SQcircuit.functions as sqf


def test_k0e():
    x = torch.rand(20, requires_grad=True, dtype=torch.double)
    assert gradcheck(sqf.k0e, x, eps=1e-6, atol=1e-4)

def test_log_k0():
    x = torch.rand(20, requires_grad=True, dtype=torch.double)
    assert gradcheck(sqf.log_k0, x, eps=1e-6, atol=1e-4)
