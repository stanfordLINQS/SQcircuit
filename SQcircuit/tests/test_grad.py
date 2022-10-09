"""
test_elements contains the test cases for the SQcircuit elements
functionalities.
"""

from SQcircuit.elements import Capacitor, Junction
from SQcircuit.settings import set_optim_mode
from SQcircuit.circuit import Circuit, unt

import numpy as np

trunc_num = 60
eigen_count = 60
tolerance = 1e-3

all_units = unt.farad_list | unt.freq_list | unt.henry_list

def update_circuit(circuit):
    circuit._update_H()
    circuit.set_trunc_nums([trunc_num, ])
    circuit.diag(eigen_count)
    return circuit

def max_ratio(a, b):
    return np.max([np.abs(b / a), np.abs(a / b)])


def function_grad_test(circuit_numpy,
                       function_numpy,
                       circuit_torch,
                       function_torch, delta=1e-4):
    set_optim_mode(False)
    circuit_numpy = update_circuit(circuit_numpy)
    val_init = function_numpy(circuit_numpy)
    set_optim_mode(True)
    circuit_torch = update_circuit(circuit_torch)
    tensor_val = function_torch(circuit_torch)
    tensor_val.backward()
    for edge_idx, elements_by_edge in enumerate(circuit_numpy.elements.values()):
        for element_idx, element_numpy in enumerate(elements_by_edge):
            set_optim_mode(False)
            scale_factor = (1 / (2 * np.pi) if type(element_numpy) is Junction else 1)
            element_numpy.set_value(scale_factor * element_numpy.get_value() / all_units[element_numpy.unit] + delta, element_numpy.unit)
            circuit_numpy = update_circuit(circuit_numpy)
            val_delta = function_numpy(circuit_numpy)
            grad_numpy = (val_delta - val_init) / (delta * all_units[element_numpy.unit])
            element_numpy.set_value(element_numpy.get_value() / all_units[element_numpy.unit] - delta, element_numpy.unit)

            edge_elements_torch = list(circuit_torch.elements.values())[0]
            grad_torch = edge_elements_torch[element_idx]._value.grad.detach().numpy()
            if type(element_numpy) is Junction:
                grad_torch *= (2 * np.pi)
            print(f"grad torch: {grad_torch}")
            print(f"grad numpy: {grad_numpy}")
            assert max_ratio(grad_torch, grad_numpy) <= 1 + tolerance
    assert False

def test_omega():
    cap_value, ind_value, Q = 7.746, 12, 1e6
    cap_unit, ind_unit = 'pF', 'GHz'
    ## Create numpy circuit
    set_optim_mode(False)
    C_numpy = Capacitor(cap_value, cap_unit, Q=Q)
    J_numpy = Junction(ind_value, ind_unit)
    cr_transmon = Circuit({(0, 1): [C_numpy, J_numpy], })
    circuit_numpy = update_circuit(cr_transmon)

    ## Create torch circuit
    set_optim_mode(True)
    C_torch = Capacitor(cap_value, cap_unit, Q=Q, requires_grad=True)
    J_torch = Junction(ind_value, ind_unit, requires_grad=True)
    circuit_torch = Circuit({(0, 1): [C_torch, J_torch]})
    circuit_torch.set_trunc_nums([trunc_num, ])

    def first_eigendifference_numpy(circuit):
        return circuit._efreqs[1] - circuit._efreqs[0]
    def first_eigendifference_torch(circuit):
        eigenvals, _ = circuit.diag(eigen_count)
        return (eigenvals[1] - eigenvals[0]) * 2 * np.pi * 1e9
    function_grad_test(circuit_numpy,
                       first_eigendifference_numpy,
                       circuit_torch,
                       first_eigendifference_torch)
    set_optim_mode(False)

def test_T1():
    cap_value, ind_value, Q = 7.746, 12, 1e6
    cap_unit, ind_unit = 'mF', 'GHz'
    ## Create numpy circuit
    set_optim_mode(False)
    C_numpy = Capacitor(cap_value, cap_unit, Q=Q)
    J_numpy = Junction(ind_value, ind_unit)
    cr_transmon = Circuit({(0, 1): [C_numpy, J_numpy], })
    circuit_numpy = update_circuit(cr_transmon)

    ## Create torch circuit
    set_optim_mode(True)
    C_torch = Capacitor(cap_value, cap_unit, Q=Q, requires_grad=True)
    J_torch = Junction(ind_value, ind_unit, requires_grad=True)
    circuit_torch = Circuit({(0, 1): [C_torch, J_torch]})
    circuit_torch.set_trunc_nums([trunc_num, ])

    def T1_inv(circuit):
        return circuit.dec_rate('capacitive', (0, 1))

    function_grad_test(circuit_numpy,
                       T1_inv,
                       circuit_torch,
                       T1_inv,
                       delta=1e-4)
    set_optim_mode(False)