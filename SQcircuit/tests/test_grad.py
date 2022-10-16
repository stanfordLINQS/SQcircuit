"""
test_elements contains the test cases for the SQcircuit elements
functionalities.
"""

from SQcircuit.elements import Capacitor, Junction, Inductor, Loop
from SQcircuit.settings import set_optim_mode
from SQcircuit.circuit import Circuit, unt

import numpy as np
import torch

trunc_num = 120
eigen_count = 20
tolerance = 2e-2

all_units = unt.farad_list | unt.freq_list | unt.henry_list

def update_circuit(circuit):
    circuit._update_H()
    circuit.set_trunc_nums([trunc_num, ])
    return circuit

def max_ratio(a, b):
    return np.max([np.abs(b / a), np.abs(a / b)])


def function_grad_test(circuit_numpy,
                       function_numpy,
                       circuit_torch,
                       function_torch, delta=1e-4):
    set_optim_mode(False)
    circuit_numpy = update_circuit(circuit_numpy)
    circuit_numpy.diag(eigen_count)
    val_init = function_numpy(circuit_numpy)
    set_optim_mode(True)
    circuit_torch = update_circuit(circuit_torch)
    circuit_torch.diag(eigen_count)
    tensor_val = function_torch(circuit_torch)
    tensor_val.backward()
    for edge_idx, elements_by_edge in enumerate(circuit_numpy.elements.values()):
        for element_idx, element_numpy in enumerate(elements_by_edge):
            set_optim_mode(False)
            scale_factor = (1 / (2 * np.pi) if type(element_numpy) is Junction else 1)
            element_numpy.set_value(scale_factor * element_numpy.get_value() / all_units[element_numpy.unit] + delta, element_numpy.unit)
            circuit_numpy = update_circuit(circuit_numpy)
            circuit_numpy.diag(eigen_count)
            val_delta = function_numpy(circuit_numpy)
            grad_numpy = (val_delta - val_init) / (delta * all_units[element_numpy.unit])
            element_numpy.set_value(scale_factor * element_numpy.get_value() / all_units[element_numpy.unit] - delta, element_numpy.unit)

            edge_elements_torch = list(circuit_torch.elements.values())[0]
            grad_torch = edge_elements_torch[element_idx]._value.grad.detach().numpy()
            if type(element_numpy) is Junction:
                grad_torch *= (2 * np.pi)
            print(f"Element  value: {element_numpy.get_value()}")
            print(f"Function value (numpy): {function_numpy(circuit_numpy)}")
            print(f"partial_omega (numpy): {circuit_numpy._get_partial_omega_mn(element_numpy, (0, 1))}")
            print(f"grad torch: {grad_torch}")
            print(f"grad numpy: {grad_numpy}")
            print(f"error ratio: {max_ratio(grad_torch, grad_numpy)}")
            assert max_ratio(grad_torch, grad_numpy) <= 1 + tolerance

def test_omega():
    cap_value, ind_value, Q = 7.746, 12, 1e6
    cap_unit, ind_unit = 'pF', 'GHz'
    ## Create numpy circuit
    set_optim_mode(False)
    C_numpy = Capacitor(cap_value, cap_unit, Q=Q)
    J_numpy = Junction(ind_value, ind_unit)
    cr_transmon = Circuit({(0, 1): [C_numpy, J_numpy], })
    circuit_numpy = update_circuit(cr_transmon)
    circuit_numpy.diag(eigen_count)

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
    cap_value, ind_value, Q = 7.746, 5, 1e6
    cap_unit, ind_unit = 'fF', 'GHz'
    ## Create numpy circuit
    set_optim_mode(False)
    C_numpy = Capacitor(cap_value, cap_unit, Q=Q)
    J_numpy = Junction(ind_value, ind_unit)
    cr_transmon = Circuit({(0, 1): [C_numpy, J_numpy], })
    circuit_numpy = update_circuit(cr_transmon)
    circuit_numpy.diag(2)

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
                       delta=1e-6)
    set_optim_mode(False)

def test_grad_multiple_steps():
    set_optim_mode(True)
    cap_unit, ind_unit = 'pF', 'uH'

    ## Test C differentiation
    C = Capacitor(7.746, cap_unit, Q=1e6, requires_grad=True)
    L = Inductor(81.67, ind_unit)
    elements = {
        (0, 1): [C, L],
    }
    cr = Circuit(elements)
    cr.set_trunc_nums([10, ])
    eigenvalues, _ = cr.diag(2)
    optimizer = torch.optim.SGD(cr.parameters, lr=1)
    omega_target = 20e6 / (1e9) # convert to GHz
    N = 10
    for idx in range(N):
        print(
            f"Parameter values (C [pF] and L [uH]): {C.get_value().detach().numpy(), L.get_value().detach().numpy()}\n")
        optimizer.zero_grad()
        cr = update_circuit(cr)
        eigenvalues, _ = cr.diag(2)
        omega = (eigenvalues[1] - eigenvalues[0])
        loss = (omega - omega_target) ** 2 / omega_target ** 2
        loss.backward()
        C._value.grad *= (C._value)**2
        optimizer.step()
    assert loss <= 6e-3

    ## Test L differentiation
    C = Capacitor(7.746, cap_unit, Q=1e6)
    L = Inductor(81.67, ind_unit, requires_grad=True)
    elements = {
        (0, 1): [C, L],
    }
    cr = Circuit(elements)
    cr.set_trunc_nums([10, ])
    eigenvalues, _ = cr.diag(2)
    optimizer = torch.optim.SGD(cr.parameters, lr=1)
    omega_target = 20e6 / (1e9)  # convert to GHz
    N = 10
    for idx in range(N):
        print(
            f"Parameter values (C [pF] and L [uH]): {C.get_value().detach().numpy(), L.get_value().detach().numpy()}\n")
        optimizer.zero_grad()
        cr = update_circuit(cr)
        eigenvalues, _ = cr.diag(2)
        omega = (eigenvalues[1] - eigenvalues[0])
        loss = (omega - omega_target) ** 2 / omega_target ** 2
        loss.backward()
        L._value.grad *= (L._value) ** 2
        optimizer.step()
    assert loss <= 6e-3
    set_optim_mode(False)

def test_grad_fluxonium():
    set_optim_mode(True)
    loop1 = Loop()
    C = Capacitor(3.6, 'GHz', Q=1e6, requires_grad=True)
    L = Inductor(0.46, 'GHz', Q=500e6, loops=[loop1], requires_grad=True)
    JJ = Junction(10.2, 'GHz', cap=C, A=1e-7, x=3e-06, loops=[loop1], requires_grad=True)

    # define the circuit
    elements = {
        (0, 1): [L, JJ]
    }
    cr = Circuit(elements, flux_dist='all')
    cr.set_trunc_nums([60])
    loop1.set_flux(0)
    eigenvalues, _ = cr.diag(2)
    optimizer = torch.optim.SGD(cr.parameters, lr=1e-1)
    omega_target = 2e9 / (1e9)  # convert to GHz (initial value: ~8.2 GHz)
    N = 10
    for idx in range(N):
        print(
            f"Parameter values (C [F], L [H], JJ [Hz]): {C.get_value().detach().numpy(), L.get_value().detach().numpy() ,JJ.get_value().detach().numpy()}\n")
        optimizer.zero_grad()
        cr = update_circuit(cr)
        eigenvalues, _ = cr.diag(2)
        omega = (eigenvalues[1] - eigenvalues[0])
        loss = (omega - omega_target) ** 2 / omega_target ** 2
        loss.backward()
        C._value.grad *= (C._value) ** 2
        L._value.grad *= (L._value) ** 2
        JJ._value.grad *= (JJ._value) ** 2
        optimizer.step()
    assert loss <= 5e-3
    set_optim_mode(False)
