"""
test_elements contains the test cases for the SQcircuit elements
functionalities.
"""

from copy import copy

import numpy as np
import torch

from SQcircuit.elements import Capacitor, Junction, Inductor, Loop
from SQcircuit.settings import set_optim_mode
from SQcircuit.circuit import Circuit, unt
import SQcircuit.functions as sqf
from SQcircuit.tests.conftest import (
    create_fluxonium_numpy,
    create_fluxonium_torch,
    create_transmon_numpy,
    create_transmon_torch,
    create_flux_transmon_numpy,
    create_flux_transmon_torch,
    create_fluxonium_torch_flux,
    create_JJL_numpy,
    create_JJL_torch,
)


trunc_num = 120
eigen_count = 20
tolerance = 2e-2

all_units = unt.farad_list | unt.freq_list | unt.henry_list


def max_ratio(a, b):
    return np.max([np.abs(b / a), np.abs(a / b)])

def function_grad_test(circuit_numpy: Circuit,
                       function_numpy,
                       circuit_torch: Circuit,
                       function_torch,
                       num_eigenvalues=20,
                       delta=1e-4):
    """General test function for comparing linear approximation with 
    gradient computed with PyTorch backpropagation.

    Parameters
    ----------
        circuit_numpy:
            Numpy circuit for which linear approximation will be calculated.
        function_numpy:
            Function to call on the numpy circuit. This should match the 
            expected output of `function_torch`.
        circuit_torch:
            Equivalent circuit to `circuit_numpy`, but constructed in PyTorch.
        function_torch:
            Equivalent function to `function_numpy`, but written in PyTorch.
        delta:
            Perturbation dx to each parameter value in `circuit_numpy` to
            compute linear gradient df/dx~(f(x+dx)-f(x)/dx).
    """
    set_optim_mode(False)
    circuit_numpy.diag(num_eigenvalues)

    set_optim_mode(True)
    circuit_torch.diag(num_eigenvalues)
    tensor_val = function_torch(circuit_torch)
    tensor_val.backward()

    for edge_idx, elements_by_edge in enumerate(circuit_numpy.elements.values()):
        for element_idx, element_numpy in enumerate(elements_by_edge):
            set_optim_mode(False)
            scale_factor = (1 / (2 * np.pi) if isinstance(element_numpy, Junction) else 1)

            # Calculate f(x+delta)
            elem_value = element_numpy.get_value(u=element_numpy.value_unit)
            element_numpy.set_value(
                scale_factor * elem_value * (1 + delta),
                element_numpy.value_unit
            )
            circuit_numpy.update()
            circuit_numpy.diag(num_eigenvalues)
            val_plus = function_numpy(circuit_numpy)

            # Calculate f(x-delta)
            element_numpy.set_value(
                scale_factor * elem_value * (1 - delta),
                element_numpy.value_unit
            )
            circuit_numpy.update()
            circuit_numpy.diag(num_eigenvalues)
            val_minus = function_numpy(circuit_numpy)

            # Calculate gradient
            grad_numpy = (val_plus - val_minus) / (2 * delta * elem_value * scale_factor * all_units[element_numpy.value_unit])

            # Reset circuit
            element_numpy.set_value(
                scale_factor * elem_value,
                element_numpy.value_unit
            )

            edge_elements_torch = list(circuit_torch.elements.values())[0]
            for edge_element in edge_elements_torch:
                print(f"edge element: {edge_element}")
                print(f"value: {edge_element._value}")
                print(f"value grad: {edge_element._value.grad}")
            grad_torch = edge_elements_torch[element_idx].internal_value.grad.detach().numpy()
            # TODO: Modify Element class so that following gradient scaling is not necessary

            if isinstance(element_numpy, Capacitor) and element_numpy.value_unit in unt.freq_list:
                grad_factor = -unt.e**2/2/element_numpy._value**2/(2*np.pi*unt.hbar)
                grad_torch /= grad_factor
            elif isinstance(element_numpy, Inductor) and element_numpy.value_unit in unt.freq_list:
                grad_factor = -(unt.Phi0/2/np.pi)**2/element_numpy._value**2/(2*np.pi*unt.hbar)
                grad_torch /= grad_factor
            if isinstance(element_numpy, Junction):
                grad_torch *= (2 * np.pi)
            print(f"grad torch: {grad_torch}, grad numpy: {grad_numpy}")

            assert np.sign(grad_torch) == np.sign(grad_numpy)
            assert max_ratio(grad_torch, grad_numpy) <= 1 + tolerance
    for loop_idx, loop in enumerate(circuit_numpy.loops):
        set_optim_mode(False)

        loop_flux = loop.internal_value / 2 / np.pi
        # assert 0 + delta < loop_flux < 1 - delta

        # Calculate f(x+delta)
        loop.set_flux(loop_flux + delta)
        circuit_numpy.diag(num_eigenvalues)
        val_plus = function_numpy(circuit_numpy)

        # Calculate f(x-delta)
        loop.set_flux(loop_flux - delta)
        circuit_numpy.diag(num_eigenvalues)
        val_minus = function_numpy(circuit_numpy)

        # Calculate gradient, being careful for flux insensitivity
        if val_plus == val_minus:
            grad_numpy = 0
        else:
            grad_numpy = (val_plus - val_minus) / (2 * delta * 2 * np.pi)

        # Reset circuit
        loop.set_flux(loop_flux)

        grad_torch = circuit_torch.loops[loop_idx].internal_value.grad
        if grad_torch is not None:
            print(f"loop #: {loop_idx}")
            print(f"grad torch: {grad_torch}, grad numpy: {grad_numpy}")
            grad_torch = grad_torch.detach().numpy()
            assert np.sign(grad_torch) == np.sign(grad_numpy)
            assert max_ratio(grad_torch, grad_numpy) <= 1 + tolerance

    set_optim_mode(True)
    circuit_torch.zero_parameters_grad()

def first_eigendifference_numpy(circuit):
    return circuit._efreqs[1] - circuit._efreqs[0]


def first_eigendifference_torch(circuit):
    eigenvals, _ = circuit.diag(eigen_count)
    return (eigenvals[1] - eigenvals[0]) * 2 * np.pi * 1e9


def test_omega_transmon():
    """Verify gradient of first eigendifference omega_1-omega_0 
    in transmon circuit with linearized value."""

    # Create circuits
    circuit_numpy = create_transmon_numpy(trunc_num)
    circuit_torch = create_transmon_torch(trunc_num)

    function_grad_test(circuit_numpy,
                       first_eigendifference_numpy,
                       circuit_torch,
                       first_eigendifference_torch)
    set_optim_mode(False)

def test_omega_flux_transmon():
    """Verify gradient of first eigendifference omega_1-omega_0 
    in transmon circuit with linearized value."""

    # Create circuits

    for flux_point in [1+1e-2, 0.3, 0.5 - 1e-3, 0.5+1e-3, 0.7, 1-1e-2]:
        print('flux point:', flux_point)
        circuit_numpy = create_flux_transmon_numpy(trunc_num, flux_point)
        circuit_torch = create_flux_transmon_torch(trunc_num, flux_point)

        function_grad_test(circuit_numpy,
                           first_eigendifference_numpy,
                           circuit_torch,
                           first_eigendifference_torch)
    set_optim_mode(False)

def test_omega_flux_fluxonium():
    """Verify gradient of first eigendifference omega_1-omega_0 
    in transmon circuit with linearized value."""

    # Create circuits

    for flux_point in [1+1e-2, 0.3, 0.5 - 1e-3, 0.5+1e-3, 0.7, 1-1e-2]:
        print('flux point:', flux_point)
        circuit_numpy = create_fluxonium_numpy(trunc_num, flux_point)
        circuit_torch = create_fluxonium_torch_flux(trunc_num, flux_point)

        function_grad_test(circuit_numpy,
                           first_eigendifference_numpy,
                           circuit_torch,
                           first_eigendifference_torch)
    set_optim_mode(False)


def T1_inv(circuit):
    return (
        circuit.dec_rate('capacitive', (0, 1)) 
        + circuit.dec_rate('inductive', (0, 1))
        +  circuit.dec_rate('quasiparticle', (0, 1))
    )

def test_T1_transmon():
    """Compare gradient of T1 decoherence due to capacitive, inductive, and 
    quasiparticle noise in transmon circuit with linearized value."""

    # Create circuits
    circuit_numpy = create_transmon_numpy(trunc_num)
    circuit_torch = create_transmon_torch(trunc_num)

    function_grad_test(circuit_numpy,
                       T1_inv,
                       circuit_torch,
                       T1_inv,
                       delta=1e-6)
    set_optim_mode(False)

def test_T1_transmon_flux():
    """Compare gradient of T1 decoherence due to capacitive, inductive, and 
    quasiparticle noise in transmon circuit with linearized value."""

    # Create circuits
    for flux_point in [1+1e-2, 0.3, 0.5 - 1e-3, 0.5+1e-3, 0.7, 1-1e-2]:
        print('flux point:', flux_point)
        circuit_numpy = create_flux_transmon_numpy(trunc_num, flux_point)
        circuit_torch = create_flux_transmon_torch(trunc_num, flux_point)

        function_grad_test(circuit_numpy,
                        T1_inv,
                        circuit_torch,
                        T1_inv,
                        delta=1e-6)
    set_optim_mode(False)


def test_grad_multiple_steps():
    """Sample ability of PyTorch to successfully update circuit parameters and 
    iteratively decrease loss for simple LC resonator frequency optimization 
    task."""
    set_optim_mode(True)
    cap_unit, ind_unit = 'pF', 'uH'

    # Test C differentiation
    C = Capacitor(7.746, cap_unit, Q=1e6, requires_grad=True)
    L = Inductor(81.67, ind_unit)
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
        eigenvalues, _ = cr.diag(2)
        omega = (eigenvalues[1] - eigenvalues[0])
        loss = (omega - omega_target) ** 2 / omega_target ** 2
        loss.backward()
        C._value.grad *= (C._value**2)
        optimizer.step()
        cr.update()
    assert loss <= 6e-3

    # Test L differentiation
    C = Capacitor(7.746, cap_unit, Q=1e6)
    L = Inductor(81.67, ind_unit, requires_grad=True)
    elements = {
        (0, 1): [C, L],
    }
    cr = Circuit(elements)
    cr.set_trunc_nums([10, ])
    eigenvalues, _ = cr.diag(2)
    optimizer = torch.optim.SGD(cr.parameters, lr=1)
    omega_target = 20e6 / 1e9  # convert to GHz
    N = 10
    for idx in range(N):
        print(
            f"Parameter values (C [pF] and L [uH]): {C.get_value().detach().numpy(), L.get_value().detach().numpy()}\n")
        optimizer.zero_grad()
        eigenvalues, _ = cr.diag(2)
        omega = (eigenvalues[1] - eigenvalues[0])
        loss = (omega - omega_target) ** 2 / omega_target ** 2
        loss.backward()
        L._value.grad *= (L._value) ** 2
        optimizer.step()
        cr.update()
    assert loss <= 6e-3
    set_optim_mode(False)


def test_grad_fluxonium():
    """Verify gradient values on more complex circuit, first resonant 
    eigendifference in fluxonium. As opposed to previous test with transmon 
    circuit, note that this also involves linear inductor and loop."""
    set_optim_mode(True)
    loop1 = Loop()
    C = Capacitor(3.6, 'GHz', Q=1e6, requires_grad=True)
    L = Inductor(0.46, 'GHz', Q=500e6, loops=[loop1], requires_grad=True)
    JJ = Junction(10.2, 'GHz', cap=C, A=1e-7, x=3e-06, loops=[loop1], requires_grad=True)

    # define the circuit
    elements = {
        (0, 1): [C, L, JJ]
    }
    cr = Circuit(elements, flux_dist='all')
    cr.set_trunc_nums([trunc_num])
    loop1.set_flux(0)
    eigenvalues, _ = cr.diag(2)
    optimizer = torch.optim.SGD(cr.parameters, lr=1e-1)
    omega_target = 2e9 / 1e9  # convert to GHz (initial value: ~8.2 GHz)
    N = 10
    for idx in range(N):
        C_value = C.get_value().detach().numpy()
        L_value = L.get_value().detach().numpy()
        JJ_value = JJ.get_value().detach().numpy()
        print(f"Parameter values (C [F], L [H], JJ [Hz]): {C_value, L_value, JJ_value}\n")
        optimizer.zero_grad()
        eigenvalues, _ = cr.diag(2)
        omega = (eigenvalues[1] - eigenvalues[0])
        loss = (omega - omega_target) ** 2 / omega_target ** 2
        loss.backward()
        C._value.grad *= (C._value) ** 2
        L._value.grad *= (L._value) ** 2
        JJ._value.grad *= (JJ._value) ** 2
        optimizer.step()
        cr.update()
    assert loss <= 5e-3
    set_optim_mode(False)


def test_spectrum_fluxonium():
    """Verify gradient of first eigendifference omega_1-omega_0 in fluxonium 
    circuit with linearized value."""

    # Create circuits
    circuit_numpy = create_fluxonium_numpy(trunc_num)
    circuit_torch = create_fluxonium_torch(trunc_num)

    function_grad_test(circuit_numpy,
                       first_eigendifference_numpy,
                       circuit_torch,
                       first_eigendifference_torch)
    set_optim_mode(False)


def test_T1_fluxonium():
    """Verify gradient of fluxonium for T1 noise sources, including capacitive,
    inductive, and quasiparticle decoherence."""

    # Create circuits
    circuit_numpy = create_fluxonium_numpy(trunc_num)
    circuit_torch = create_fluxonium_torch(trunc_num)

    def T1_inv_capacitive(circuit):
        return circuit.dec_rate('capacitive', (0, 1))

    def T1_inv_inductive(circuit):
        return circuit.dec_rate('inductive', (0, 1))

    def T1_inv_quasiparticle(circuit):
        return circuit.dec_rate('quasiparticle', (0, 1))

    function_grad_test(circuit_numpy,
                       T1_inv_capacitive,
                       circuit_torch,
                       T1_inv_capacitive,
                       delta=1e-6)
    print('inductive')
    # Test inductive T1 decoherence
    function_grad_test(circuit_numpy,
                       T1_inv_inductive,
                       circuit_torch,
                       T1_inv_inductive,
                       delta=1e-6)
    print('quasiparticle')
    # Test quasiparticle T1 decoherence
    function_grad_test(circuit_numpy,
                       T1_inv_quasiparticle,
                       circuit_torch,
                       T1_inv_quasiparticle,
                       delta=1e-9)
    set_optim_mode(False)


def flux_sensitivity_function(sensitivity_function,
                              flux_point=0.4,
                              delta=0.1):
    def flux_sensitivity(circuit):
        S = sqf.cast(0)
        for loop_idx, loop in enumerate(circuit.loops):
            first_harmonic_values = []
            for offset in (-delta, 0, delta):
                new_circuit = copy(circuit)
                new_loop = Loop()
                new_loop.set_flux(flux_point + offset)
                new_circuit.loops[loop_idx] = new_loop
                _ = new_circuit.diag(2)
                first_harmonic = sensitivity_function(new_circuit)
                first_harmonic_values.append(first_harmonic)
            (f_minus, f_0, f_plus) = first_harmonic_values
            S = ((f_0 - f_minus) / delta) ** 2 + ((f_plus - f_0) / delta) ** 2
            # Normalize loss function
            S /= sqf.abs(f_0 / delta) ** 2
        return S
    return flux_sensitivity


def test_flux_sensitivity():

    # Create circuits
    circuit_numpy = create_fluxonium_numpy(trunc_num)
    circuit_torch = create_fluxonium_torch(trunc_num)

    function_grad_test(circuit_numpy,
                       flux_sensitivity_function(first_eigendifference_numpy),
                       circuit_torch,
                       flux_sensitivity_function(first_eigendifference_torch),
                       delta=1e-4)


def test_anharmonicity():
    def anharmonicity_numpy(circuit):
        B = circuit._efreqs[2] - circuit._efreqs[1]
        A = circuit._efreqs[1] - circuit._efreqs[0]
        return B / A

    def anharmonicity_torch(circuit):
        eigenvals, _ = circuit.diag(eigen_count)
        B = (eigenvals[2] - eigenvals[1]) * 2 * np.pi * 1e9
        A = (eigenvals[1] - eigenvals[0]) * 2 * np.pi * 1e9
        return B / A


    # Create circuits
    circuit_numpy = create_fluxonium_numpy(trunc_num)
    circuit_torch = create_fluxonium_torch(trunc_num)

    function_grad_test(circuit_numpy,
                       anharmonicity_numpy,
                       circuit_torch,
                       anharmonicity_torch,
                       delta=1e-4)

def test_T2_cc():
    flux_points = [1e-2, 0.25, 0.5 - 1e-2, 0.5 + 1e-2, 0.75]

    for phi_ext in flux_points:
        set_optim_mode(False)
        circuit_numpy = create_fluxonium_numpy(trunc_num, phi_ext)

        # Create torch circuit
        set_optim_mode(True)
        circuit_torch = create_fluxonium_torch(trunc_num, phi_ext)

        function_grad_test(
            circuit_numpy,
            lambda cr: cr.dec_rate('cc', states=(0, 1)),
            circuit_torch,
            lambda cr: cr.dec_rate('cc', states=(0, 1)),
            num_eigenvalues=50,
            delta=1e-6
        )

def test_T2_charge():
    charge_offsets = [1e-2, 0.2, 0.4, 0.5 - 1e-2, 0.5 + 1e-2, 0.6, 0.8, 1-1e-2]

    for ng in charge_offsets:
        circuit_numpy = create_transmon_numpy(trunc_num)
        circuit_numpy.set_charge_offset(1, ng)

        circuit_torch = create_transmon_torch(trunc_num)
        circuit_torch.set_charge_offset(1, ng)

        function_grad_test(
            circuit_numpy,
            lambda cr: cr.dec_rate('charge', states=(0, 1)),
            circuit_torch,
            lambda cr: cr.dec_rate('charge', states=(0, 1)),
            num_eigenvalues=50,
            delta=1e-6
        )

def test_T2_flux():
    flux_points = [1e-2, 0.25, 0.5 - 1e-2, 0.5 + 1e-2, 0.75]

    for phi_ext in flux_points:
        circuit_numpy = create_fluxonium_numpy(trunc_num, phi_ext)
        circuit_torch = create_fluxonium_torch(trunc_num, phi_ext)

        function_grad_test(
            circuit_numpy,
            lambda cr: cr.dec_rate('flux', states=(0, 1)),
            circuit_torch,
            lambda cr: cr.dec_rate('flux', states=(0, 1)),
            num_eigenvalues=50,
            delta=1e-6
        )

def test_T2_cc_phi_ext():
    flux_points = [1e-2, 0.25, 0.5 - 1e-2, 0.5 + 1e-2, 0.75]

    for phi_ext in flux_points:
        set_optim_mode(False)
        circuit_numpy = create_fluxonium_numpy(trunc_num, phi_ext)

        # Create torch circuit
        set_optim_mode(True)
        circuit_torch = create_fluxonium_torch_flux(trunc_num, phi_ext)

        function_grad_test(
            circuit_numpy,
            lambda cr: cr.dec_rate('cc', states=(0, 1)),
            circuit_torch,
            lambda cr: cr.dec_rate('cc', states=(0, 1)),
            num_eigenvalues=50,
            delta=1e-6
        )

def test_T2_charge_phi_ext():
    charge_offsets = [1e-2, 0.2, 0.4, 0.5 - 1e-2, 0.5 + 1e-2, 0.6, 0.8, 1-1e-2]
    phi_ext = 0.5-1e-3

    for ng in charge_offsets:
        circuit_numpy = create_flux_transmon_numpy(trunc_num, phi_ext)
        circuit_numpy.set_charge_offset(1, ng)

        circuit_torch = create_flux_transmon_torch(trunc_num, phi_ext)
        circuit_torch.set_charge_offset(1, ng)

        function_grad_test(
            circuit_numpy,
            lambda cr: cr.dec_rate('charge', states=(0, 1)),
            circuit_torch,
            lambda cr: cr.dec_rate('charge', states=(0, 1)),
            num_eigenvalues=50,
        )

def test_T2_flux_phi_ext():
    flux_points = [1e-2, 0.25, 0.5 - 1e-2, 0.5 + 1e-2, 0.75]

    for phi_ext in flux_points:
        circuit_numpy = create_fluxonium_numpy(trunc_num, phi_ext)
        circuit_torch = create_fluxonium_torch_flux(trunc_num, phi_ext)

        function_grad_test(
            circuit_numpy,
            lambda cr: cr.dec_rate('flux', states=(0, 1)),
            circuit_torch,
            lambda cr: cr.dec_rate('flux', states=(0, 1)),
            num_eigenvalues=50,
            delta=1e-6
        )

# def test_T2_flux_JJL():
#     flux_points = [0.5] #, 0.25, 0.5 - 1e-2, 0.5 + 1e-2, 0.75]

#     for phi_ext in flux_points:
#         print('phi_ext', phi_ext)
#         circuit_numpy = create_JJL_numpy(45, phi_ext)
#         circuit_torch = create_JJL_torch(45, phi_ext)

#         function_grad_test(
#             circuit_numpy,
#             first_eigendifference_numpy, #lambda cr: cr.dec_rate('flux', states=(0, 1)),
#             circuit_torch,
#             first_eigendifference_torch, #lambda cr: cr.dec_rate('flux', states=(0, 1)),
#             num_eigenvalues=50,
#             delta=1e-4
#         )
