"""test_sampler contains tests for generating random circuit topologies with
circuit values sampled from a fixed range."""

import SQcircuit as sq
import numpy as np

from SQcircuit.sampler import CircuitSampler

def test_convergence():
    # Create fluxonium circuit
    def create_fluxonium(cap_value, cap_unit, ind_value, ind_unit, junction_value, junction_unit):
        loop = sq.Loop()
        loop.set_flux(0)

        capacitor = sq.Capacitor(cap_value, cap_unit, Q=1e6)
        inductor = sq.Inductor(ind_value, ind_unit, Q=500e6, loops=[loop])
        junction = sq.Junction(junction_value, junction_unit, cap=capacitor, A=1e-7, x=3e-06, loops=[loop])
        circuit_fluxonium = sq.Circuit(
            {(0, 1): [capacitor, inductor, junction], }
        )
        return circuit_fluxonium
    trunc_cutoff = 100
    trunc_range = np.arange(12, trunc_cutoff + 1)
    cutoff = 3

    num_eigenvalues = 10
    for x in trunc_range:
        fluxonium = create_fluxonium(2, 'GHz', 0.46, 'GHz', 10.2, 'GHz')
        # Assuming contiguous divergence/convergence trunc nums, find lowest truncation number that converges
        fluxonium.set_trunc_nums([x, ])
        fluxonium.diag(num_eigenvalues)
        if fluxonium.test_convergence([x, ])[0] is False:
            cutoff = x + 1
    assert cutoff == 39

def test_circuit_topologies():
    n = 3
    circuit_sampler = CircuitSampler(n)
    assert circuit_sampler.topologies == ['JJJ', 'JLJ', 'JLL'] or \
           circuit_sampler.topologies == ['JJJ', 'JJL', 'JLL']

    '''n = 4
    circuit_sampler = CircuitSampler(n)
    assert circuit_sampler.topologies == ['JJJJ', 'JJLJ', 'JLJL', 'JLLJ', 'JLLL', ]'''

def test_circuit_sampling():
    pass
    # High n, low circuit count
    # circuit_sampler = CircuitSampler(10)
    # circuits = circuit_sampler.sample_one_loop_circuits(n=1)
    # truncate_circuit

    # Low n, high circuit count
    # circuit_sampler = CircuitSampler(4)
    # circuits = circuit_sampler.sample_one_loop_circuits(100)