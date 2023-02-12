from collections import defaultdict
import random
from scipy.stats import loguniform
from typing import Set
import numpy as np

from SQcircuit.circuit import Circuit
from SQcircuit.elements import Capacitor, Inductor, Junction, Loop
from SQcircuit.settings import get_optim_mode


def test_equivalent(code_1, code_2):
    '''Given two strings indicating a sequence of junctions and inductors (ex. JLJ and JJL), determines whether the
    patterns are equivalent up to cyclic permutations.'''
    assert len(code_1) == len(code_2)
    for i in range(1, len(code_2)):
        if code_1[i:] + code_1[:i] == code_2:
            return True

def filter(element_codes):
    '''Given a sequence of junctions and inductors encoded in string format (ex. JJLJ), removes any that are equivalent
    up to cyclic permutation.'''
    element_codes_list = list(element_codes)
    for i in range(len(element_codes_list)):
        j = i + 1
        while j < len(element_codes_list):
            if test_equivalent(element_codes_list[i], element_codes_list[j]):
                del element_codes_list[j]
                j -= 1
            j += 1
    return set(element_codes_list)

def _generate_topologies(num_elements) -> Set[str]:
    '''Generates the set of all unique orderings of N junctions and inductors on a one-loop ring, for later use in
    sampling random circuit topologies.'''
    assert num_elements >= 1
    element_codes = set()
    element_codes.add('J')  # There should always be at least one junction
    length = 1
    while length < num_elements:
        new_element_codes = set()
        for element_code in element_codes:
            for inductive_element in ['J', 'L']:
                new_element_codes.add(element_code + inductive_element)
        element_codes = new_element_codes
        length += 1
    element_codes = filter(element_codes)
    return element_codes

class CircuitSampler:
    """Class used to randomly sample different circuit configurations."""

    def __init__(
            self,
            num_elements: int
    ) -> None:
        self.num_elements = num_elements
        self.topologies = list(_generate_topologies(num_elements))
        self.topologies.sort()
        self.capacitor_range = [12e-15, 12e-9]
        self.inductor_range = [12e-9, 12e-6]
        self.junction_range = [1e9, 10e9]
        self.trunc_num = 40

    def sample_circuit(self):
        sampled_topology = random.sample(self.topologies, 1)[0]
        return sampled_topology
        # Build circuit, assign random values by sampling from range for each element

    def sample_circuit_code(self, codename):
        loop = Loop()
        loop.set_flux(0.5)
        circuit_elements = defaultdict(list)

        # Add inductive elements to circuit
        for element_idx, element_code in enumerate(codename):
            if element_code == 'J':
                # Add requires grad to element here?
                junction_value = loguniform.rvs(*self.junction_range, size=1)[0]
                junction_value /= (2 * np.pi)
                element = Junction(junction_value, 'Hz', loops=[loop], requires_grad=get_optim_mode(),
                                   min_value=self.junction_range[0], max_value=self.junction_range[1])
            elif element_code == 'L':
                # TODO: Include default quality factor Q in inductor?
                inductor_value = loguniform.rvs(*self.inductor_range, size=1)[0]
                element = Inductor(inductor_value, 'H', loops=[loop], requires_grad=get_optim_mode(),
                                   min_value=self.inductor_range[0], max_value=self.inductor_range[1])

            min_idx = min(element_idx, (element_idx + 1) % len(codename))
            max_idx = max(element_idx, (element_idx + 1) % len(codename))
            if self.num_elements == 2:
                # Edge case for n=2: Two elements on same edge
                circuit_elements[(min_idx, max_idx)] += [element, ]
            else:
                circuit_elements[(min_idx, max_idx)] = [element, ]

        # Introduce all-to-all capacitive coupling
        for first_element_idx in range(len(codename)):
            for second_element_idx in range(first_element_idx + 1, len(codename)):
                capacitor_value = loguniform.rvs(*self.capacitor_range, size=1)[0]
                capacitor = Capacitor(capacitor_value, 'F', requires_grad=get_optim_mode(),
                                      min_value=self.capacitor_range[0], max_value=self.capacitor_range[1])
                circuit_elements[(first_element_idx, second_element_idx)] += [capacitor, ]

        circuit = Circuit(circuit_elements, flux_dist='all')
        # If mode j > 100 * mode i, set mode j trunc num to 1
        # circuit.set_trunc_nums([np.pow(1000,1/n), np.pow(1000,1/n), np.pow(1000,1/n), np.pow(1000,1/n)])
        # Weight based on natural frequency?
        return circuit

    def sample_one_loop_circuits(self, n, with_replacement = True) -> [Circuit]:
        circuits = []
        if not with_replacement:
            assert n <= len(self.topologies), "Number of circuit topologies sampled without replacement must be less" \
                                              "than or equal to number of distinct arrangements of inductive elements."
            sampled_topologies = random.sample(self.topologies, n)
        else:
            sampled_topologies = random.choices(self.topologies, k = n)

        for topology in sampled_topologies:
            circuit = self.sample_circuit_code(topology)
            circuits.append(circuit)

        return circuits