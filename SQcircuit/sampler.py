from collections import defaultdict
import random
from scipy.stats import loguniform
from typing import Set

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
        self.topologies = _generate_topologies(num_elements)
        self.capacitor_range = [12e-6, 12]
        self.capacitor_unit = 'nF'
        self.inductor_range = [12e-3, 12]
        self.inductor_unit = 'uH'
        self.junction_range = [1, 10]
        self.junction_unit = 'GHz'
        self.trunc_num = 40

    def sample_circuit(self):
        circuit_topology = random.sample(self.topologies, 1)[0]
        print(circuit_topology)
        # Build circuit, assign random values by sampling from range for each element

    def sample_one_loop_circuits(self, n, with_replacement = True) -> [Circuit]:
        circuits = []
        if not with_replacement:
            assert n <= len(self.topologies), "Number of circuit topologies sampled without replacement must be less" \
                                              "than or equal to number of distinct arrangements of inductive elements."
            sampled_topologies = random.sample(list(self.topologies), n)
        else:
            sampled_topologies = random.choices(list(self.topologies), k = n)

        for topology in sampled_topologies:
            loop = Loop()
            loop.set_flux(0)
            circuit_elements = defaultdict(list)

            # Add inductive elements to circuit
            for element_idx, element_code in enumerate(topology):
                if element_code == 'J':
                    # Add requires grad to element here?
                    junction_value = loguniform.rvs(*self.junction_range, size=1)
                    element = Junction(junction_value, self.junction_unit, loops=[loop], requires_grad=get_optim_mode())
                elif element_code == 'L':
                    # TODO: Include default quality factor Q in inductor?
                    inductor_value = loguniform.rvs(*self.inductor_range, size=1)
                    element = Inductor(inductor_value, self.inductor_unit, loops=[loop], requires_grad=get_optim_mode())
                circuit_elements[(element_idx, (element_idx + 1) % len(topology))] = [element, ]

            # Introduce all-to-all capacitive coupling
            for first_element_idx in range(len(topology)):
                for second_element_idx in range(first_element_idx + 1, len(topology)):
                    capacitor_value = loguniform.rvs(*self.capacitor_range, size=1)
                    capacitor = Capacitor(capacitor_value, self.capacitor_unit, requires_grad=get_optim_mode())
                    circuit_elements[(first_element_idx, second_element_idx)] += [capacitor, ]

            circuit = Circuit(circuit_elements, flux_dist='all')
            # If mode j > 100 * mode i, set mode j trunc num to 1
            # circuit.set_trunc_nums([np.pow(1000,1/n), np.pow(1000,1/n), np.pow(1000,1/n), np.pow(1000,1/n)])
            # Weight based on natural frequency?
            circuits.append(circuit)

        return circuits