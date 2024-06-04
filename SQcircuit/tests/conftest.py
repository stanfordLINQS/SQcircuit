"""
conftest.py contains the general test classes.
"""
import os

import numpy as np

from SQcircuit import set_optim_mode
from SQcircuit.sweep import *
from SQcircuit.storage import SQdata
from SQcircuit.circuit import Circuit

TESTDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(TESTDIR, "data/qubits")


class QubitTest:
    """
    class that contains the general tests
    """

    @classmethod
    def setup_class(cls):
        cls.fileName = None

    def test_transform_process(self):
        # load the data
        set_optim_mode(False)
        data = SQdata.load(DATADIR + "/" + self.fileName)

        # build the new circuit based on data circuit parameters
        newCr = Circuit(data.cr.elements)

        # check the modes and natural frequencies
        assert np.allclose(newCr.omega, data.cr.omega)
        # check the transformed w matrix
        assert np.allclose(newCr.wTrans, data.cr.wTrans)

    def test_if_description_run(self):
        """ Test if description run without error"""
        # load the data
        data = SQdata.load(DATADIR + "/" + self.fileName)

        # build the new circuit based on data circuit parameters
        newCr = Circuit(data.cr.elements)

        newCr.description(_test=True)

    def test_data(self):
        # load the data
        data = SQdata.load(DATADIR + "/" + self.fileName)

        efreq = None
        dec = None

        # build the new circuit based on data circuit parameters
        newCr = Circuit(data.cr.elements)
        newCr.set_trunc_nums(data.cr.m)

        if data.dec:
            properties = ["efreq", "loss"]
        else:
            properties = None

        numEig = data.efreq.shape[0]

        sweep1 = Sweep(newCr, numEig, properties)

        if data.type == "sweepFlux":
            efreq, dec = sweep1.sweepFlux(data.params, data.grid)
        elif data.type == "sweepCharge":
            efreq, dec = sweep1.sweepCharge(data.params, data.grid)

        for i in range(efreq.shape[0]):
            assert np.allclose(efreq[i, :], data.efreq[i, :],
                               rtol=1e-4, atol=1e-3)

        if data.dec:
            for decType in data.dec.keys():
                assert np.allclose(dec[decType], data.dec[decType]),\
                    "The \"{}\" loss has issue".format(decType)
                

def create_transmon_numpy(trunc_num):
    cap_value, ind_value, Q = 7.746, 5, 1e6
    cap_unit, ind_unit = 'fF', 'GHz'

    set_optim_mode(False)
    C_numpy = Capacitor(cap_value, cap_unit, Q=Q)
    J_numpy = Junction(ind_value, ind_unit)
    circuit_numpy = Circuit({(0, 1): [C_numpy, J_numpy], })
    circuit_numpy.set_trunc_nums([trunc_num, ])
    return circuit_numpy

def create_transmon_torch(trunc_num):
    cap_value, ind_value, Q = 7.746, 5, 1e6
    cap_unit, ind_unit = 'fF', 'GHz'

    set_optim_mode(True)
    C_torch = Capacitor(cap_value, cap_unit, Q=Q, requires_grad=True)
    J_torch = Junction(ind_value, ind_unit, requires_grad=True)
    circuit_torch = Circuit({(0, 1): [C_torch, J_torch]})
    circuit_torch.set_trunc_nums([trunc_num, ])
    return circuit_torch


def create_flux_transmon_numpy(trunc_num, phi_ext):
    cap_value, ind_value, Q = 7.746, 5, 1e6
    cap_unit, ind_unit = 'fF', 'GHz'
    disorder = 1.1

    set_optim_mode(False)
    loop = Loop(phi_ext)
    C_numpy = Capacitor(cap_value, cap_unit, Q=Q)
    J1_numpy = Junction(ind_value, ind_unit, loops=[loop])
    J2_numpy = Junction(ind_value * disorder, ind_unit, loops=[loop])
    circuit_numpy = Circuit({(0, 1): [C_numpy, J1_numpy, J2_numpy], })
    circuit_numpy.set_trunc_nums([trunc_num, ])
    return circuit_numpy


def create_flux_transmon_torch(trunc_num, phi_ext):
    cap_value, ind_value, Q = 7.746, 5, 1e6
    cap_unit, ind_unit = 'fF', 'GHz'
    disorder = 1.1

    set_optim_mode(True)
    loop_torch = Loop(phi_ext, requires_grad=True)
    C_torch = Capacitor(cap_value, cap_unit, Q=Q, requires_grad=True)
    J1_torch = Junction(ind_value, ind_unit, loops=[loop_torch], requires_grad=True)
    J2_torch = Junction(ind_value * disorder, ind_unit, loops=[loop_torch], requires_grad=True)
    circuit_torch = Circuit({(0, 1): [C_torch, J1_torch, J2_torch], })
    circuit_torch.set_trunc_nums([trunc_num, ])
    return circuit_torch


def create_fluxonium_numpy(trunc_num, phi_ext=0):
    set_optim_mode(False)
    loop = Loop(phi_ext)
    C_numpy = Capacitor(3.6, 'GHz', Q=1e6, requires_grad=False)
    L_numpy = Inductor(0.46, 'GHz', Q=500e6, loops=[loop], requires_grad=False)
    JJ_numpy = Junction(10.2, 'GHz', A=1e-7, x=3e-06, loops=[loop], requires_grad=False)
    circuit_numpy = Circuit({(0, 1): [C_numpy, L_numpy, JJ_numpy], }, flux_dist='junctions')
    circuit_numpy.set_trunc_nums([trunc_num, ])
    return circuit_numpy


def create_fluxonium_torch(trunc_num, phi_ext=0):
    set_optim_mode(True)
    loop = Loop(phi_ext)
    C_torch = Capacitor(3.6, 'GHz', Q=1e6, requires_grad=True)
    L_torch = Inductor(0.46, 'GHz', Q=500e6, loops=[loop], requires_grad=True)
    JJ_torch = Junction(10.2, 'GHz', A=1e-7, x=3e-06, loops=[loop], requires_grad=True)
    circuit_torch = Circuit({(0, 1): [C_torch, L_torch, JJ_torch], }, flux_dist='junctions')
    circuit_torch.set_trunc_nums([trunc_num, ])
    return circuit_torch


def create_fluxonium_torch_flux(trunc_num, phi_ext=0):
    set_optim_mode(True)
    loop = Loop(phi_ext, requires_grad=True)
    C_torch = Capacitor(3.6, 'GHz', Q=1e6, requires_grad=True)
    L_torch = Inductor(0.46, 'GHz', Q=500e6, loops=[loop], requires_grad=True)
    JJ_torch = Junction(10.2, 'GHz', A=1e-7, x=3e-06, loops=[loop], requires_grad=True)
    circuit_torch = Circuit({(0, 1): [C_torch, L_torch, JJ_torch], }, flux_dist='junctions')
    circuit_torch.set_trunc_nums([trunc_num, ])
    return circuit_torch
