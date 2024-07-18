"""
test_qubits contains the test cases for well-known qubits.
"""
from typing import Optional

import numpy as np

import SQcircuit as sq

from SQcircuit import Circuit


def assert_qubit_freq_sweep(
    circuit: Circuit,
    target_freqs: np.ndarray,
    target_decays: Optional[dict] = None,
    sweep_type: str = 'flux',
    n_points: int = 3,
) -> None:
    """This function assumes that circuit has only either one loop or one
    charge island."""

    params = np.linspace(0, 0.5, n_points)
    spec = np.zeros(n_points)

    decays = {}
    if target_decays is not None:
        for dec_channel in target_decays.keys():
            decays[dec_channel] = np.zeros(n_points)

    for i, param in enumerate(params):

        if sweep_type == 'flux':
            circuit.loops[0].set_flux(param)
        elif sweep_type == 'charge':
            circuit.set_charge_offset(1, param)

        # diagonalize the circuit
        efreqs, _ = circuit.diag(2)

        spec[i] = efreqs[1] - efreqs[0]

        for dec_channel in decays.keys():
            decays[dec_channel][i] = circuit.dec_rate(
                dec_type=dec_channel, states=(1, 0)
            )

    assert np.allclose(spec, target_freqs, rtol=0.01)

    for dec_channel in decays.keys():
        assert np.allclose(
            decays[dec_channel], target_decays[dec_channel], atol=1e-5
        )


def test_zero_pi() -> None:

    target_freqs = np.array([0.69721321, 0.45628775, 0.0239571])

    loop1 = sq.Loop()

    # define the circuit ’s elements
    C = sq.Capacitor(0.15, "GHz")
    CJ = sq.Capacitor(10, "GHz")
    JJ = sq.Junction(5, "GHz", loops=[loop1])
    L = sq.Inductor(0.13, "GHz", loops=[loop1])

    # define the circuit
    elements = {
        (0, 1): [CJ, JJ],
        (0, 2): [L],
        (0, 3): [C],
        (1, 2): [C],
        (1, 3): [L],
        (2, 3): [CJ, JJ]
    }

    zrpi = sq.Circuit(elements)

    zrpi.set_trunc_nums([25, 1, 13])

    assert_qubit_freq_sweep(zrpi, target_freqs)


def test_inductively_shunted() -> None:

    target_freqs = np.array([6.65474896, 3.73775924, 1.24085844])

    loop1 = sq.Loop()

    # define the circuit ’s elements
    C_r = sq.Capacitor(20.3, "fF")
    C_q = sq.Capacitor(5.3, "fF")
    L_r = sq.Inductor(15.6, "nH")
    L_q = sq.Inductor(386, "nH", loops=[loop1])
    L_s = sq.Inductor(4.5, "nH", loops=[loop1])
    JJ = sq.Junction(6.2, "GHz", loops=[loop1])

    # define the circuit
    elements = {
        (0, 1): [C_r],
        (1, 2): [L_r],
        (0, 2): [L_s],
        (2, 3): [L_q],
        (0, 3): [JJ, C_q]
    }

    cr = sq.Circuit(elements)

    cr.set_trunc_nums([1, 9, 23])

    assert_qubit_freq_sweep(cr, target_freqs)


def test_fluxonium_with_added_node() -> None:

    target_freqs = np.array([5.94930425, 4.83738897, 1.75964137])

    # fluxonium with charge island(added by Yudan)
    loop1 = sq.Loop()
    C = sq.Capacitor(11, 'fF', Q=1e6)
    Cg = sq.Capacitor(0.5, 'fF', Q=1e6)
    L = sq.Inductor(1, 'GHz', Q=500e6, loops=[loop1])
    JJ = sq.Junction(3, 'GHz', cap=C, A=5e-7, loops=[loop1])

    elements = {
        (0, 1): [Cg],
        (1, 2): [L, JJ],
        (0, 2): [Cg]
    }

    cr = sq.Circuit(elements)

    cr.set_trunc_nums([30, 1])

    assert_qubit_freq_sweep(cr, target_freqs)


def test_fluxonium() -> None:

    target_freqs = np.array([8.21271281, 4.21408414, 0.63935901])

    target_decays = {
        'capacitive': np.array([7230.42954319, 1831.72786237, 1935.58917645]),
        'inductive': np.array([2.84034766,   2.73297225, 125.4597119]),
        'cc': np.array([4209.46216758,  961.34657151, 2953.53540182]),
        'quasiparticle': np.array([9.09495148, 3.06603484, 0.]),
        'flux': np.array([8.80960173e-09, 4.53860939e+05, 3.63215029e-05]),
    }

    loop1 = sq.Loop()

    # define the circuit elements
    cap = sq.Capacitor(3.6, 'GHz', Q=1e6)
    ind = sq.Inductor(0.46, 'GHz', Q=500e6, loops=[loop1])
    junc = sq.Junction(
        10.2, 'GHz', cap=cap, A=1e-7, x=3e-06, loops=[loop1]
    )

    # define the circuit
    elements = {
        (0, 1): [ind, junc]
    }

    cr = sq.Circuit(elements, flux_dist='all')

    cr.set_trunc_nums([100])

    assert_qubit_freq_sweep(cr, target_freqs, target_decays)


def test_transmon() -> None:
    
    target_freqs = np.array([81.15024681, 42.87817012, 14.96712909])

    target_decays = {
        'capacitive': np.array([209194.40115, 245791.03928, 961486.30626]),
        'inductive': np.array([0, 0, 0]),
        'cc': np.array([6251.40470818, 15286.3367821, 41186.35210817]),
    }
    
    cap = sq.Capacitor(20, 'GHz')
    junc = sq.Junction(15.0, 'GHz', A=1e-7)

    elements = {
        (0, 1): [cap, junc]
    }

    cr = sq.Circuit(elements)

    cr.set_trunc_nums([100])

    assert_qubit_freq_sweep(
        cr, target_freqs, target_decays, sweep_type='charge'
    )


def test_resonator():
    """
    function for testing simple resonator.
    """
    ind = sq.Capacitor(1 / 2 / np.pi, 'pF', Q=1e6)
    cap = sq.Inductor(1 / 2 / np.pi, 'uH', Q=500e6)

    elements = {(0, 1): [ind, cap]}

    cr = sq.Circuit(elements)

    assert np.isclose(cr.omega/2/np.pi/1e9, 1)[0]

    # check if cr.description() run without error
    cr.description(_test=True)


def test_coupled_fluxonium_transmon():
    """Function for testing coupled fluxonium to the transmon."""

    target_efreqs = np.array([
        0., 2.07145177, 2.34187898, 3.09228726, 4.92917697, 5.19354651,
        5.87412504, 7.49054673, 7.74403853, 8.2146904
    ])

    loop1 = sq.Loop()

    # define the circuit elements
    C = sq.Capacitor(1, 'GHz')
    L = sq.Inductor(1, 'GHz', loops=[loop1])
    JJ = sq.Junction(1, 'GHz', loops=[loop1])
    JJ2 = sq.Junction(1, 'GHz')
    C_c = sq.Capacitor(1, 'GHz')

    elements = {
        (0, 1): [L, JJ, C],
        (0, 2): [JJ2, C],
        (1, 2): [C_c],
    }

    cr = sq.Circuit(elements)

    cr.set_trunc_nums([30, 10])

    efreqs, _ = cr.diag(10)

    efreqs = efreqs - efreqs[0]

    assert np.allclose(efreqs, target_efreqs)
