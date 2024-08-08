"""
test_elements contains the test cases for the SQcircuit elements functionalities.
"""
import os

import SQcircuit as sq

TESTDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(TESTDIR, "data/descriptions")


def test_zeropi_description():

    loop1 = sq.Loop(0.5)

    C = sq.Capacitor(0.15, "GHz")
    CJ = sq.Capacitor(10, "GHz")
    JJ = sq.Junction(5, "GHz", loops=[loop1])
    L = sq.Inductor(0.13, "GHz", loops=[loop1])

    elements = {
        (0, 1): [CJ, JJ],
        (0, 2): [L],
        (3, 0): [C],
        (1, 2): [C],
        (1, 3): [L],
        (2, 3): [CJ, JJ],
    }

    ###########################################################################
    # flux_dist = "junctions"
    ###########################################################################

    cr = sq.Circuit(elements, flux_dist='junctions')

    txt = cr.description(tp='txt', _test=True)

    with open(DATADIR + '/zero_pi_junctions_txt-a.txt', 'r') as f:
        txt_data_a = f.read()
    with open(DATADIR + '/zero_pi_junctions_txt-b.txt', 'r') as f:
        txt_data_b = f.read()

    assert (txt == txt_data_a) or (txt == txt_data_b)

    ltx = cr.description(tp='ltx', _test=True)

    with open(DATADIR + '/zero_pi_junctions_ltx-a.txt', 'r') as f:
        ltx_data_a = f.read()
    with open(DATADIR + '/zero_pi_junctions_ltx-b.txt', 'r') as f:
        ltx_data_b = f.read()

    assert (ltx == ltx_data_a) or (ltx == ltx_data_b)

    ###########################################################################
    # flux_dist = "inductors"
    ###########################################################################

    cr = sq.Circuit(elements, flux_dist='inductors')

    txt = cr.description(tp='txt', _test=True)

    with open(DATADIR + '/zero_pi_inductors_txt-a.txt', 'r') as f:
        txt_data_a = f.read()
    with open(DATADIR + '/zero_pi_inductors_txt-b.txt', 'r') as f:
        txt_data_b = f.read()

    assert (txt == txt_data_a) or (txt == txt_data_b)

    ltx = cr.description(tp='ltx', _test=True)

    with open(DATADIR + '/zero_pi_inductors_ltx-a.txt', 'r') as f:
        ltx_data_a = f.read()
    with open(DATADIR + '/zero_pi_inductors_ltx-b.txt', 'r') as f:
        ltx_data_b = f.read()

    assert (ltx == ltx_data_a) or (ltx == ltx_data_b)


def test_loop_description():

    C = sq.Capacitor(1)

    loop1 = sq.Loop(id_str="loop1")

    JJ1 = sq.Junction(1, loops=[loop1], cap=C, id_str="JJ1")
    JJ2 = sq.Junction(1, loops=[loop1], cap=C, id_str="JJ2")
    L = sq.Inductor(1, loops=[loop1], cap=C, id_str="ind")

    elements = {
        (0, 1): [JJ1],
        (0, 2): [JJ2],
        (1, 2): [L]
    }

    ###########################################################################
    # flux_dist = "all"
    ###########################################################################

    cr = sq.Circuit(elements, flux_dist='all')
    desc = cr.loop_description(_test=True)

    with open(DATADIR + '/flux_dist_all.txt', 'r') as f:
        desc_data = f.read()

    assert desc == desc_data

    ###########################################################################
    # flux_dist = "inductors"
    ###########################################################################

    cr = sq.Circuit(elements, flux_dist='inductors')
    desc = cr.loop_description(_test=True)

    with open(DATADIR + '/flux_dist_inductors.txt', 'r') as f:
        desc_data = f.read()

    assert desc == desc_data

    ###########################################################################
    # flux_dist = "junctions"
    ###########################################################################

    cr = sq.Circuit(elements, flux_dist='junctions')
    desc = cr.loop_description(_test=True)

    with open(DATADIR + '/flux_dist_junctions.txt', 'r') as f:
        desc_data = f.read()

    assert desc == desc_data
