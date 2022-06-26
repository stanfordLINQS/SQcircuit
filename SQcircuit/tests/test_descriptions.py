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

    f = open(DATADIR + '/zero_pi_junctions_txt.txt', 'r')
    txt_data = f.read().replace("\\n", "\n")

    assert txt == txt_data

    ltx = cr.description(tp='ltx', _test=True)

    f = open(DATADIR + '/zero_pi_junctions_ltx.txt', 'r')
    ltx_data = (f.read().replace("\\n", "\n")).replace("\\\\", "\\")

    assert ltx_data == ltx

    ###########################################################################
    # flux_dist = "inductors"
    ###########################################################################

    cr = sq.Circuit(elements, flux_dist='inductors')

    txt = cr.description(tp='txt', _test=True)

    f = open(DATADIR + '/zero_pi_inductors_txt.txt', 'r')
    txt_data = f.read().replace("\\n", "\n")

    assert txt == txt_data

    ltx = cr.description(tp='ltx', _test=True)

    f = open(DATADIR + '/zero_pi_inductors_ltx.txt', 'r')
    ltx_data = (f.read().replace("\\n", "\n")).replace("\\\\", "\\")

    assert ltx_data == ltx


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

    f = open(DATADIR + '/flux_dist_all.txt', 'r')
    desc_data = f.read().replace("\\n", "\n")

    assert desc == desc_data

    ###########################################################################
    # flux_dist = "inductors"
    ###########################################################################

    cr = sq.Circuit(elements, flux_dist='inductors')
    desc = cr.loop_description(_test=True)

    f = open(DATADIR + '/flux_dist_inductors.txt', 'r')
    desc_data = f.read().replace("\\n", "\n")

    assert desc == desc_data

    ###########################################################################
    # flux_dist = "junctions"
    ###########################################################################

    cr = sq.Circuit(elements, flux_dist='junctions')
    desc = cr.loop_description(_test=True)

    f = open(DATADIR + '/flux_dist_junctions.txt', 'r')
    desc_data = f.read().replace("\\n", "\n")

    assert desc == desc_data
