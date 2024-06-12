"""
test_noise.py contains the test cases for the SQcircuit noise.py
functionalities.
"""
import numpy as np

import SQcircuit.noise as noise


def test_temp_assignment():

    noise.set_temp(0.013)

    assert noise.ENV["T"] == 0.013

    noise.reset_to_default()


def test_freq_low_assignment():

    noise.set_low_freq(1, 'kHz')

    assert noise.ENV["omega_low"] == 2 * np.pi * 1e3

    noise.reset_to_default()


def test_freq_high_assignment():

    noise.set_high_freq(1, 'kHz')

    assert noise.ENV["omega_high"] == 2 * np.pi * 1e3

    noise.reset_to_default()


def test_t_exp_assignment():

    noise.set_t_exp(1, 'ms')

    assert noise.ENV["t_exp"] == 0.001

    noise.reset_to_default()

