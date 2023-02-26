""" Module for text and latex output of SQcircuit"""

import numpy as np
from IPython.display import display, Latex, Math
import sympy as sm
from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.pretty import PrettyPrinter
from typing import List

import SQcircuit.symbolic as symbolic


def is_notebook():
    """
    The function that checks whether we are working in notebook environment or
    Python terminal.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class HamilTxt:
    """
    Class that contains the methods for printing the Hamiltonian in text or
    latex format.
    """

    def __init__(self, tp='ltx'):
        self.tp = tp
        self.line = 60 * "-" + "\n"
        if tp == 'ltx':
            self.printer = LatexPrinter(dict(order='none', fold_short_frac=True))
        else:
            self.printer = PrettyPrinter(dict(use_unicode=True, order='none'))

    def tab(self):
        if self.tp == 'ltx':
            return 11*"~"
        elif self.tp == 'txt':
            return 7*" "
        
    def plaintxt(self, text):
        if self.tp == 'ltx':
            return rf'\text{{{text}}}'
        else:
            return text

    def ham_txt(self, sym_ham):
        return self.printer.doprint(sm.Eq(sm.symbols('\hat{H}'), sym_ham)) + '\n'
    
    def mode_txt(self, modes):
        txt = ''
        for mode in modes:
            txt += self.plaintxt(f'mode {mode[0]}:') + self.tab() \
                    + self.plaintxt(mode[1]) + self.tab()
            txt += self.tab().join([self.printer.doprint(e) for e in mode[2:]])
            txt += '\n'
        return txt

    def param_txt(self, params):
        txt = self.plaintxt('parameters:') + self.tab() \
                + self.tab().join([self.printer.doprint(p) for p in params])
        txt += '\n'
        return txt
    
    def loop_txt(self, loops):
        txt = self.plaintxt('loops:') + self.tab() \
                + self.tab().join([self.printer.doprint(l) for l in loops])
        txt += '\n'
        return txt

    def display(self, text):
        if self.tp == 'ltx':
            for line in text.split('\n'):
                display(Math(line))
        elif self.tp == 'txt':
            print(text)
