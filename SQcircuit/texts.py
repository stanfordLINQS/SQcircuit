""" Module for text and latex output of SQcircuit"""

import numpy as np
from IPython.display import display, Latex, Math
import sympy as sm
from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from typing import List

import SQcircuit.symbolic as sym


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

def newDiv(self, den, slashed=False):
    """
    prettyForm division to always be inline x/y
    """
    return prettyForm(binding=prettyForm.DIV, s=self.s + '/' + den.s)


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
            prettyForm.__truediv__ = newDiv
            self.printer = PrettyPrinter(dict(use_unicode=False, order='none'))

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

    def ham_txt(self, coeff_dict):
        sym_ham = coeff_dict['H']
        return self.printer.doprint(sm.Eq(sym.H, sym_ham)) + '\n'
    
    def mode_txt(self, coeff_dict):
        txt = ''
        for i in range(coeff_dict['har_dim']):
            if i < coeff_dict['har_dim']:
                kind = 'harmonic'
                omega_val  = np.round(coeff_dict['omega'][i], 5)
                zp_val = np.round(coeff_dict['phi_zp'][i], 2)
                info = [sm.Eq(sym.phi_op(i+1), 
                              sym.phi_zp(i+1)*(sym.a(i+1) + sym.ad(i+1))), 
                        sm.Eq(sym.omega(i+1)/(2 * sm.pi), omega_val), 
                        sm.Eq(sym.phi_zp(i+1), zp_val)]
            else:
                kind = 'charge'
                ng_val = np.round(coeff_dict['ng'][i], 3)
                info = [sm.Eq(sym.ng(i+1), ng_val)]
            txt += self.plaintxt(f'mode {i+1}:') + self.tab() \
                    + self.plaintxt(kind) + self.tab()
            txt += self.tab().join([self.printer.doprint(e) for e in info])
        return txt + '\n'

    def param_txt(self, coeff_dict):
        params = []
        for key in coeff_dict['EC']:
            i,j = key
            params.append(sm.Eq(sym.EC(i+1,j+1),
                                np.round(coeff_dict['EC'][(i,j)], 2)))
        for i, EL_val in enumerate(coeff_dict['EL']):
            if coeff_dict['EL_incl'][i]:
                params.append(sm.Eq(sym.EL(i+1),
                                    np.round(EL_val, 3)))
        for i, EJ_val in enumerate(coeff_dict['EJ']):
            params.append(sm.Eq(sym.EJ(i+1), 
                                np.round(EJ_val, 3)))
        txt = self.plaintxt('parameters:') + self.tab() \
                + self.tab().join([self.printer.doprint(p) for p in params])
        return txt + '\n'
    
    def loop_txt(self, coeff_dict):
        info = [sm.Eq(sym.phi_ext(i+1)/(2*sm.pi), 
                      np.round(coeff_dict['loops'][i], 2)) \
                for i in range(coeff_dict['n_loops'])]
        txt = self.plaintxt('loops:') + self.tab() \
                + self.tab().join([self.printer.doprint(l) for l in info])
        txt += '\n'
        return txt
    
    def print_description(self, coeff_dict):
        finalTxt = self.ham_txt(coeff_dict) + self.line \
                    + self.mode_txt(coeff_dict) + self.line \
                    + self.param_txt(coeff_dict) \
                    + self.loop_txt(coeff_dict)

        self.display(finalTxt)
        return finalTxt

    def display(self, text):
        if self.tp == 'ltx':
            for line in text.split('\n'):
                display(Math(line))
        elif self.tp == 'txt':
            print(text)
