"""Module for text and LaTeX output of SQcircuit"""

import numpy as np
from IPython.display import display, Math
import sympy as sm
from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.printing.pretty.stringpict import prettyForm

from SQcircuit.elements import Inductor, Junction
import SQcircuit.symbolic as sym


def is_notebook():
    """Checks whether we are working in notebook environment or the Python
    terminal.
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


def newDiv(self, den):
    """prettyForm division to always be inline x/y
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
            self.printer = LatexPrinter({'order': 'none', 'fold_short_frac': True})
        elif tp == 'txt':
            prettyForm.__truediv__ = newDiv
            self.printer = PrettyPrinter({'use_unicode': False, 'order': 'none'})
        else:
            raise ValueError('Permitted values for `tp` are \'ltx\' and '
                             '\'txt\'.')

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
        for i in range(coeff_dict['n_modes']):
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
                ng_val = np.round(coeff_dict['ng'][i - coeff_dict['har_dim']], 3)
                info = [sm.Eq(sym.ng(i+1), ng_val)]
            txt += (
                self.plaintxt(f'mode {i+1}:') + self.tab()
                + self.plaintxt(kind) + self.tab()
            )
            txt += self.tab().join([self.printer.doprint(e) for e in info]) + '\n'
        return txt + '\n'

    def param_txt(self, coeff_dict):
        params = []
        # For the LC part, only print out if not completely absorbed into the
        # omegas of the harmonic modes.
        ## Only capacitve energies on charge modes
        for i in range(coeff_dict['har_dim'], coeff_dict['n_modes']):
            for j in range(i, coeff_dict['n_modes']):
                params.append(sm.Eq(sym.EC(i+1,j+1),
                                    np.round(coeff_dict['EC'][i, j], 3)))
        ## Only inductive energy if it has external flux assigned to it
        for i, EL_val in enumerate(coeff_dict['EL']):
            if coeff_dict['EL_has_ext_flux'][i]:
                params.append(sm.Eq(sym.EL(i+1),
                                    np.round(EL_val, 3)))
        # Need to print all junctions
        for i, EJ_val in enumerate(coeff_dict['EJ']):
            params.append(sm.Eq(sym.EJ(i+1),
                                np.round(EJ_val, 3)))
        txt = (
            self.plaintxt('parameters:') + self.tab()
            + self.tab().join([self.printer.doprint(p) for p in params])
        )
        return txt + '\n'

    def loop_txt(self, coeff_dict):
        info = [sm.Eq(sym.phi_ext(i+1)/(2*sm.pi),
                      np.round(coeff_dict['loops'][i], 2))
                for i in range(coeff_dict['n_loops'])]
        txt = (
            self.plaintxt('loops:') + self.tab()
            + self.tab().join([self.printer.doprint(l) for l in info])
        )
        return txt

    def print_circuit_description(self, coeff_dict):
        finalTxt = (
            self.ham_txt(coeff_dict) + self.line
            + self.mode_txt(coeff_dict) + self.line 
            + self.param_txt(coeff_dict)
            + self.loop_txt(coeff_dict)
        )

        self.display(finalTxt)
        return finalTxt

    @staticmethod
    def print_loop_description(cr):
        # maximum length of element ID strings
        nr = max(
            [len(el.id_str) for _, el, _, _ in cr.elem_keys[Junction]]
            + [len(el.id_str) for _, el, _ in cr.elem_keys[Inductor]]
        )

        # maximum length of loop ID strings
        nh = max([len(lp.id_str) for lp in cr.loops])

        # number of loops
        nl = len(cr.loops)

        # space between elements in rows
        ns = 5

        # header with names of loops
        header = (nr + ns + len(', b1:')) * " "
        header += (' ' * 10).join([f'{lp.id_str:{nh}}' for lp in cr.loops])

        loop_description_txt = header + '\n'

        # add line under header
        loop_description_txt += "-" * len(header) + '\n'


        for i in range(cr.B.shape[0]):
            el = None
            for _, el_ind, b_id in cr.elem_keys[Inductor]:
                if i == b_id:
                    el = el_ind
            for _, el_ind, b_id, _ in cr.elem_keys[Junction]:
                if i == b_id:
                    el = el_ind

            id_str = el.id_str
            row = f'{id_str:{nr}}' + f'{{:{ns + len(", b1:")}}}'.format(', b' + str(i+1) + ':')
            row += ('').join([f'{np.round(np.abs(cr.B[i, j]), 2):<{nh + 10}}'
                              for j in range(nl)])
            loop_description_txt += row + '\n'

        print(loop_description_txt)
        return loop_description_txt

    def display(self, text):
        if self.tp == 'ltx':
            for line in text.split('\n'):
                display(Math(line))
        elif self.tp == 'txt':
            print(text)
