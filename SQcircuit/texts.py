""" Module for text and latex output of SQcircuit"""

import numpy as np
from IPython.display import display, Latex


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

    def Ec(self, i, j):
        if self.tp == 'ltx':
            return "E_{C_{" + f"{i}{j}" + "}}"
        elif self.tp == 'txt':
            return f"EC_{i}{j}"

    def Ej(self, i):
        if self.tp == 'ltx':
            return "E_{J_{" + f"{i}" + "}}"
        elif self.tp == 'txt':
            return f"EJ_{i}"

    def El(self, i):
        if self.tp == 'ltx':
            return "E_{L_{" + f"{i}" + "}}"
        elif self.tp == 'txt':
            return f"EL_{i}"

    def phi(self, i):
        if self.tp == 'ltx':
            return "\hat{\\varphi}_" + f"{i}"
        elif self.tp == 'txt':
            return f"\u03C6_{i}"

    def phiExt(self, i):
        if self.tp == 'ltx':
            return "\\varphi_{\\text{ext}_{" + f"{i}" + "}}"
        elif self.tp == 'txt':
            return f"\u03C6_e{i}"

    def zp(self, i):
        if self.tp == 'ltx':
            return "\\varphi_{zp_{" + f"{i}" + "}}"
        elif self.tp == 'txt':
            return f"zp_{i}"

    def omega(self, i, F=True):
        if self.tp == 'ltx':
            if F:
                return f"\omega_{i}"
            else:
                return f"\omega_{i}" + "/2\pi"
        elif self.tp == 'txt':
            if F:
                return f"\u03C9_{i}"
            else:
                return f"\u03C9_{i}" + "/2\u03C0"

    def tPi(self):
        if self.tp == 'ltx':
            return "/2\pi"
        elif self.tp == 'txt':
            return "/2\u03C0"

    def a(self, i):
        if self.tp == 'ltx':
            return f"\hat a_{i}"
        elif self.tp == 'txt':
            return f"a_{i}"

    def ad(self, i):
        if self.tp == 'ltx':
            return f"\hat a^\dagger_{i}"
        elif self.tp == 'txt':
            return f"ad_{i}"

    def n(self, i, j):
        if self.tp == 'ltx':
            if i == j:
                return "(\hat{n}_" + f"{i}" + "-n_{g_{" + f"{i}" + "}})^2"
            else:
                return "(\hat{n}_" + f"{i}" + "-n_{g_{" + f"{i}" + "}})" +\
                       "(\hat{n}_" + f"{j}" + "-n_{g_{" + f"{j}" + "}})"
        elif self.tp == 'txt':
            if i == j:
                return f"(n_{i}-ng_{i})^2"
            else:
                return f"(n_{i}-ng_{i})(n_{j}-ng_{j})"

    def ng(self, i):
        if self.tp == 'ltx':
            return "n_{g_{" + f"{i}" + "}}"

        elif self.tp == 'txt':
            return f"ng_{i}"

    def mode(self, i):
        if self.tp == 'ltx':
            return "\\text{mode}" + f"~{i}:"
        elif self.tp == 'txt':
            return f"mode {i}:"

    def param(self):
        if self.tp == 'ltx':
            return "\\text{parameters}:"
        elif self.tp == 'txt':
            return f"parameters:"

    def har(self):
        if self.tp == 'ltx':
            return "\\text{harmonic}"
        elif self.tp == 'txt':
            return "harmonic"

    def ch(self):
        if self.tp == 'ltx':
            return "\\text{charge}~~~~~"
        elif self.tp == 'txt':
            return "charge  "

    def loops(self):
        if self.tp == 'ltx':
            return "\\text{loops}:~~~~~~~~~"
        elif self.tp == 'txt':
            return f"loops:     "

    def H(self):
        if self.tp == 'ltx':
            return "\hat{H} =~"
        elif self.tp == 'txt':
            return "H = "

    def cos(self):
        if self.tp == 'ltx':
            return "\cos"
        elif self.tp == 'txt':
            return "cos"

    def eq(self):
        if self.tp == 'ltx':
            return "~=~"
        elif self.tp == 'txt':
            return " = "

    def tab(self):
        if self.tp == 'ltx':
            return 11*"~"
        elif self.tp == 'txt':
            return 7*" "

    def p(self):
        if self.tp == 'ltx':
            return "~+~"
        elif self.tp == 'txt':
            return " + "

    @staticmethod
    def ltx(txt):
        return f"${txt}$"

    @staticmethod
    def linear(method, w, st=True):

        txt = ''

        for j in range(len(w)):

            if w[j] == 1:
                if np.sum(np.abs(w[:j])) == 0 and st is True:
                    txt += method(j+1)
                else:
                    txt += '+' + method(j+1)
            elif w[j] == -1:
                txt += '-' + method(j+1)
            elif w[j] == 0:
                continue
            else:
                if w[j] > 0:
                    if np.sum(np.abs(w[:j])) == 0 and st is True:
                        txt += str(w[j]) + method(j+1)
                    else:
                        txt += '+' + str(w[j]) + method(j+1)
                else:
                    txt += '-' + str(-w[j]) + method(j + 1)

        return txt

    def display(self, text):

        if self.tp == 'ltx':
            for line in text.split('\n'):
                display(Latex(self.ltx(line)))
        elif self.tp == 'txt':
            print(text)
