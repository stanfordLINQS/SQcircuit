import numpy as np
import sympy as sm
from sympy import (
    Expr,
    Symbol
)
from sympy.physics.quantum import Operator

from SQcircuit.elements import (
    Capacitor,
    Inductor,
    Junction,
    Loop,
    Charge
)

from sympy.printing.latex import LatexPrinter

class ExplicitSymbol(Symbol):
    def __new__(cls, plainname, latexname):
        self = super().__new__(cls, plainname)
        self.lsymbol = Symbol(latexname)
        return self

    def _latex(self, printer):
        return printer.doprint(self.lsymbol)


class qOperator(Operator):
    """
    Class extending sympy physics operator, mostly for custom printing purposes.
    Created with a base `name` and a `subscript`. When printing in LaTeX, a hat 
    will be put over the name, but not in plaintext.
    """
    def __new__(cls, name, subscript=None):
        if subscript:
            self = super().__new__(cls,f'{name}_{subscript}')
        else:
            self = super().__new__(cls, name)
        self.opname = name
        self.sub = subscript
        return self

    def _latex(self, printer):
        if self.opname[-1] == 'd':
            tex = fr'\hat{{{self.opname[:-1]}}}^\dagger'
        else:
            tex = fr'\hat{{{self.opname}}}'
        if self.sub:
            tex  += fr'_{{{self.sub}}}'
        return printer.doprint(Symbol(tex))

def phi_op(i):
    return ExplicitSymbol(f'phi_{i}',
                          fr'\hat{{\varphi}}_{{{i}}}')

def a(i: int) -> qOperator:
    return qOperator('a', i)

def ad(i: int) -> qOperator:
    return qOperator('ad', i)

def n(i: int) -> qOperator:
    return qOperator('n', i)

def phi_ext(i: int) -> ExplicitSymbol:
    return ExplicitSymbol(f'phi_e{i}',
                          r'\varphi_{\text{ext}_{' + str(i) + '}}')

def phi_zp(i: int) -> ExplicitSymbol:
    return ExplicitSymbol(f'zp_{i}',
                          r'\varphi_{\text{zp}_{' + str(i) + '}}')

def omega(i: int) -> Symbol:
    return Symbol(rf'omega_{i}')

def ng(i: int) -> ExplicitSymbol:
    return ExplicitSymbol(f'ng_{i}',
                          fr'n_{{g_{{{i}}}}}')

def EC(i: int, j: int) -> ExplicitSymbol:
    return ExplicitSymbol(f'EC_{i}{j}',
                          fr'E_{{C_{{{i}{j}}}}}')

def EL(i: int) -> ExplicitSymbol:
    return ExplicitSymbol(f'EL_{i}',
                          fr'E_{{L_{{{i}}}}}')

def EJ(i: int) -> ExplicitSymbol:
    return ExplicitSymbol(f'EJ_{i}',
                          fr'E_{{J_{{{i}}}}}')

H = qOperator('H')


def har_mode_hamil(coeff_dict):
    hamil = sm.Add(*[sm.Mul(omega(i+1), ad(i+1), a(i+1), evaluate=False) \
                    for i in range(coeff_dict['har_dim'])], evaluate=False)
    return hamil

def charge_mode_hamil(coeff_dict):
    hamil = 0
    for i in range(coeff_dict['har_dim'], coeff_dict['n_modes']):
        for j in range(i, coeff_dict['n_modes']):
            hamil += EC(i+1,j+1) * (n(i+1) - ng(i+1)) * (n(i+1) -ng(j+1))
    return hamil

def inductive_hamil(elem_keys, coeff_dict):
    S = coeff_dict['S']
    B = coeff_dict['B']
    hamil = 0
    for i, (edge, el, B_idx) in enumerate(elem_keys[Inductor]):
        if np.sum(np.abs(B[B_idx, :])) == 0:
            continue
        if 0 in edge:
            w = S[edge[0] + edge[1] - 1, :]
        else:
            w = S[edge[0] - 1, :] - S[edge[1] - 1, :]
        w = np.round(w[:coeff_dict['har_dim']], 3)

        w_sym = sm.Matrix(w)
        B_sym = sm.Matrix(B[B_idx, :])

        phis = [phi_op(i+1) for i in range(len(w_sym))]
        phi_exts = [phi_ext(i+1) for i in range(len(B_sym))]

        hamil += sm.Mul(EL(i+1), sm.nsimplify(w_sym.dot(phis)), 
                        sm.nsimplify(B_sym.dot(phi_exts)), evaluate=False)
    return hamil

def jj_hamil(elem_keys, coeff_dict):
    hamil = 0
    for i, (edge, el, B_idx, W_idx) in enumerate(elem_keys[Junction]):
        W_sym = sm.Matrix(coeff_dict['W'][W_idx, :])
        phis = [phi_op(i+1) for i in range(len(W_sym))]

        B_sym = sm.Matrix(coeff_dict['B'][B_idx, :])
        phi_exts = [phi_ext(i+1) for i in range(len(B_sym))]

        hamil += EJ(i+1) * sm.cos(sm.Add(sm.nsimplify(W_sym.dot(phis)),
                                        sm.nsimplify(B_sym.dot(phi_exts)),
                                        evaluate=False))
    return hamil

def construct_hamiltonian(cr) -> Expr:
    LC_hamil = sm.Add(har_mode_hamil(cr.descrip_vars),
                      charge_mode_hamil(cr.descrip_vars), evaluate=False)
    Ind_hamil = inductive_hamil(cr.elem_keys, cr.descrip_vars)
    JJ_hamil = jj_hamil(cr.elem_keys, cr.descrip_vars)

    return sm.Add(*[H for H in [LC_hamil, Ind_hamil, JJ_hamil] if H!=0], 
                  evaluate=False)
