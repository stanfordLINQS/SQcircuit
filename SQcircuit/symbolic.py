import numpy as np
import sympy as sm
from sympy.physics.quantum import Operator as qOperator

from SQcircuit.elements import (
    Capacitor,
    Inductor,
    Junction,
    Loop,
    Charge
)

# class ad_op(smq.operator.Operator): # inherits from QExpr inherits from Expr
#     def _latex(self, printer):
#         return r"a^\dagger"
    
def phi_op(i):
    return qOperator(rf'\hat{{\varphi}}_{{{i}}}')

def phi_ext(i):
    return sm.Symbol(r'\varphi_{\text{ext}_{' + str(i) + '}}')

def phi_zp(i):
    return sm.Symbol(r'\varphi_{\text{zp}_{' + str(i) + '}}')

def a(i):
    return qOperator(rf'\hat{{a}}_{{{i}}}')

def ad(i):
    return qOperator(rf'\hat{{a}}^\dagger_{{{i}}}')

def omega(i):
    return sm.Symbol(rf'\omega_{i}')

def n(i):
    return qOperator(rf'\hat{{n}}_{{{i}}}')

def ng(i):
    return sm.Symbol(f'n_{{g_{{{i}}}}}')

def EC(i,j):
    return sm.Symbol(f'E_{{C_{{{i}{i}}}}}')

def EL(i):
    return sm.Symbol(f'E_{{L_{{{i}}}}}')

def EJ(i):
    return sm.Symbol(f'E_{{J_{{{i}}}}}')

H = qOperator(r'\hat{H}')


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

        hamil = EJ(i+1) * sm.cos(sm.nsimplify(W_sym.dot(phis) + B_sym.dot(phi_exts)))
    return hamil
                    
def construct_hamiltonian(cr):
    LC_hamil = har_mode_hamil(cr._descrip_vars) + charge_mode_hamil(cr._descrip_vars)
    Ind_hamil = inductive_hamil(cr.elem_keys, cr._descrip_vars)
    JJ_hamil = jj_hamil(cr.elem_keys, cr._descrip_vars)
            
    return sm.Add(LC_hamil, Ind_hamil, JJ_hamil, evaluate=False)

def har_mode(coeff_dict):
    modes = []
    for i in range(coeff_dict['har_dim']):
        omega_val  = np.round(coeff_dict['omega'][i], 5)
        zp_val = np.round(coeff_dict['phi_zp'][i], 2)

        modes.append([i+1,
                      'harmonic', 
                      sm.Eq(phi_op(i+1), phi_zp(i+1)*(a(i+1) + ad(i+1))), 
                      sm.Eq(omega(i+1)/(2 * sm.pi), omega_val), 
                      sm.Eq(phi_zp(i+1), zp_val)])

    return modes

def charge_mode(coeff_dict):
    modes = []
    for i in range(coeff_dict['har_dim'], coeff_dict['n_modes']):
            ng_val = np.round(coeff_dict['ng'][i], 3)
            modes.append([i + 1,
                          'charge',
                          sm.Eq(ng(i+1), ng_val)

            ])
    return modes

def mode_info(coeff_dict):
    # [mode type, *infos]
    return har_mode(coeff_dict) + charge_mode(coeff_dict)

def param_info(coeff_dict):
    params = []
    for key in coeff_dict['EC']:
        i,j = key
        params.append(sm.Eq(EC(i+1,j+1),
                            np.round(coeff_dict['EC'][(i,j)], 2)))
    for i, EL_val in enumerate(coeff_dict['EL']):
        if coeff_dict['EL_incl'][i]:
            params.append(sm.Eq(EL(i+1),
                                np.round(EL_val, 3)))
    for i, EJ_val in enumerate(coeff_dict['EJ']):
        params.append(sm.Eq(EJ(i+1), 
                            np.round(EJ_val, 3)))
    return params

def loop_info(coeff_dict):
    return [sm.Eq(phi_ext(i+1)/(2*sm.pi), 
                  np.round(coeff_dict['loops'][i], 2)) \
            for i in range(coeff_dict['n_loops'])]



