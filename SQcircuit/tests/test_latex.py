"""
test_elements contains the test cases for the SQcircuit elements functionalities.
"""
import SQcircuit as sq


def test_zeropi_description():
    loop1 = sq.Loop(0.5)

    C = sq.Capacitor(0.15, "GHz")
    CJ = sq.Capacitor(10, "GHz")
    JJ = sq.Junction(5, "GHz", loops=[loop1])
    L = sq.Inductor(0.13, "GHz", loops=[loop1])

    elements = {(0, 1): [CJ, JJ],
                (0, 2): [L],
                (3, 0): [C],
                (1, 2): [C],
                (1, 3): [L],
                (2, 3): [CJ, JJ],
                }

    # cr is an object of Qcircuit
    cr = sq.Circuit(elements, flux_dist='junctions')

    txt = cr.description(tp='txt', _test=True)

    assert txt == 'H = ω_1ad_1a_1 + ω_2ad_2a_2 + EC_33(n_3-ng_3)^2 + EJ_1cos(-φ_1+φ_3+0.5φ_e1) + EJ_2cos(-φ_1-φ_3+0.5φ_e1) \n------------------------------------------------------------\nmode 1:       harmonic       φ_1 = zp_1(a_1+ad_1)       ω_1/2π = 3.22489       zp_1 = 2.49e+00\nmode 2:       harmonic       φ_2 = zp_2(a_2+ad_2)       ω_2/2π = 0.39497       zp_2 = 8.72e-01\nmode 3:       charge         ng_3 = 0\n------------------------------------------------------------\nparameters:       EC_33 = 0.296       EJ_1 = 5.0       EJ_2 = 5.0       \nloops:            φ_e1/2π = 0.5       '

    txt = cr.description(tp='ltx', _test=True)

    assert txt == '\\hat{H} =~\\omega_1\\hat a^\\dagger_1\\hat a_1~+~\\omega_2\\hat a^\\dagger_2\\hat a_2~+~E_{C_{33}}(\\hat{n}_3-n_{g_{3}})^2~+~E_{J_{1}}\\cos(-\\hat{\\varphi}_1+\\hat{\\varphi}_3+0.5\\varphi_{\\text{ext}_{1}})~+~E_{J_{2}}\\cos(-\\hat{\\varphi}_1-\\hat{\\varphi}_3+0.5\\varphi_{\\text{ext}_{1}})~\n------------------------------------------------------------\n\\text{mode}~1:~~~~~~~~~~~\\text{harmonic}~~~~~~~~~~~\\hat{\\varphi}_1~=~\\varphi_{zp_{1}}(\\hat a_1+\\hat a^\\dagger_1)~~~~~~~~~~~\\omega_1/2\\pi~=~3.22489~~~~~~~~~~~\\varphi_{zp_{1}}~=~2.49e+00\n\\text{mode}~2:~~~~~~~~~~~\\text{harmonic}~~~~~~~~~~~\\hat{\\varphi}_2~=~\\varphi_{zp_{2}}(\\hat a_2+\\hat a^\\dagger_2)~~~~~~~~~~~\\omega_2/2\\pi~=~0.39497~~~~~~~~~~~\\varphi_{zp_{2}}~=~8.72e-01\n\\text{mode}~3:~~~~~~~~~~~\\text{charge}~~~~~~~~~~~~~~~~n_{g_{3}}~=~0\n------------------------------------------------------------\n\\text{parameters}:~~~~~~~~~~~E_{C_{33}}~=~0.296~~~~~~~~~~~E_{J_{1}}~=~5.0~~~~~~~~~~~E_{J_{2}}~=~5.0~~~~~~~~~~~\n\\text{loops}:~~~~~~~~~~~~~~~~~~~~\\varphi_{\\text{ext}_{1}}/2\\pi~=~0.5~~~~~~~~~~~'