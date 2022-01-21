from circuit import *
import matplotlib.pyplot as plt

# E_C = hbar * 2 * np.pi * 0.15 * GHz
# E_CJ = hbar * 2 * np.pi * 10 * GHz
# E_J = hbar * 2 * np.pi * 5 * GHz
# E_L = hbar * 2 * np.pi * 0.13 * GHz
# C = e ** 2 / 2 / E_C
# C_J = e ** 2 / 2 / E_CJ
# L = (Phi0 / 2 / np.pi) ** 2 / E_L
#
# circuitParam = {(0, 1): {"C": C_J, "JJ": E_J / hbar},
#                 (0, 2): {"L": L},
#                 (0, 3): {"C": C},
#                 (1, 2): {"C": C},
#                 (1, 3): {"L": L},
#                 (2, 3): {"C": C_J, "JJ": E_J / hbar}}

C = Capacitor(0.15, "GHz")
CJ = Capacitor(10, "GHz")
JJ = Junction(5, "GHz")
L = Inductor(2*0.13, "GHz")

# circuitParam = {(0, 1): [CJ, JJ],
#                 (0, 2): [L],
#                 (0, 3): [C],
#                 (1, 2): [C],
#                 (1, 3): [L],
#                 (2, 3): [CJ, JJ]}
#
#
# # cr is an object of Qcircuit
# cr1 = Circuit(circuitParam)
#
# cr1.setTruncationNumbers([25, 1, 25])
# numEig = 5
# phiExt = np.linspace(0, 1, 100) * 2 * np.pi
# eigenValues = np.zeros((5, len(phiExt)))
# for i in range(len(phiExt)):
#     cr1.setExternalFluxes({(0, 1): phiExt[i]})
#     eigenValues[:, i], _ = cr1.run(numEig)
#
# plt.figure()
# for i in range(5):
#     plt.plot(phiExt / 2 / np.pi, (eigenValues[i, :] - eigenValues[0, :]).real / GHz / 2 / np.pi)
#
# plt.xlabel(r"$\Phi_{ext}/\Phi_0$")
# plt.ylabel(r"($\omega_i-\omega_0$)GHz")
# plt.show()


circuitParam = {(0, 1): [CJ, JJ, L ]}

# cr is an object of Qcircuit
cr1 = Circuit(circuitParam)
cr1.setTruncationNumbers([100])
cr1.run(numEig=5)
phi = np.pi * np.linspace(-1, 1, 300)
state = cr1.eigVecPhaseSpace(1, [phi])
plt.plot(phi, np.abs(state) ** 2)
plt.show()
print(np.sum(np.abs(state) ** 2)*np.diff(phi)[0])

# c1 = Capacitor(0.15, "GHz")
# print(c1.energy)
#
# l1 = Inductor(1.2573952415, "uF")
# print(l1.energy)
