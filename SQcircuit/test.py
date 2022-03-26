# import SQcircuit as sq
# import numpy as np
# import matplotlib.pyplot as plt

# test case 1:
# loop1 = sq.Loop()
#
# C = sq.Capacitor(0.15, "GHz", error=10)
# CJ = sq.Capacitor(10, "GHz", error=10)
# JJ1 = sq.Junction(5, "GHz", error=10, loops=[loop1], cap=CJ)
# JJ2 = sq.Junction(5, "GHz", error=10, loops=[loop1])
# L = sq.Inductor(0.13, "GHz", error=10, loops=[loop1])
#
# circuitElements = {(0, 1): [JJ2, CJ],
#                    (0, 2): [L],
#                    (0, 3): [C],
#                    (1, 2): [C],
#                    (1, 3): [L],
#                    (2, 3): [JJ1]}
#
# cr1 = sq.Circuit(circuitElements)
#
# cr1.setTruncationNumbers([25, 1, 25])
#
# print(cr1.K2)
#
# numEig = 5
# phiExt = np.linspace(0, 1, 100) * 2 * np.pi
# eigenValues = np.zeros((numEig, len(phiExt)))
#
# for i in range(len(phiExt)):
#     loop1.setFlux(phiExt[i])
#     eigenValues[:, i], _ = cr1.run(numEig)
#
# plt.figure()
# for i in range(5):
#     plt.plot(phiExt / 2 / np.pi, (eigenValues[i, :] - eigenValues[0, :]))
# plt.show()

# test case 2:

# loop1 = sq.Loop()
# loop2 = sq.Loop()
#
# C = sq.Capacitor(0.15, "GHz", error=10)
# JJ1 = sq.Junction(5, "GHz", error=10, loops=[loop1])
# JJ2 = sq.Junction(5, "GHz", error=10, loops=[loop2])
# L = sq.Inductor(0.13, "GHz", error=10, loops=[loop1, loop2])
#
# circuitElements = {(0, 1): [C, L, JJ1, JJ2]}
#
# cr1 = sq.Circuit(circuitElements)
#
# print(loop1.K1)
# print(loop2.K1)
#
# print(cr1.loops)


# test case 3:
# loop1 = sq.Loop()
# loop2 = sq.Loop()
#
# C = sq.Capacitor(1, "F", error=10)
# JJ1 = sq.Junction(5, "GHz", error=10, loops=[loop1], cap=C)
# JJ2 = sq.Junction(5, "GHz", error=10, loops=[loop2], cap=C)
# L = sq.Inductor(0.13, "GHz", error=10, loops=[loop1, loop2], cap=C)
#
# circuitElements = {(0, 1): [JJ1],
#                    (1, 2): [JJ1],
#                    (0, 2): [L],
#                    (2, 3): [JJ2],
#                    (0, 3): [JJ2]}
#
# cr1 = sq.Circuit(circuitElements)

# print(loop1.K1)
# print(loop2.K1)

# print(cr1.K1)
#
# a = np.zeros_like(loop1.K1)
# select = np.sum(loop1.K1 != a, axis=0) != 0
# print(select)
# print(np.array(loop1.K1)[:, select].T)

# p = loop1.getP()
# a = np.zeros((1, len(circuitElements)))
# a[0, loop1.indices] = p
# print(a)
#
# p = loop2.getP()
# a = np.zeros((1, len(circuitElements)))
# a[0, loop2.indices] = p
# print(a)
#
# print(cr1.K1.T)


#
# omegaList = []
#
# for i in range(10000):
#     cr1 = sq.Circuit(circuitElements, random=True)
#     omegaList.append(cr1.omega[0].real/sq.unit.freqList['GHz'])
#
# plt.hist(omegaList, 30, color="orange", edgecolor='black', density=True)
# plt.show()


# plt.xlabel(r"$\Phi_{ext}/\Phi_0$")
# plt.ylabel(r"($\omega_i-\omega_0$)GHz")
# plt.show()

# circuitParam = {(0, 1): [CJ, JJ, L]}

# cr is an object of Qcircuit
# cr1 = Circuit(circuitParam)
# cr1.setTruncationNumbers([100])
# cr1.run(numEig=5)
# phi = np.pi * np.linspace(-1, 1, 300)
# state = cr1.eigVecPhaseSpace(1, [phi])
# plt.plot(phi, np.abs(state) ** 2)
# plt.show()
# print(np.sum(np.abs(state) ** 2)*np.diff(phi)[0])

# c1 = Capacitor(0.15, "GHz")
# print(c1.energy)
#
# l1 = Inductor(1.2573952415, "uF")
# print(l1.energy)


# loop1 = sq.Loop()
#
# C_r = sq.Capacitor(20.3, "fF")
# L_r = sq.Inductor(15.6, "nH")
# C_q = sq.Capacitor(5.3, "fF")
# L_q = sq.Inductor(386, "nH", loops=[loop1])
# JJ = sq.Junction(6.2, "GHz", loops=[loop1])
# L_s = sq.Inductor(4.5, "nH", loops=[loop1])
# # L_q = sq.Inductor(386, "nH")
# # JJ = sq.Junction(6.2, "GHz")
# # L_s = sq.Inductor(4.5, "nH")
#
# circuitElements = {(0, 1): [C_r],
#                    (1, 2): [L_r],
#                    (0, 2): [L_s],
#                    (2, 3): [L_q],
#                    (0, 3): [JJ, C_q]}
#
# # cr is an object of Qcircuit
# cr1 = sq.Circuit(circuitElements)
#
# print(cr1.omega)
# print(cr1.cInvDiag)

# loop1 = sq.Loop()
# loop2 = sq.Loop()
#
# # import sympy
#
# CJ = sq.Capacitor(0.15, "GHz")
# JJ1 = sq.Junction(5, "GHz", loops=[loop1], cap=CJ)
# JJ2 = sq.Junction(5, "GHz", loops=[loop2], cap=CJ)
# L = sq.Inductor(0.13, "GHz", loops=[loop1, loop2])
#
# circuitElements = {(0, 1): [JJ1],
#                    (1, 2): [JJ1],
#                    (2, 3): [JJ1],
#                    (3, 5): [JJ2],
#                    (5, 4): [JJ2],
#                    (4, 0): [JJ2],
#                    (0, 3): [L]}
#
# cr1 = sq.Circuit(circuitElements)
#
# print(cr1.wTrans)


# loop1 = sq.Loop()
# loop2 = sq.Loop()
#
# # import sympy
#
# CJ = sq.Capacitor(0.15, "GHz")
# JJ1 = sq.Junction(5, "GHz", loops=[loop1], cap=CJ)
# JJ2 = sq.Junction(5, "GHz", loops=[loop2], cap=CJ)
# L = sq.Inductor(0.13, "GHz", loops=[loop1, loop2])
#
# circuitElements = {(0, 1): [JJ1],
#                    (1, 2): [JJ1],
#                    (2, 3): [JJ1],
#                    (3, 5): [JJ2],
#                    (5, 4): [JJ2],
#                    (4, 0): [JJ2],
#                    (0, 3): [L]}
#
# cr1 = sq.Circuit(circuitElements)
#
# print(cr1.wTrans)


# loop1 = sq.Loop()
# C = sq.Capacitor(11, 'fF', Q=1e6)
# Cg = sq.Capacitor(0.5, 'fF', Q=1e6)
# L = sq.Inductor(1, 'GHz', Q=500e6, loops=[loop1])
# JJ = sq.Junction(3, 'GHz', cap=C, A_c=5e-7, loops=[loop1])
# # JJ = sq.Junction(3, 'GHz', A_c=5e-7, loops=[loop1])
#
# circuitElements = {
#     (0, 1): [Cg],
#     (1, 2): [L, JJ],
#     (0, 2): [Cg]
# }
#
# cr1 = sq.Circuit(circuitElements)
#
# print(cr1.wTrans)

# cr1.setTruncationNumbers([30, 1])
# numEig = 5
# phiExt = np.linspace(0, 1, 100) * 2 * np.pi
# eigenValues = np.zeros((numEig, len(phiExt)))
# for i in range(len(phiExt)):
#     loop1.setFlux(phiExt[i])
#     eigenValues[:, i], _ = cr1.run(numEig)
#
# for i in range(numEig):
#     plt.plot(phiExt / 2 / np.pi, eigenValues[i, :] - eigenValues[0, :])
# plt.show()

# loop1 = sq.Loop()
#
# C = sq.Capacitor(0.15, "GHz")
# CJ = sq.Capacitor(10, "GHz")
# JJ = sq.Junction(5, "GHz", loops=[loop1])
# L = sq.Inductor(0.13, "GHz", loops=[loop1])
#
# circuitElements = {(0, 1): [CJ, JJ],
#                    (0, 2): [L],
#                    (0, 3): [C],
#                    (1, 2): [C],
#                    (1, 3): [L],
#                    (2, 3): [CJ, JJ]}
#
# # cr is an object of Qcircuit
# cr1 = sq.Circuit(circuitElements)
#
# print(cr1.wTrans)

# import the libraries
import os
os.chdir("../..")

print(os.getcwd())

# import src

from circuit import *
# from elements import *

import numpy as np
import matplotlib.pyplot as plt

# define the loop of the circuit
loop1 = sq.Loop()

# define the circuit's elements
C = sq.Capacitor(3.6, "GHz", Q=1e6)
L = sq.Inductor(0.46, "GHz",
                loops=[loop1])
JJ = sq.Junction(10.2, "GHz",
                 cap=C, A=5e-7,
                 loops=[loop1])

# define the Fluxonium circuit
elements = {(0, 1): [L, JJ]}
cr = sq.Circuit(elements)

# set the truncation numbers
cr.truncationNumbers([100])

# external flux for sweeping over
phi = np.linspace(0, 2 * np.pi, 300)

# T_1 and T_phi
T_1 = np.zeros_like(phi)
T_phi = np.zeros_like(phi)

for i in range(len(phi)):
    # set the external flux for the loop
    loop1.setFlux(phi)

    # diagonalize the circuit
    _, _ = cr.diag(numEig=2)

    # get the T_1 for capacitive loss
    T_1[i] = 1 / cr.decRate(
        decType="capacitive",
        states=(1, 0))

    # get the T_phi for cc noise
    T_phi[i] = 1 / cr.decRate(
        decType="cc",
        states=(1, 0))

# plot the T_1 from the capacitive loss
plt.semilogy(phi / 2 / np.pi, T_1)
plt.xlabel(r"$\Phi_{ext}/\Phi_0$")
plt.ylabel(r"$s$")

# plot the T_phi from the critical
# current noise
plt.semilogy(phi / 2 / np.pi, T_phi)
plt.xlabel(r"$\Phi_{ext}/\Phi_0$")
plt.ylabel(r"$s$")

