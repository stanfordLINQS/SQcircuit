import SQcircuit as sq
import numpy as np
import matplotlib.pyplot as plt

C = sq.Capacitor(0.15, "GHz", error=10)
CJ = sq.Capacitor(10, "GHz", error=10)
JJ = sq.Junction(5, "GHz", error=10)
L = sq.Inductor(0.13, "GHz", error=10)

# C = sq.Capacitor(0.15, "GHz", error=10)
# print(C.value()/sq.unit.faradList['pF'])
# cList = []
# for i in range(100000):
#     cList.append(C.value(random=True)/sq.unit.faradList['pF'])
#
# plt.hist(cList, 30)
# plt.show()

circuitElements = {(0, 1): [CJ, JJ],
                   (0, 2): [L],
                   (0, 3): [C],
                   (1, 2): [C],
                   (1, 3): [L],
                   (2, 3): [CJ, JJ]}


cr1 = sq.Circuit(circuitElements)
cr1.setTruncationNumbers([25, 1, 25])
#
# omegaList = []
#
# for i in range(10000):
#     cr1 = sq.Circuit(circuitElements, random=True)
#     omegaList.append(cr1.omega[0].real/sq.unit.freqList['GHz'])
#
# plt.hist(omegaList, 30, color="orange", edgecolor='black', density=True)
# plt.show()

numEig = 5
phiExt = np.linspace(0, 1, 100) * 2 * np.pi
eigenValues = np.zeros((numEig, len(phiExt)))

for i in range(len(phiExt)):
    cr1.linkFluxes({(1, 3): sq.Flux(phiExt[i])})
    eigenValues[:, i], _ = cr1.run(numEig)

plt.figure()
for i in range(5):
    plt.plot(phiExt / 2 / np.pi, (eigenValues[i, :] - eigenValues[0, :]))

plt.xlabel(r"$\Phi_{ext}/\Phi_0$")
plt.ylabel(r"($\omega_i-\omega_0$)GHz")
plt.show()

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
