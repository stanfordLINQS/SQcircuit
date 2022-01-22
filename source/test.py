import SQcircuit as sq
import numpy as np
import matplotlib.pyplot as plt

C = sq.Capacitor(0.15, "GHz")
CJ = sq.Capacitor(10, "GHz")
JJ = sq.Junction(5, "GHz")
L = sq.Inductor(0.13, "GHz")

circuitParam = {(0, 1): [CJ, JJ],
                (0, 2): [L],
                (0, 3): [C],
                (1, 2): [C],
                (1, 3): [L],
                (2, 3): [CJ, JJ]}

# cr is an object of Qcircuit
cr1 = sq.Circuit(circuitParam)
cr1.setTruncationNumbers([25, 1, 25])
numEig = 5
phiExt = np.linspace(0, 1, 100) * 2 * np.pi
eigenValues = np.zeros((numEig, len(phiExt)))

for i in range(len(phiExt)):
    cr1.setExternalFluxes({(0, 1): phiExt[i]})
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
