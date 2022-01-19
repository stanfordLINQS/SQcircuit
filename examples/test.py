from circuit import *
import matplotlib.pyplot as plt

circuitParam = {(0, 1): {"C": 200* pF, "JJ": 5*GHz}}

# cr is an object of Qcircuit
cr1 = SQcircuit(circuitParam)
cr1.setTruncationNumbers([100])
cr1.run(numEig=5)
phi = np.pi*np.linspace(-1, 1, 300)
state = cr1.eigVecPhaseSpace(0, [phi])
plt.plot(phi, np.abs(state))
plt.show()