from circuitClass import *

# graph = [[0,2],[0,3],[2,3],[1,2],[1,3]];
# # inductors
# L = [0,1,1,0,1];
# # capacitors
# C = [1,1,1,1,1];
# # Josephson junctions
# JJ = [1,0,0,1,0];

# graph = [[0,1],[0,2],[0,3],[1,2],[2,3]]
# L = [0,4.5*nH,0,15.6*nH,386*nH]
# # I assumed Cx = 1nF 
# C = [20.3*fH, 10*nF, 5.3*fF,0,0]
# JJ = [0,0,6.2*GHz+0*1j,0,0]
# phi = np.linspace(-0.5,0.5,30)*2*np.pi


# graph = [[0,1],[0,2],[1,2],[1,3],[2,5],[2,6],[6,5],[3,5],[3,4],[4,5]]
# L = [1,0,0,1,0,1,0,0,1,0]
# C = [1,1,1,1,1,1,1,1,1,1]
# JJ = [0,1,1,0,[1,1],0,1,1,0,1]
# phi = np.linspace(0,0.5,15)*2*np.pi

# cr1 = Qcircuit(graph,L,C,JJ)


Ec = hbar*0.3*GHz 
Ej = hbar*15*GHz
Cx = e**2/2/Ec

graph = [[0,1]]
L = [0]
C = [Cx]
JJ = [[Ej/hbar , Ej/hbar]]

# define the circuit
cr1 = Qcircuit(graph,L,C,JJ)

phiExt = np.linspace(0,2,300)*2*np.pi
cr1.setExcitation([([0,1],phiExt)])

cr1.configure()

cr1.solveCircuit()

cr1.plotEigFreq(4)
# lRotated, cInvRotated , S = cr1.buildDiag()
# print("lRotated:")
# print(np.diag(lRotated))
# print("cInvRotated:")
# print(cInvRotated)
# print("S:")
# print(S)

# cr1.configure()

# print(cr1.cInvRotated)

# print(cr1.intOp(1,3))

# cr1.solveCircuit()

# print(cr1.omega/GHz)

# cr1.plotEigFreq(3,1)

# print(cr1.HamilEig[:,1]/GHz)

# cr1.saveData('circuit1')

# cr2 = loadData('202003031657_circuit1')

print("*********************************************************************************")