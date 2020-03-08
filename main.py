from circuitClass import *

# # First define the circuit graph
# graph = [[0,2],[0,3],[2,3],[1,2],[1,3]];
# # inductors
# L = [0,1,1,0,1];
# # capacitors
# C = [1,1,1,1,1];
# # Josephson junctions
# JJ = [1,0,0,1,0];

graph = [[0,1],[0,2],[0,3],[1,2],[2,3]]
L = [0,4.5*nH,0,15.6*nH,386*nH]
# I assumed Cx = 1nF 
C = [20.3*fH, 10*nF, 5.3*fF,0,0]
JJ = [0,0,6.2*GHz+0*1j,0,0]
phi = np.linspace(-0.5,0.5,30)*2*np.pi

cr1 = Qcircuit(graph,L,C,JJ,phi)

print(cr1.giveMatC())
print(cr1.giveMatL())


lRotated, cInvRotated , S = cr1.buildDiag()
print("lRotated:")
print(lRotated)
print("cInvRotated:")
print(cInvRotated)
print("S:")
print(S)

cycles = cr1.findAllCycles();
simplestCycles = cr1.giveSimplestCycles(cycles);


# printing out the cycles
for cy in simplestCycles:
    path = [str(node) for node in cy];
    s = "-".join(path);
    print(s)

spanningTrees = cr1.findSpanningTrees()
print("After Cleaning:")

for trees in spanningTrees:
    print(trees)

spanTree = cr1.giveMaxInd()
print("span tree with max inductor:")
print(spanTree)

tree = spanTree.copy();

treeWithDirec = cr1.treeDirec(0,None,[],tree)
print("direction corrected:")
print(treeWithDirec)

O, notInSpan = cr1.cycleEqu(simplestCycles,treeWithDirec) 
# O = np.array(O);
print("Equation without rotation:")
print(O)
print("Equation after rotation:")
print(O@S)
print("Edges that are not in the spanning Tree:")
print(notInSpan)



JJEj , JJEq , JJExcite = cr1.giveJJEq(O,notInSpan)

print("JJEj:")
print(JJEj)

print("JJEq:")
print(np.array(JJEq)@S)

print("JJExcite:")
print(JJExcite)

print("*********************************************************************************")

cr1.configure()

# cr1.solveCircuit()

# print(cr1.omega/GHz)

# cr1.plotEigFreq(3,1)

# print(cr1.HamilEig[:,1]/GHz)

cr1.saveData('circuit1')

cr2 = loadData('202003031657_circuit1')

print(cr2.graph)









