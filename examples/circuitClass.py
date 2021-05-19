# Liberaries:
import numpy as np
from PhysicsConstants import *
import matplotlib.pyplot as plt
import qutip as q

# XXXXXXXXXXXX
import pickle
import os
import time
import scipy.special
import scipy.sparse
import copy
import collections
import itertools


#class for double ring resonators(DRR)

class Qcircuit:

    def __init__(self,circuitParam):

        # circuitParam is a dictionary that contains information about
        # the circuit element and its topology.
        # self.circuitParam = copy.deepcopy(circuitParam)
        self.circuitParam = collections.defaultdict(lambda: [],copy.deepcopy(circuitParam))

        # graph structure
        self.graph = self.circuitParam['graph'];

        # incuctor list 
        self.L  =  self.circuitParam['inductors'];

        # capacitor list
        self.C = self.circuitParam['capacitors'];

        # Josephson junction list
        self.JJ = self.circuitParam['JJs'].copy();

        # number of nodes
        self.n = max(max(self.graph));

        self.reset()

    def reset(self):
        # this function reset the memmory of the circuit not the inputs of 
        # of the class

        # dielectric tangent loss
        self.dieTanList = [0 for i in range(self.n)]
        # quasiparticle fraction list
        self.x_qpList = [0 for i in range(self.n)]
        # tempreture of the circuit
        self.T = None;

        # truncation number of fock state
        # m should be an odd number because of charge operator
        self.m = np.zeros(self.n,dtype='int')
        for i in range(self.n):
            self.m[i] = 21;

        #number of edges
        self.E = len(self.graph);

        # list of loop and edges that can get connected to external fluxes
        self.externalList =[] 

        self.cInv = []
        self.cInvRotated =[]
        self.lRotated = []
        self.S = []
        self.R = []
        self.omega = []
        self.JJEj = []
        self.JJIndex = []
        self.JJEq = []
        self.JJEqRotated = []
        self.JJExcite = []
        self.JJExt=[]
        self.phiExt = []
        self.HamilEigVal = 0;
        self.HamilEigVecList = [];

        # list of charge operators(transformed operators)
        self.chargeOpList = [];

        ''' list of charge operator multiplyed by the other charge operator. 
        For expample, self.chargeByChargeList[0,2] will give you Q1 * Q3 
        Note: chargeBycharge has triangle. For example, for a circuit with 3 node 
        First list has 3 element, second list has 2 element, and third list has 1 element.
        '''
        self.chargeByChargeList = [];

        # number operators list
        self.numOpList = [];

        # LC Hamiltonian Operator
        self.HLC = 0;

        # JJ Hamiltonian befor adding the external flux
        self.HJJprevList = [];

        # exponentail part of sin(phi/2) for each JJ to calculate the quasiparticle loss
        self.qpSinPrevList = [];

        # sin(phi/2) for each external flux and JJ
        self.qpSinList =[]
        

    def setCapacitors(self,cList):
        self.C = cList.copy()
        self.circuitParam['capacitors'] = self.C

    def setInductors(self,lList):
        self.L = lList.copy() 
        self.circuitParam['inductors'] = self.L

    def setJJs(self,JJList):
        self.JJ = JJList.copy()
        self.circuitParam['JJs'] = self.JJ

    def cleanUpLC(self):

        # First we check if we have parallel C(as a list), and we put the net capacitor instead of capcitor list
        for i in range(len(self.C)):
            if(isinstance(self.C[i],list)):
                self.C[i] = sum(self.C[i]);

        # first we check if we have parallel L(as a list), and we put the net incuctor instead of inductor list
        for i in range(len(self.L)):
            if(isinstance(self.L[i],list)):
                invL = 0;
                for j in range(len(self.L[i])):
                    invL += 1/self.L[i][j];
                self.L[i] = 1/invL 

        # multiply each E_JJ with 2pi to write it in the units of angular frequency instead of Hz
        for i in range(len(self.JJ)):
            if(isinstance(self.JJ[i],list)):
                for j in range(len(self.JJ[i])):
                    self.JJ[i][j] = 2*np.pi*self.JJ[i][j]
            elif(self.JJ[i]!= None):
                self.JJ[i] = self.JJ[i]*2*np.pi

    def correctC(self,A):
        # In case of singularity we change the flux and charge operator to take out the coefficeint in the cosine
        # and if this is not possinle we send an error that the circuit is not possible to be solved with this 
        # solver

        for j in range(self.n):
            if(self.omega[j] == 0):
                s = np.max(np.abs(A[:,j]))
                if(s==0):
                    continue;
                for i in range(len(A[:,j])):
                    # check if abs(A[i,j]/s is etither zero or one with 1e-11 accuracy
                    if(abs(A[i,j]/s)>=1e-11 and abs(abs(A[i,j]/s)-1)>=1e-11):
                        raise ValueError("This Solver Cannot solve your circuits")
                    if(abs(A[i,j]/s)<=1e-11):
                        A[i,j] = 0

                # correncting the cInvRotated values
                # print(s)
                for i in range(self.n):
                    if(i == j):
                        self.cInvRotated[i,j] = self.cInvRotated[i,j]* s**2;
                    else:
                        self.cInvRotated[i,j] = self.cInvRotated[i,j] * s;


    def getMatC(self):
        # function that gives matrix representation of the a
        # a is vector of C

        # aMat matrix representation of a
        cMat = np.zeros((self.n,self.n));
        
        for i0 in range(len(self.graph)):
            i1 , i2 = self.graph[i0];
            if(i1 != 0 and i2 == 0):
                cMat[i1-1,i1-1] += self.C[i0];
            elif(i1 == 0 and i2 != 0):
                cMat[i2-1,i2-1] +=self.C[i0];
            else:
                cMat[i1-1,i2-1] = - self.C[i0];
                cMat[i2-1,i1-1] = - self.C[i0];
                cMat[i1-1,i1-1] += self.C[i0];
                cMat[i2-1,i2-1] += self.C[i0];

        return cMat


    def getMatL(self):
        # function that gives matrix representation of the a
        # a is vector of C
        
        # aMat matrix representation of a
        lMat = np.zeros((self.n,self.n));
        
        for i0 in range(len(self.graph)):
            i1 , i2 = self.graph[i0];    
            
            if(self.L[i0] == None):
                x = 0
            else:
                x = 1/self.L[i0]

            if(i1 != 0 and i2 == 0):
                lMat[i1-1,i1-1] += x;
            elif(i1 == 0 and i2 != 0):
                lMat[i2-1,i2-1] += x;
            else:
                lMat[i1-1,i2-1] = -x;
                lMat[i2-1,i1-1] = -x;
                lMat[i1-1,i1-1] += x;
                lMat[i2-1,i2-1] += x;
        
        return lMat


    def buildDiag(self):

        from scipy.linalg import sqrtm

        cMat = self.getMatC();
        lMat = self.getMatL();
        cMatInv = np.linalg.inv(cMat);
        self.cInv = cMatInv

        cMatRoot = sqrtm(cMat)
        cMatRootInv = np.linalg.inv(cMatRoot)
        lMatRoot = sqrtm(lMat)

        V , D, U = np.linalg.svd(lMatRoot@cMatRootInv)

        # singularity location in D:
        singLoc =[]
        for i in range(self.n):
            if(np.max(D)==0):
                # the case that there is not any inductor in the circuit
                D = np.ones(len(D))
                continue;
            elif(D[i]/np.max(D)<1e-6):
                singLoc.append(i)

        D[singLoc] = np.max(D)

        S = cMatRootInv@U.T@np.diag(np.sqrt(D))
        R = (np.linalg.inv(S).T)

        cInvRotated = R.T@cMatInv@R
        lRotated = S.T@lMat@S

        lRotated[singLoc,singLoc] = 0

        self.cInvRotCopy = cInvRotated.copy()

        return lRotated, cInvRotated , S, R


    def findAllCycles(self,graph):
        cycles = [];
        for edge in graph:
            for node in edge:
                self.findNewCycles([node],cycles,graph)
        return cycles

    def findNewCycles(self,path,cycles,graph):
        start_node = path[0]
        next_node= None
        sub = []

        #visit each edge and each node of each edge
        for edge in graph:
            node1, node2 = edge
            if start_node in edge:
                    if node1 == start_node:
                        next_node = node2
                    else:
                        next_node = node1
                    if not self.visited(next_node, path):
                            # neighbor node not on path yet
                            sub = [next_node]
                            sub.extend(path)
                            # explore extended path
                            self.findNewCycles(sub,cycles,graph);
                    elif len(path) > 2  and next_node == path[-1]:
                            # cycle found
                            p = self.rotate_to_smallest(path);
                            inv = self.invert(p)
                            if self.isNew(p,cycles) and self.isNew(inv,cycles):
                                cycles.append(p)

    def invert(self,path):
        return self.rotate_to_smallest(path[::-1])

    #  rotate cycle path such that it begins with the smallest node
    def rotate_to_smallest(self,path):
        n = path.index(min(path))
        return path[n:]+path[:n]

    def isNew(self,path,cycles):
        return not path in cycles

    def visited(self,node, path):
        return node in path

    def giveSimplestCycles(self,cycles,graph):
        # function that finds simplest cycles form all graph cycles.
        
        simplestCycles =[];
        for cy in cycles:
            
            # flag to check if the cycle is simple or not 
            simple = True;
            cyLen = len(cy);
            # if the cycle length is equal to 3 then it is definitely simple
            if(cyLen>3):
                #check all the nodes in the cycle to find a link inside the loop to detect 
                # whether it is simple or not.
                for i1 in range(cyLen):
                    for i2 in range(cyLen):
                        # make sure that we are not counting edges of the loop as a link
                        if(i1 != i2 and abs(i1-i2)>1 and abs(i1 - i2)!=cyLen-1):
                            if ([cy[i1],cy[i2]] or [cy[i2],cy[i1]]) in graph:
                                simple = False;
            if(simple):
                simplestCycles.append(cy)
            
        return simplestCycles

    def getCycleList(self,cycles):
        # This function gets a list of cycles and convert each cylce to list of edges

        cyclesList = []
        for cycle in cycles:
            cycleNew = []
            for i in range(len(cycle)):
                edge = [cycle[i-1],cycle[i]]
                edgeSwap = [cycle[i],cycle[i-1]]
                if edge in self.graph:
                    cycleNew.append(edge)
                else:
                    cycleNew.append(edgeSwap)
            cyclesList.append(cycleNew)

        return cyclesList


    def checkGraph(self):
        # this function checks if the graph is a valid circuit graph

        # # first condition is that the graph has all the nodes 
        # graphNumNode = len(np.unique(np.array(self.graph).reshape(2*len(self.graph))))
        # firstCond = numNode == graphNumNode

        # find all cycles of the graph 
        allCycles = self.findAllCycles(self.graph)
        allCyclesList = self.getCycleList(allCycles)

        # find all edges connected to each node
        neighborList = [[edge for edge in self.graph if i in edge] for i in range(self.n+1)]

        # second condtion is that the each edge of the graph should be at list in one cycle
        secondCond = True
        for edge in self.graph:
            if (sum([edge in cycle for cycle in allCyclesList]) == 0):
                secondCond = False
                break

        # third condtion checks for the following graph is not acceptable
        # imaging two squar that are connected with only one node
        thirdCond = True
        for neighbor in neighborList:
            for eachTwoEdge in itertools.combinations(neighbor,2):
                if(sum([(eachTwoEdge[0] in cycle and eachTwoEdge[1] in cycle) for cycle in allCyclesList]) == 0):
                    thirdCond = False
                    break

        return secondCond and thirdCond

    def getExternalLinks(self):

        return self.externalList


    def getJJEq(self):
        # This function gives:
        # JJEj -- list of JJ energies
        # JJEq -- list of JJ equations which specifies the displacement operators
        # JJexcite -- list of JJ that are connected to external fluxes
        # JJIndex -- list of JJ indeces in self.JJ

        JJEj = [];
        JJEq = [];
        JJExcite = [];
        JJIndex = [];

        # List of all JJ edges
        allJJ = []
        # List of JJ edges that are not parallel with inductors
        JJGraphNotInduc = []
        # List of JJ edges that are parallel with inductors
        JJGraphParalInduc = []
        # the graph that build with inductive elements
        fluxGraph = []
        for i in range(len(self.graph)):
            if(self.JJ[i] != None):
                allJJ.append(self.graph[i])
            if(self.JJ[i] != None and self.L[i] == None):
                JJGraphNotInduc.append(self.graph[i])

            if(self.JJ[i] != None and self.L[i] != None):
                JJGraphParalInduc.append(self.graph[i])

            if(self.JJ[i] != None or self.L[i] != None):
                fluxGraph.append(self.graph[i])

        # list of simplest cycles made by inductive elements
        cycles = self.findAllCycles(fluxGraph);
        simplest = self.giveSimplestCycles(cycles,fluxGraph)
        simplestCycles = self.getCycleList(simplest);


        # loops and edges that we can apply external flux
        externalList = []
        externalList += simplest


        # for each simplest Cycles we pick one JJ
        JJPicked = []
        JJPickedParalJJ = []
        for cycle in simplestCycles:
            for edge in cycle:
                if edge in JJGraphNotInduc:
                    JJPicked.append(edge)
                    # find the edges that has JJs in parralel
                    if isinstance(self.JJ[self.graph.index(edge)],list):
                        JJPickedParalJJ.append(edge)
                        externalList += (len(self.JJ[self.graph.index(edge)])-1)*[edge]
                    break;

        # JJ that are not in either 
        JJNotPicked = [edge for edge in allJJ if edge not in JJPicked and edge not in JJGraphParalInduc]

        # writing the loop equation to find how each external flux is related to each JJ
        JJSeen = []
        O = np.zeros((len(simplestCycles),len(simplestCycles)))
        for cycle in simplestCycles:
            for edge in cycle:
                if edge in JJPicked:
                    if edge not in JJSeen:
                        O[simplestCycles.index(cycle),JJPicked.index(edge)] += 1
                        JJSeen.append(edge)
                    else:
                        O[simplestCycles.index(cycle),JJPicked.index(edge)] -= 1

        # each row of Oinv shows how each JJ in JJPicked is related to each index in self.externalList
        Oinv = np.abs(np.linalg.inv(O))


        # First, we find the edges that we stored them in JJPicked(JJs in the loops with more than one edge)
        for i,edge in enumerate(JJPicked):
            JJIndex.append(self.graph.index(edge)) 
            JJ = self.JJ[JJIndex[-1]]
            JJEj.append(JJ)
            excite = np.where(Oinv[i,:]==1)[0].tolist()
            if edge not in JJPickedParalJJ:
                JJExcite.append(excite)
            else:
                excite = [excite]
                strartIndex = externalList.index(edge)
                excite.append(list(range(strartIndex,strartIndex+len(JJ)-1)))
                JJExcite.append(excite)

        # Second, we find the JJs that are in parralel with and inductor. Therefore, we can apply an external 
        # flux between them.
        for edge in JJGraphParalInduc:
            JJIndex.append(self.graph.index(edge)) 
            JJ = self.JJ[JJIndex[-1]]
            JJEj.append(JJ)
            if isinstance(JJ,list):
                externalList += len(JJ)*[edge]
                strartIndex = externalList.index(edge)
                excite =[0,list(range(strartIndex,strartIndex+len(JJ)))]
                JJExcite.append(excite)
            else:
                externalList += [edge]
                strartIndex = externalList.index(edge)
                JJExcite.append([strartIndex])
        # Third, we find the JJs that are not linked to a loop or are not parallel with Inductor. However, 
        # at these edges we can have JJS that are in parallel, which means we can apply extrernal flux to that 
        # edges as well.
        for edge in JJNotPicked: 
            JJIndex.append(self.graph.index(edge)) 
            JJ = self.JJ[JJIndex[-1]]
            JJEj.append(JJ)
            if isinstance(JJ,list):
                externalList += (len(JJ)-1)*[edge]
                strartIndex = externalList.index(edge)
                excite = [0,list(range(strartIndex,strartIndex+(len(JJ)-1)))]
                JJExcite.append(excite)
            else:
                JJExcite.append(0)

        # We find the linear combinaion of the mode fluxes inside each JJs cosine.
        for index in JJIndex:
            Eq = [0 for i in range(self.n+1)];
            i1 , i2 = self.graph[index];
            if(i1==0 or i2 ==0):
                Eq[i1 + i2]+=1;
            else:
                Eq[i1]+=1;
                Eq[i2]-=1;

            JJEq.append(Eq[1:])

        self.externalList = externalList

        return JJEj , JJEq , JJExcite , JJIndex


    def configure(self):
        # This function connects all the above functions and find the needed coefficient to describe the Hamiltonian
        
        # 200226 Last Update: In this part I assumed that we can diogonalize the LC part of the Hamiltonian.

        self.cleanUpLC();

        self.lRotated, self.cInvRotated , self.S, self.R = self.buildDiag()

        # vector of frequencies
        self.omega = np.sqrt(np.diag(self.cInvRotated)*np.diag(self.lRotated))

        # ceofficient needed to write JJ Hamiltonian
        self.JJEj , self.JJEq , self.JJExcite, self.JJIndex = self.getJJEq()

        # JJEq in rotated frame
        self.JJEqRotated = np.array(self.JJEq) @ self.S

        # check if we can use this algorithm to solve the circuit and change c inorder to handel
        # flux coordinates inside the cosine in case of singularity
        self.correctC(self.JJEqRotated)

        # we pre-process quantum oparators that we need to solve the circuit
        self.buildOpMemory()
        self.buildLCHamil()
        self.buildHJJprev()

    def setExcitation(self,external):
        self.JJExt = external;

    def setPhi(self):
        self.phiExt = self.JJExt;

        #reset the phiExt:
        # self.phiExt = [];

        # for el in self.JJExt:
        #     _ , phi = el;
        #     self.phiExt.append(phi); 

    def indExtParall(self,edgeParall):
        for i in range(len(self.JJExt)):
            edge, _  = self.JJExt[i];
            if(edge == edgeParall):
                return i ;

    def setModeNumbers(self,modeNum):
        # set the truncation number for the fock state

        assert len(modeNum) == self.n, "You should specify truncation number for all modes as a list"

        for i in range(self.n):
            if(isinstance(modeNum,list)):
                self.m[i] = modeNum[i];
            else:
                self.m[i] = modeNum

    def buildOpMemory(self):
        # this function will build and store the charge operator, number operator, and mulitplication of charge with charge for each node.

        # list of charge operators in their own mode basis(needed to be tensor producted with other modes)
        QList = []
        for i in range(self.n):
            if(self.omega[i] == 0):
                Q0 = (2*e/np.sqrt(hbar))*q.charge((self.m[i]-1)/2)
            else:
                coef = -1j*np.sqrt(1/2*np.sqrt(self.lRotated[i,i]/self.cInvRotated[i,i]))
                Q0 = coef*(q.destroy(self.m[i]) - q.create(self.m[i]))
            QList.append(Q0)

        # list of number operators in their own mode basis(needed to be tensor producted with other modes)
        nList = []
        for i in range(self.n):
            if(self.omega[i] == 0):
                num0 = q.charge((self.m[i]-1)/2)
            else:
                num0 = q.num(self.m[i])
            nList.append(num0)

        for i in range(self.n):
            chargeRowList = []
            for j in range(self.n):

                # we find the appropriate charge and number operator for first mode 
                if(j == 0 and i==0):
                    Q2 = QList[j] * QList[j];
                    Q = QList[j];
                    num = nList[j];

                    # we tensor product the charge with I for other modes
                    for k in range(self.n-1):
                        Q2 = q.tensor(Q2,q.qeye(self.m[k+1]))
                    chargeRowList.append(Q2)

                elif(j == 0 and i!=0):
                    I = q.qeye(self.m[j])
                    Q = I;
                    num = I

                # now we find the rest of the modes
                elif(j !=0 and j<i):
                    I = q.qeye(self.m[j])
                    Q = q.tensor(Q,I)
                    num = q.tensor(num,I)

                elif(j != 0 and j == i):
                    Q2 = q.tensor(Q,QList[j]*QList[j])
                    Q = q.tensor(Q,QList[j])
                    num = q.tensor(num,nList[j])

                    # we tensor product the charge with I for other modes
                    for k in range(self.n-j-1):
                        Q2 = q.tensor(Q2,q.qeye(self.m[k+j+1]))
                    chargeRowList.append(Q2)

                elif(j>i):
                    QQ = q.tensor(Q,QList[j])

                    # we tensor product the QQ with I for other modes
                    for k in range(self.n-j-1):
                        QQ = q.tensor(QQ,q.qeye(self.m[k+j+1]))
                    chargeRowList.append(QQ)

                    I = q.qeye(self.m[j])
                    Q = q.tensor(Q,I)
                    num = q.tensor(num,I)


            self.chargeOpList.append(Q) 
            self.chargeByChargeList.append(chargeRowList)
            self.numOpList.append(num)


    def buildLCHamil(self):
        # function that gives the Hamiltionian of the LC part of the circuits

        HLC = 0;

        for i in range(self.n):
            # we write j in this form because of self.chargebycharge shape
            for j in range(self.n-i):
                if(j==0):
                    if(self.omega[i]==0):
                        HLC += 1/2* self.cInvRotated[i,i]* self.chargeByChargeList[i][j]
                    else:
                        HLC += self.omega[i]*self.numOpList[i]
                elif(j>0):
                    if(self.cInvRotated[i,i+j] != 0):
                        HLC += self.cInvRotated[i,i+j]* self.chargeByChargeList[i][j]

        self.HLC = HLC


    def chargeDisp(self,num):

        d = np.zeros((num,num))
        for i in range(num):
            for j in range(num):
                if(j-1==i):
                    d[i,j] = 1;
        d = q.Qobj(d);

        d = d.dag()

        return d

    def buildHJJprev(self):
        H=0;
        # this for calculating sin(phi/2) operator for quasiparticle loss decay rate
        H2 = 0;

        for i in range(len(self.JJEj)):

            # tensor multiplication of displacement operator for JJ Hamiltonian
            for j in range(self.n):
                if(j == 0 and self.omega[j]==0):
                    if(self.JJEqRotated[i,j] == 0):
                        I = q.qeye(self.m[j])
                        H = I;
                        H2 = I;
                    elif(self.JJEqRotated[i,j] > 0):
                        d = self.chargeDisp(self.m[j])
                        H = d;
                    else:
                        d = self.chargeDisp(self.m[j])
                        H = d.dag();

                elif(j == 0 and self.omega[j]!=0):
                    alpha = 2*np.pi/Phi0*1j*np.sqrt(hbar/2*np.sqrt(self.cInvRotated[j,j]/self.lRotated[j,j]))\
                    *self.JJEqRotated[i,j]
                    H = q.displace(self.m[j],alpha)
                    H2 = q.displace(self.m[j],alpha/2)


                if(j!=0 and self.omega[j] == 0):
                    if(self.JJEqRotated[i,j] == 0):
                        I = q.qeye(self.m[j])
                        H = q.tensor(H,I);
                        H2 = q.tensor(H2,I)
                    elif(self.JJEqRotated[i,j] > 0):
                        d = self.chargeDisp(self.m[j])
                        H = q.tensor(H,d);
                    else:
                        d = self.chargeDisp(self.m[j])
                        H = q.tensor(H,d.dag());

                elif(j!=0 and self.omega[j] != 0):
                    alpha = 2*np.pi/Phi0*1j*np.sqrt(hbar/2*np.sqrt(self.cInvRotated[j,j]/self.lRotated[j,j]))\
                    *self.JJEqRotated[i,j]
                    H = q.tensor(H,q.displace(self.m[j],alpha))
                    H2 = q.tensor(H2,q.displace(self.m[j],alpha/2))

            self.HJJprevList.append(H)
            self.qpSinPrevList.append(H2)


    def getJJHamil(self,phiInput):
        # function that gives the Hamiltionian of the JJ of the circuits. with external charge of phi:

        # List of individual JJ Hamiltonian
        HJJList = [];
        HJJCheckList = [];

        # list of sin(phi/2) of each JJs
        H2SinList = []

        for i in range(len(self.JJEj)):

            # add external excitation

            # Parallel JJ case
            if(isinstance(self.JJEj[i],list)):
                JJExcIn, JJExcOut = self.JJExcite[i]

                phi = 0
                if(JJExcIn):
                    phi  += sum(phiInput[JJExcIn]);

                H = 0
                for j in range(len(self.JJEj[i])):
                    # we differ between the case that we have a parralel JJs with inductor or just parralel JJs
                    if(len(JJExcOut)==len(self.JJEj[i])):
                        phi += phiInput[JJExcOut[j]]
                        H += np.exp(1j*phi) * self.JJEj[i][j]/2* self.HJJprevList[i];
                    else:
                        if(j == 0):
                            H += np.exp(1j*phi) * self.JJEj[i][j]/2* self.HJJprevList[i];
                            continue;
                        phi += phiInput[JJExcOut[j-1]]
                        H += np.exp(1j*phi) * self.JJEj[i][j]/2* self.HJJprevList[i];
                H = H + H.dag();
                # needed to be implemented 
                H2 = 0

            # single JJ case
            else:
                phi = 0

                if(self.JJExcite[i]):
                    for ind in self.JJExcite[i]:
                        phi+= phiInput[ind];

                H  =  np.exp(1j*phi) * self.JJEj[i]/2* self.HJJprevList[i];
                H = H + H.dag();

                # sin(phi/2) for the quasiparticles decay rate
                H2 = np.exp(1j*phi/2) * self.qpSinPrevList[i];
                H2 = q.Qobj(H2)
                H2 = (H2.dag()-H2)/(2j);

            HJJList.append(H);
            H2SinList.append(H2)
        
        HJJ = sum(HJJList)

        return HJJ, H2SinList

    def run(self,numBand,showLoading=True):
        # function that use qutip package to define number operator and displacement operators to calculate
        # the hailtonian and diogonalize it. At the end,the final eignevalues of the
        # of the total Hamiltonian is stored in the self.HamilEig.

        # numBand is the number of band that we want.

        #prepare the external fluxes for calculating the JJ Hamiltonian
        self.setPhi()

        # caluclate the total dimension of circuit 
        netDimension = 1;
        for i in range(len(self.m)):
            netDimension *= self.m[i]

        # idex for sweeping over
        indSweep = 0;
        # flag shows that if we want sweep over a flux or not 
        indSweepFlag = False;
        for i in range(len(self.phiExt)):
            if(not isinstance(self.phiExt[i],int) and not isinstance(self.phiExt[i],float)):
                indSweep = i;
                indSweepFlag = True;

        phiInput = self.phiExt.copy()
        self.HamilEigVecList = []

        if(indSweepFlag):
            # self.HamilEigVal = np.zeros((netDimension,len(self.phiExt[indSweep])),dtype='complex');

            # the new method 
            self.HamilEigVal = np.zeros((numBand,len(self.phiExt[indSweep])),dtype='complex');
            # diogonalize Hamiltonain for sweeping over phiExt
            for i in range(len(self.phiExt[indSweep])):
                
                if(showLoading):
                    print(i)

                phiInput[indSweep] = self.phiExt[indSweep][i];

                HJJ, H2SinList = self.getJJHamil(phiInput)

                H = -HJJ + self.HLC
                self.qpSinList.append(H2SinList)


                # find the eigenvalues of the hamiltonian for each external phi
                # eigenValues , eigenVectors = H.eigenstates();

                # The new method
                eigenValues , eigenVectors = scipy.sparse.linalg.eigs(H.data,numBand,which='SR')
                eignevaluesSorted = np.sort(eigenValues.real)
                sortArg = np.argsort(eigenValues)
                eigenVectorsSorted = [q.Qobj(eigenVectors[:,ind],dims=[self.m.tolist(),len(self.m)*[1]])
                 for ind in sortArg]

                # store the eigenfunction of the Hamiltonian for each frequency
                # (we devided the eigenValues by 2*np.pi to express the eigenvalues in unit of frequency rather than angular frequency)
                self.HamilEigVal[:,i] = eignevaluesSorted

                # list of eigenvalues for each external phi
                self.HamilEigVecList.append(eigenVectorsSorted)
        else:
            self.HamilEigVal = np.zeros((numBand,1),dtype='complex');
            HJJ, H2SinList = self.getJJHamil(phiInput)

            H = -HJJ + self.HLC
            self.qpSinList.append(H2SinList)

            # find the eigenvalues of the hamiltonian for each external phi
            # eigenValues , eigenVectors = H.eigenstates();

            # the new mehtod 
            eigenValues , eigenVectors = scipy.sparse.linalg.eigs(H.data,numBand,which='SR')
            eignevaluesSorted = np.sort(eigenValues.real)
            sortArg = np.argsort(eigenValues)
            eigenVectorsSorted = [q.Qobj(eigenVectors[:,ind],dims=[self.m.tolist(),len(self.m)*[1]])
                 for ind in sortArg]

            # store the eigenfunction of the Hamiltonian for each frequency
            # (we devided the eigenValues by 2*np.pi to express the eigenvalues in unit of frequency rather than angular frequency)
            self.HamilEigVal[:,0] = eignevaluesSorted
            # list of eigenvalues for each external phi
            self.HamilEigVecList.append(eigenVectorsSorted)


    ############################################################################################

    def setDieTanLoss(self,DieTanList):
        self.dieTanList= DieTanList;

    def setQuasiparticleFraction(self,x_qpList):
        self.x_qpList=x_qpList;

    def setTemperature(self,T):
        self.T = T

    def decayRateProcess(self,state1,state2,mode='all'):
        """function that calculate the effective decay rates for 
        specific process of |state1> and |state2>"""

        # state2 should be larger than state1
        assert state2>state1, "State2 index should be larger than state1 index" 
        assert self.T != None, "Set the temperature first"


        decayList = []

        # loop over all the external fluxes
        for i1 in range(len(self.HamilEigVecList)):
            # list of effective decay rate for each capacitor
            dieDecayList = [];
            qpDecayList = [];

            ketState1= self.HamilEigVecList[i1][state1]
            ketState2= self.HamilEigVecList[i1][state2]

            omegaState1 = self.HamilEigVal[state1,i1]
            omegaState2 = self.HamilEigVal[state2,i1]

            omega_q = omegaState2 - omegaState1

            # the vector that holds the expectation values of charge operators
            q_ge = np.array([(ketState1.dag()*Qtilde*ketState2)[0,0] for Qtilde in self.chargeOpList])

            # the decayVec is the vector which makes calculation easier and faster more 
            # explanation in Qcircuit notes
            decayVec = self.cInv@self.R@q_ge

            if(mode == 'dielectric' or mode == 'all'):

                k_B = 1.38e-23
                # effect of tempreture

                # prevent the exponential over flow(exp(709) is biggest number that numpy can calculate)
                if(hbar*(omega_q)/(k_B*self.T)>709):
                    nbar = 0
                else:
                    nbar = 1/(np.exp(hbar*(omega_q)/(k_B*self.T))-1)

                for i,c_x in enumerate(self.C):

                    # extracting the nodes
                    node1, node2 = self.graph[i]

                    # check if the node is connected to ground
                    if(node1 == 0):
                        dieDecayList.append(2 * c_x * self.dieTanList[i]*np.abs(decayVec[node2-1])**2*(2*nbar+1))
                    elif(node2 == 0):
                        dieDecayList.append(2 * c_x * self.dieTanList[i]*np.abs(decayVec[node1-1])**2*(2*nbar+1))

                    else:
                        dieDecayList.append(2 * c_x * self.dieTanList[i]*
                            np.abs(decayVec[node1-1]-decayVec[node2-1])**2*(2*nbar+1))

            if(mode == 'quasiparticles' or mode == 'all'):
                for i in range(len(self.JJEj)):
                    """https://www.researchgate.net/figure/SIS-parameters-for-cold-
                    and-room-temperature-evaporations-Number-of-samples-measured_tbl1_1896356
                    """
                    Delta = 250 * 1e-6 * 1.6e-19;

                    # opExpt = (ketState1.dag()*(self.qpPrevList[i].dag() - self.qpPrevList[i])/(2j)*ketState2)[0,0]
                    opExpt = (ketState1.dag()*self.qpSinList[i1][i]*ketState2)[0,0]+1e-23
                    S = self.x_qpList[self.JJIndex[i]]*8*hbar*self.JJEj[i]/np.pi/hbar*np.sqrt(2*Delta/hbar/omega_q)
                    qpDecayList.append(np.abs(opExpt)**2 *S)

            decayList.append(np.sum(dieDecayList)+np.sum(qpDecayList))

        return np.array(decayList).real


    def setDrive(self,node):
        # function that builds the Hamiltonian related to drive the special node of the circuit. 
        # The varibales are the same as the notation that I used in the notes.

        # charge drive vector
        qd = np.zeros((self.n,1));
        qd[node-1] = 1

        W_total = 0;
        weights = (qd.T@self.cInv@self.R)[0,:]

        # calculate the W
        for i in range(self.n):
            W = 0
            for j in range(self.n):
                if(j == 0 and j == i):
                    if(self.omega[j] == 0):
                        Q = 2*e/hbar*q.charge((self.m[j]-1)/2)
                        W = Q
                    else:
                        disMinCreat = q.destroy(self.m[j]) - q.create(self.m[j]);
                        coef = -1j*np.sqrt(1/2*np.sqrt(self.lRotated[j,j]/self.cInvRotated[j,j])/hbar)
                        W = coef*disMinCreat;

                elif(j == 0 and j != i):
                    I = q.qeye(self.m[j]) 
                    W = I

                if(j != 0 and j == i):
                    if(self.omega[j] == 0):
                        Q = 2*e/hbar*q.charge((self.m[j]-1)/2)
                        W = q.tensor(W,Q)
                    else:
                        disMinCreat = q.destroy(self.m[j]) - q.create(self.m[j]);
                        coef = -1j*np.sqrt(1/2*np.sqrt(self.lRotated[j,j]/self.cInvRotated[j,j])/hbar)
                        W = q.tensor(W,coef*disMinCreat);

                elif(j != 0 and j != i):
                    I = q.qeye(self.m[j]) 
                    W = q.tensor(w,I)

            W_total = weights[i] * W

        return W_total


    def plotEigFreq(self,numBand,numDegen = 0):
        # function that plots eigen frequency of the total Hamiltonian.
        # -- numDegen is the parameter to avoid the degenercy of the bands when we have eigenfrequency 
        # close to zero. For example if we have one resonance frequency of the circuit is zero. Therfore, 
        # numDegen is zero.
        # -- numBand is the number of bands that we want to plot.


        indSweep = 0;
        for i in range(len(self.phiExt)):
            if(not isinstance(self.phiExt[i],int)):
                indSweep = i;

        for i in range(numBand):
            plt.plot(self.phiExt[indSweep]/2/np.pi,self.HamilEig[i*(self.m**numDegen),:].real/GHz);

        plt.show()


    def eigVecPhaseSpace(self,eigInd,phiList):
        # This function gives the egin vectors in the phase space representation.
        # eigInd: Index of eigen vector that we try to represent in phase space
        # phiList: list of phases for all nodes

        # Eigen vector for the eigInd index
        eigVec = self.HamilEigVecList[0][eigInd]

        # caluclate the total dimension of circuit 
        netDimension = 1;
        for i in range(len(self.m)):
            netDimension *= self.m[i]

        state = 0;

        for i in range(netDimension):
            # here I assume that each mode has equal truncation number( I should change it lader)
            # index of each mode for each i 
            phiInd = self.getPhiIndex(i)

            # phiInd = list(base(i,10,self.m[0]))
            # phiInd = (self.n-(len(phiInd)))*[0]+phiInd

            term = self.HamilEigVecList[0][eigInd][i][0,0]

            for node in range(self.n):

                # mode number related to that node
                n = phiInd[node]

                if(self.omega[node] == 0):
                    term *= 1/np.sqrt(2*np.pi)*np.exp(1j*phiList[node]*n)
                    
                else:
                    x0 = np.sqrt(hbar*np.sqrt(self.cInvRotated[node,node]/self.lRotated[node,node]));
                    varphi0 = x0/Phi0
                    coef = 1/np.sqrt(np.sqrt(np.pi)*2**n*scipy.special.factorial(n)*x0)
                    term *= coef*np.exp(-(phiList[node]/varphi0)**2/2)*scipy.special.eval_hermite(n,phiList[node]/varphi0)

            state += term

        return state

    def getPhiIndex(self, index):
        # this function gives the decomposed mode indices from the tensor product space index.
        # For example index 5 of the tensor product space can be decomposed to [1,0,1] modes if
        # the truncation number for each mode is 2.

        # ith mP element is the multiplicatoin of the self.m elements until its ith element
        mP = []
        for i in range(self.n-1):
            if(i == 0):
                mP.append(self.m[-1])
            else:
                mP = [mP[0]*self.m[-1-i]] + mP

        indList = []
        indexP = index
        for i in range(self.n):
            if(i==self.n-1):
                indList.append(indexP)
                continue
            indList.append(int(indexP/mP[i]))
            indexP = indexP%mP[i]

        return indList    

    def getPotentialNode(self,node,phi,phiExt):
        # This function gives the potential related to speicific node as a function 
        # of phase.(I'm assuming that the phase related to other nodes is zero)

        potential = 1/2*self.lRotated[node-1,node-1]*(phi*Phi0)**2 +0j

        # for i, EJ in enumerate(self.JJEj):
        #     potential -= hbar * EJ * np.cos(2*np.pi*self.JJEqRotated[i,node-1]*phi)

        JJprevList = []

        for i, EJ in enumerate(self.JJEj):
            JJprevList.append(np.exp(1j*2*np.pi*self.JJEqRotated[i,node-1]*phi)) 

        # List of individual JJ Hamiltonian
        potenList = [];

        for i in range(len(self.JJEj)):

            # add external excitation

            # Parallel JJ case
            if(isinstance(self.JJEj[i],list)):
                JJEjIn , JJEjOut = self.JJEj[i]
                JJExcIn, JJExcOut = self.JJExcite[i]
                phiIn = 0;
                PhiOut = 0;
                if(JJExcIn):
                    for ind in JJExcIn:
                        phiIn = phiExt[ind];
                phiOut = phiIn + phiExt[JJExcOut];

                H = -hbar*np.exp(1j*phiIn)*JJEjIn/2*JJprevList[i] - hbar*np.exp(1j*phiOut)*JJEjOut/2*JJprevList[i]
                H = H + np.conj(H)

            # single JJ case
            else:
                phi = 0
                # fluxoniumSituation 
                if(len(self.graph)*len(self.L) == 1):
                    self.JJExcite = [[0]]

                if(self.JJExcite[i]):
                    for ind in self.JJExcite[i]:
                        phi+= phiExt[ind];

                H  =  -hbar*np.exp(1j*phi) * self.JJEj[i]/2* JJprevList[i];
                H = H + np.conj(H)

            potenList.append(H);
        
        potential += sum(potenList)


        return potential 


    def saveData(self,fileName='untitled'):
        # function to save data

        # see if data folder exist save. if not make one.
        if(not os.path.exists('data')):
            os.mkdir('data');

        # save the date and time to add to the name of file
        currentTime = time.localtime()

        currentTimeStr='';

        for i in range(5):
            if(currentTime[i]<10):
                currentTimeStr += '0' + str(currentTime[i]);
            elif(i==0):
                currentTimeStr += str(currentTime[i]-2000);
            else:
                currentTimeStr += str(currentTime[i]);
        currentTimeStr += '_'

        fileNameTime = currentTimeStr + fileName;

        # save the class object to data folder
        savePath = 'data/' + fileNameTime;

        with open(savePath , 'wb') as saveFile:
            pickle.dump(self,saveFile);

        print("File Saved")


#### help functionssss

def loadData(fileName):
	# function to load data

	loadPath = 'data/' + fileName;

	with open(loadPath , 'rb') as loadFile:
		obj = pickle.load(loadFile)

	return obj










