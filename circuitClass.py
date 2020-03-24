# Liberaries:
import numpy as np
from PhysicsConstants import *
import matplotlib.pyplot as plt
import qutip as q

# XXXXXXXXXXXX
import pickle
import os
import time

# graph strutcture(important properties of the circuit graph)
class graphSt:

	simplestCycles = []
	spanningTree = []


#class for double ring resonators(DRR)

class Qcircuit:

    def __init__(self,graph,L,C,JJ):

        # graph structure
        self.graph = graph;

        # incuctor list 
        self.L  =  L;

        # capacitor list
        self.C = C;

        # Josephson junction list
        self.JJ = JJ;

        #number of nodes
        self.n = max(max(graph));

        # truncation number of fock state
        # m should be an odd number because of charge operator
        self.m = 11;

        #number of edges
        self.E = len(graph);

        #graph strutcture(important properties of the circuit graph)
        self.graphSt = graphSt();

        # memmory:
        self.cInvRotated =[]
        self.lRotateed = []
        self.omega = []
        self.JJEj = []
        self.JJEq = []
        self.JJEqRotated = []
        self.JJExcite = []
        self.JJExt=[]
        self.phiExt = []
        self.HamilEig = 0;


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

    def correctC(self,A):
        # In case of singularity we change the flux and charge operator to take out the coefficeint in the cosine
        # and if this is not possinle we send and error that this circuit is not possible to be solved with this 
        # solver

        for j in range(self.n):
            if(self.omega[j] == 0):
                s = np.max(np.abs(A[:,j]))
                for i in range(len(A[:,j])):
                    if(abs(A[i,j]/s)!=0 and abs(A[i,j]/s) !=1):
                        raise ValueError("This Solver Cannot solve your circuits")

                # correncting the cInvRotated values
                for i in range(self.n):
                    if(i == j):
                        self.cInvRotated[i,j] = self.cInvRotated[i,j]* s**2;
                    else:
                        self.cInvRotated[i,j] = self.cInvRotated[i,j] * s;


    def giveMatC(self):
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


    def giveMatL(self):
        # function that gives matrix representation of the a
        # a is vector of C
        
        # aMat matrix representation of a
        lMat = np.zeros((self.n,self.n));
        
        for i0 in range(len(self.graph)):
            i1 , i2 = self.graph[i0];    
            
            x = 1/self.L[i0] if self.L[i0]!=0 else 0;
            
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

    	# Function that diagonalize the LC part of the Hamiltonian.
        
        cMat = self.giveMatC();
        lMat = self.giveMatL();
        cMatInv = np.linalg.inv(cMat);
        
        # flag condition for L singularity: 
        lSing = False;
        
        # check if the lMat is singular:
        try: 
            lMatInv = np.linalg.inv(lMat)
        except:
            lSing = True;

        if(lSing):
            eigVal , S = np.linalg.eig(lMat);
            R = S;
            cInvRotated = R.T @ cMatInv @ R;
            lRotated = S.T @ lMat @ S;
            
        else:
            eigVal , P = np.linalg.eig(cMat);

            cMatRootDiag = np.diag(np.sqrt(eigVal))
            cMatRootInvDiag = np.diag(1/np.sqrt(eigVal))

            cMatRoot = P @ cMatRootDiag @ P.T
            cMatRootInv = P @ cMatRootInvDiag @ P.T

            # diogonalizing CLC to find Omega 
            CLC = cMatRootInv @ lMat @ cMatRootInv;
            eigVal , U = np.linalg.eig(CLC);

            Omega = np.diag(np.sqrt(eigVal));
            OmegaRoot = np.sqrt(Omega);

            # find the matrix SInv
            SInv = OmegaRoot @ U.T @ cMatRoot;
            S = np.linalg.inv(SInv);
            
            cInvRotated = Omega
            lRotated = Omega
        
        
        # check If The calculation is right by calculating R and seeing if R diogonalize the CInv.
            # print("S.T@ lMat @ S:")
            # print(S.T@ lMat @ S)
            
            # OmegaInvRoot = np.linalg.inv(OmegaRoot);
            
            # RInv = OmegaInvRoot @ U.T @ cMatRootInv;
            # R = np.linalg.inv(RInv);
            
            # print("R.T @ cMatInv @ R:")
            # print(R.T @ cMatInv @ R)

        return lRotated, cInvRotated , S


    def findAllCycles(self):
        cycles = [];
        for edge in self.graph:
            for node in edge:
                self.findNewCycles([node],cycles)
        return cycles

    def findNewCycles(self,path,cycles):
        start_node = path[0]
        next_node= None
        sub = []

        #visit each edge and each node of each edge
        for edge in self.graph:
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
                            self.findNewCycles(sub,cycles);
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

    def giveSimplestCycles(self,cycles):
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
                            if ([cy[i1],cy[i2]] or [cy[i2],cy[i1]]) in self.graph:
                                simple = False;
            if(simple):
                simplestCycles.append(cy)
            
        return simplestCycles

    def comb(self,combinations,sofar, rest, n):
        # function to find all 'n' combination of a list elements
        # combinations is list that saves the each detected combination
        # n is the number of combination
        if n == 0:
    #         print(sofar)
            combinations += [sofar];
        else:
            for i in range(len(rest)):
                self.comb(combinations,sofar + [rest[i]], rest[i+1:], n-1) 
        return combinations

    def checkIfTree(self,startNode, previousNode,visitList,tree):
    # recursive function to check if the input edge list is the spanning tree
    # if a node visited twice then we no that it is not a spanning tree
    
	    visitList[startNode] += 1
	    if 2 in visitList:
	        return False

	    # if startNode is not in tree its not definitely a spanning tree
    	# the flag that make sure that start node is in tree.
	    startNodeCondition = False;

	    for edge in tree:
	        if startNode in edge:
	            startNodeCondition = True;
	            for node in edge:
	                if( node != startNode and  node != previousNode):
	                    condition = self.checkIfTree(node,startNode,visitList,tree)
	                    if(condition == False):
	                        return False 
	    # has not visited any nodes twice or start node is not in the tree
	    if(startNodeCondition):
	        return True
	    else:
	        return False

    def findSpanningTrees(self):
    	# number of vertices
        V = self.n+1

        # allComb is all trees built by getting all V-1 combination of E.
        allComb = self.comb([],[],self.graph,V-1)

        spanningTrees = []

        # to remove trees contain loop
        for trees in allComb:
            # flag shows the condition of a tree whether it is spanning tree or not
            visitList = [0 for i in range(V)]
            # flag that shows the condition of a tree whether it is spanning tree or not
            fine = self.checkIfTree(0,None,visitList,trees);

            if(fine):
                spanningTrees.append(trees);

        return spanningTrees;


    def edgeListToVec(self,edgeList):
        # function that transforms a list of edges to the vector represantation of the graph
        vector = [0 for i in range(self.E)];
        
        for edge in edgeList:
            for i in range(self.E):
                if(self.graph[i] == edge or self.graph[i] == [edge[1]]+[edge[0]] ):
                    vector[i] = 1;
        return vector

    def spanTreeToVec(self,spanTrees):
        # function that transforms a spanning Tree List to the vector represantation of the graph
        
        # spanning trees vector representation.
        spanTreesVec=[]

        for edgeList in spanTrees:
            spanTreesVec.append(self.edgeListToVec(edgeList))
            
        return spanTreesVec
            
    def giveMaxInd(self):
        # function that gives the spanning tree with maximum inductor
        
        spanTrees = self.findSpanningTrees()
        
        spanTreesVec = self.spanTreeToVec(spanTrees)
        
        spanTreesVec=np.array(spanTreesVec)

        LVec = np.array(self.L);

        # finding a spanning tree with maximum correlation with inductor vector
        chosTree = np.argmax(np.sum(LVec*spanTreesVec,1))
        return spanTrees[chosTree]
    

    def treeDirec(self,startNode,previousNode,newTree,tree): 
        # To make sure we are not removing any element when we are in the loop.(For speeding up)
        treeCp = tree.copy()
        for edge in treeCp:
            if startNode in edge:
                for node in edge:
                    # find a new node
                    if(node != startNode and node != previousNode):
                        # remove it from the current tree
                        tree.remove(edge)
                        # correct the directio and add the edge to the new tree
                        newTree.append([startNode,node]);
                        # start with remaining edges 
                        self.treeDirec(node,startNode,newTree,tree);
        return newTree;

    def cycleEqu(self,simplestCycles,treeWithDirec):
        V = self.n + 1;
        # list of edges that are not in spanning tree
        notInSpan =[];
        # equations related to each simple cycle
        O =[];
        for cycle in simplestCycles:
            loopEq = [0 for i in range(V)];
            # list of edges that are not in spanning tree for specific loop
            notInSpanLoop = [] 
            for i in range(len(cycle)):
                # if the direction is the same
                if [cycle[i-1],cycle[i]] in treeWithDirec:
                    loopEq[cycle[i-1]]-=1;
                    loopEq[cycle[i]]+=1;
                # if the direction is not the same
                elif [cycle[i],cycle[i-1]] in treeWithDirec:
                    loopEq[cycle[i-1]]+=1;
                    loopEq[cycle[i]]-=1;
                # find the edges that are not in the spanning tree and store them.
                else:
                    notInSpanLoop.append([cycle[i-1],cycle[i]]);
            notInSpan.append(notInSpanLoop)
            O.append(loopEq[1:]);
            
        return O,notInSpan;


    def giveJJEq(self,O,notInSpan):
        # This function gives:
        # JJEj -- list of JJ energies
        # JJEq -- list of JJ equations which specifies the displacement operators
        # JJexcite -- list of JJ that are connected to external fluxes
        
        JJEj = [];
        JJEq = [];
        JJExcite = [];
		
		# the clean form of notInSpan that the list does not have the edges that are repeated.
        notInSpanCleaned =[]

        # clean the notInSpan
        for listEdge in notInSpan:
            for edge in listEdge:
                edgeSwap = [edge[1]] + [edge[0]]
                if edge not in notInSpanCleaned:
                	if edgeSwap not in notInSpanCleaned:
	                    notInSpanCleaned.append(edge);


        notInSpanVec = self.edgeListToVec(notInSpanCleaned);

        # The list JJs that are not in Spanning Tree(by doing and operation)
        JJNotInSpan = list(np.array(notInSpanVec) & np.array(self.JJ).astype(bool));
        
        # The list JJs that are in the spanning Tree(by doing xor operation)
        JJInSpan = list(np.array(JJNotInSpan) ^ np.array(self.JJ).astype(bool))


        # dealing with the case of cicuit with only two node 
        if(len(JJInSpan)==1 and JJInSpan[0].size>1):
            JJInSpan=[1]

        V = self.n+1

        # List of not in span edges that are proceessed
        processedEdges = [];
        processedEquationIndex = [];


        # index for rotating through the notInSpan inside the while loop 
        i0 = 0
        while(len(processedEdges) < len(notInSpan)):  
            # these while loop goes through the notInSpan to find the equation for each edge 
            # that is not in the spanning tree.

            # checkProcess is a list. It tells us that each edge in notInSpan list is processed or not.
            checkProcess = []

            equationIndex = []

            for edge in notInSpan[i0]:

            	edgeSwap = [edge[1]] + [edge[0]]
                # check if the edge is processed add 1 if not add 0 to the checkProcess list.
            	if edge in processedEdges:	
            		checkProcess.append(1)
            	elif edgeSwap in processedEdges:
            		checkProcess.append(1)
            	else:
            		checkProcess.append(0)

            if( (len(checkProcess) - sum(checkProcess)) == 1):
            	# It's time to save that edge and equation related to that edge
            	processedEdges.append(notInSpan[i0][checkProcess.index(0)]);
            	equationIndex.append(i0)

            	# Add the other equation to the related edge
            	for i1 in range(len(checkProcess)):
            		if(checkProcess[i1] == 1):
            			edgeX = notInSpan[i0][i1]
            			edgeSwapX = [edgeX[1]] + [edgeX[0]]

            			if edgeX in processedEdges:
            				index = processedEdges.index(edgeX);
            			elif edgeSwapX in processedEdges:
            				index = processedEdges.index(edgeSwapX);

            			for element in processedEquationIndex[index]:
            				equationIndex.append(element);

            processedEquationIndex.append(equationIndex);

            # check that counter does not exceed the notInSpan length
            i0 += 1;
            if(i0 == len(notInSpan) ):
            	i0 = 0;  

        for i0 in range(len(processedEdges)):
            edge = processedEdges[i0]
            edgeSwap = [edge[1]] + [edge[0]]
            if edge in self.graph:
                index = self.graph.index(edge)
            elif edgeSwap in self.graph:
                index = self.graph.index(edgeSwap)

            if(self.JJ[index] != 0):
                if(isinstance(self.JJ[index],list)):
                    indParall = self.indExtParall(self.graph[index])
                    JJEj.append(self.JJ[index]);
                    JJExcite.append([processedEquationIndex[i0]]+[indParall])

                    equation = np.zeros(self.n)
                    for index2 in processedEquationIndex[i0]:
                        equation += np.array(O[index2]);

                    JJEq.append(list(equation))                    
                else:
                    JJEj.append(self.JJ[index]);
                    JJExcite.append(processedEquationIndex[i0])

                    equation = np.zeros(self.n)
                    for index2 in processedEquationIndex[i0]:
                        equation += np.array(O[index2]);

                    JJEq.append(list(equation))
                         
        for i0 in range(self.E):
            # if JJ is in spanning tree
            if(JJInSpan[i0] != 0):         
                Eq = [0 for i in range(V)];
                edge = self.graph[i0];
                i1 , i2 = edge;
                if(i1==0 or i2 ==0):
                    Eq[i1 + i2]+=1;
                else:
                    Eq[i1]+=1;
                    Eq[i2]-=1;
                    
                JJEj.append(self.JJ[i0]);
                JJEq.append(Eq[1:]);
                if(isinstance(self.JJ[i0],list)):
                    indParall = self.indExtParall(self.graph[i0]);
                    JJExcite.append([0,indParall]);
                else:
                    JJExcite.append(0);
      
        return JJEj , JJEq , JJExcite

    def configure(self):
        # This function connects all the above functions and find the needed coefficient to describe the Hamiltonian
        
        # 200226 Last Update: In this part I assumed that we can diogonalize the LC part of the Hamiltonian.

        self.cleanUpLC();

        self.lRotated, self.cInvRotated , S = self.buildDiag()

        # vector of frequencies
        self.omega = np.sqrt(np.diag(self.cInvRotated)*np.diag(self.lRotated))

        # finding the simplest cycles and spanTree with direction:
        cycles = self.findAllCycles();
        simplestCycles = self.giveSimplestCycles(cycles);

        # store the simplestCycles
        self.graphSt.simplestCycles = simplestCycles;

		# span tree with maximum inductors and corrected direction
        spanTree = self.giveMaxInd() 
        treeWithDirec = self.treeDirec(0,None,[],spanTree.copy())

        # store the spanningTree
        self.graphSt.spanningTree = treeWithDirec;

        # finding the edges that are not in the spanning tree and KVL related to each simplest loop
        O, notInSpan = self.cycleEqu(simplestCycles,treeWithDirec) 

        # ceofficient needed to write JJ Hamiltonian
        self.JJEj , self.JJEq , self.JJExcite = self.giveJJEq(O,notInSpan)

        # check if we can use this algorithm to solve the circuit and change c inorder to handel
        # flux coordinates inside the cosine in case of singularity
        self.correctC(np.array(self.JJEq)@S)

        # JJEq in rotated frame
        self.JJEqRotated = np.array(self.JJEq) @ S


    def setExcitation(self,external):
        self.JJExt = external;

    def setPhi(self):
        for el in self.JJExt:
            _ , phi = el;
            self.phiExt.append(phi); 

    def indExtParall(self,edgeParall):
        for i in range(len(self.JJExt)):
            edge, _  = self.JJExt[i];
            if(edge == edgeParall):
                return i ;


    def numOp(self,modeNum):
    	# This function gives the Number operator in mode = modeNum

        for i in range(self.n):
            if( i == 0 and i != modeNum):
                num = q.qeye(self.m);
            elif( i == 0 and i == modeNum):
                if (self.omega[i] == 0):
                    num = q.charge((self.m-1)/2)
                else:
                    num = q.num(self.m);

            if( i!= 0 and i != modeNum):
                num = q.tensor(num,q.qeye(self.m))
            elif(i != 0 and i == modeNum):
                if(self.omega[i] == 0):
                    num = q.tensor(num,q.charge((self.m-1)/2))
                else:
                    num = q.tensor(num,q.num(self.m))

        return num

    def intOp(self,mode1,mode2):
        # give the interaction operator between each mode:

        disMinCreat = q.destroy(self.m) - q.create(self.m);
        num = q.num(self.m)
        charge = q.charge((self.m-1)/2)

        # identiry operator for each mode 
        I = q.qeye(self.m);

        for i in range(self.n):

            if( i==0 and (i == mode1 or i == mode2) ):
                if(self.omega[i]==0):
                    intr = (2*e)*charge;
                else:
                    coef = (-1j*np.sqrt(1/2*np.sqrt(self.lRotated[i,i]/self.cInvRotated[i,i])))
                    intr = coef*disMinCreat;
            elif(i == 0):
                intr = I

            if( i!=0 and (i == mode1 or i == mode2)):
                if(self.omega[i]==0):
                    intr = q.tensor(intr,(2*e)*charge)
                else:
                    coef = (-1j*np.sqrt(1/2*np.sqrt(self.lRotated[i,i]/self.cInvRotated[i,i])))
                    intr = q.tensor(intr,coef*disMinCreat);
            elif(i!=0):
                intr = q.tensor(intr,I)

        return intr 

    def giveLCHamil(self):
        # function that gives the Hamiltionian of the LC part of the circuits

        HLC = 0;

        for i in range(self.n):
            for j in range(self.n):
                if(i==j):
                    if(self.omega[i]==0):
                        HLC += (1/2* self.cInvRotated[i,i]*(2*e)**2)/hbar*self.numOp(i)*self.numOp(i)
                    else:
                        HLC += self.omega[i]*self.numOp(i)
                elif(j>i):
                    if(self.cInvRotated[i,j] != 0):
                        HLC += self.cInvRotated[i,j]/hbar*self.intOp(i,j)

        return HLC

    def giveJJHamil(self,phiInput):
        # function that gives the Hamiltionian of the JJ of the circuits. with external charge of phi:

        # List of individual JJ Hamiltonian
        HJJList = [];
        HJJCheckList = []

        # create charge diplacement operator for each mode
        d = np.zeros((self.m,self.m))
        for i in range(self.m):
            for j in range(self.m):
                if(j-1==i):
                    d[i,j] = 1;
        d = q.Qobj(d);
 
        # identity operator for each mode
        I = q.qeye(self.m)

        for i in range(len(self.JJEj)):

            # tensor multiplication of displacement operator for JJ Hamiltonian
            for j in range(self.n):
                if(j == 0 and self.omega[j]==0 and self.JJEqRotated[i,j] == 0):
                    H = I;

                elif(j == 0 and self.omega[j]==0 and self.JJEqRotated[i,j] != 0):
                    H = d;

                elif(j == 0 and self.omega[j]!=0):
                    alpha = 2*np.pi/Phi0*1j*np.sqrt(hbar/2*np.sqrt(self.cInvRotated[j,j]/self.lRotated[j,j]))\
                    *self.JJEqRotated[i,j]
                    H = q.displace(self.m,alpha)


                if(j!=0 and self.omega[j] == 0 and self.JJEqRotated[i,j] == 0):
                    H = q.tensor(H,I);

                elif(j!=0 and self.omega[j] == 0 and self.JJEqRotated[i,j] != 0):
                    H = q.tensor(H,d);

                elif(j!=0 and self.omega[j] != 0):
                    alpha = 2*np.pi/Phi0*1j*np.sqrt(hbar/2*np.sqrt(self.cInvRotated[j,j]/self.lRotated[j,j]))\
                    *self.JJEqRotated[i,j]
                    H = q.tensor(H,q.displace(self.m,alpha))

            # add external excitation

            # Parallel JJ case
            if(isinstance(self.JJEj[i],list)):
                JJEjIn , JJEjOut = self.JJEj[i]
                JJExcIn, JJExcOut = self.JJExcite[i]
                phiIn = 0;
                PhiOut = 0;
                if(JJExcIn):
                    for ind in JJExcIn:
                        phiIn = phiInput[ind];
                phiOut = phiIn + phiInput[JJExcOut];

                H = np.exp(1j*phiIn)*JJEjIn/2*H + np.exp(1j*phiOut)*JJEjOut/2*H
                H = H + H.dag()

            # single JJ case
            else:
                phi = 0
                if(self.JJExcite[i]):
                    for ind in self.JJExcite[i]:
                        phi += phiInput[ind];
                H  =  np.exp(1j*phi) * self.JJEj[i]/2* H;
                H = H + H.dag();

            HJJList.append(H);
        
        HJJ = sum(HJJList)

        return HJJ

    def solveCircuit(self):
        # function that use qutip package to define number operator and displacement operators to calculate
        # the hailtonian and diogonalize it. At the end,the final eignevalues of the
        # of the total Hamiltonian is stored in the self.HamilEig. 

        HLC = self.giveLCHamil()

        #prepare the external fluxes for calculating the JJ Hamiltonian
        self.setPhi()

        # idex for sweeping over
        indSweep = 0;
        for i in range(len(self.phiExt)):
            if(not isinstance(self.phiExt[i],int)):
                indSweep = i;

        self.HamilEig = np.zeros((self.m**self.n,len(self.phiExt[indSweep])),dtype='complex');


        phiInput = self.phiExt.copy()
        # diogonalize Hamiltonain for sweeping over phiExt
        for i in range(len(self.phiExt[indSweep])):
            
            print(i)
            phiInput[indSweep] = self.phiExt[indSweep][i];

            HJJ = self.giveJJHamil(phiInput)

            H = -0*HJJ + HLC

            # find the eigenvalues of the hamiltonian for each external phi
            eigenValues , eigenVectors = H.eigenstates();

            # store the eigenfunction of the Hamiltonian for each frequency
            self.HamilEig[:,i] = eigenValues

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










